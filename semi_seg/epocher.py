import random
from typing import Union, Tuple

import torch
from deepclustering2.augment.tensor_augment import TensorRandomFlip
from deepclustering2.decorator import FixRandomSeed
from deepclustering2.epoch import _Epocher  # noqa
from deepclustering2.meters2 import EpochResultDict, AverageValueMeter, UniversalDice, MultipleAverageValueMeter, \
    MeterInterface, SurfaceMeter
from deepclustering2.models import Model
from deepclustering2.optim import get_lrs_from_optimizer
from deepclustering2.type import T_loader, T_loss, T_optim
from deepclustering2.utils import class2one_hot, ExceptionIgnorer
from torch import nn
from torch.utils.data import DataLoader

from contrastyou.epocher._utils import preprocess_input_with_single_transformation  # noqa
from contrastyou.epocher._utils import preprocess_input_with_twice_transformation  # noqa
from contrastyou.epocher._utils import write_predict, write_img_target  # noqa
from contrastyou.trainer._utils import ClusterHead  # noqa
from semi_seg._utils import FeatureExtractor


class _num_class_mixin:
    _model: nn.Module

    @property
    def num_classes(self):
        return self._model.num_classes


class EvalEpocher(_num_class_mixin, _Epocher):

    def __init__(self, num_iter: int, model: Union[Model, nn.Module], val_loader: T_loader, sup_criterion: T_loss, cur_epoch=0,
                 device="cpu") -> None:
        assert isinstance(val_loader, DataLoader), \
            f"val_loader should be an instance of DataLoader, given {val_loader.__class__.__name__}."
        super().__init__(model, num_batches=len(val_loader), cur_epoch=cur_epoch, device=device)
        self._num_iter = num_iter
        self._val_loader = val_loader
        self._sup_criterion = sup_criterion

    def _configure_meters(self, meters: MeterInterface) -> MeterInterface:
        C = self.num_classes
        report_axis = list(range(1, C))
        meters.register_meter("loss", AverageValueMeter())
        meters.register_meter("dice", UniversalDice(C, report_axises=report_axis, ))
        return meters

    @torch.no_grad()
    def _run(self, *args, **kwargs) -> Tuple[EpochResultDict, float]:
        self._model.train()
        report_dict = EpochResultDict()
        for i, val_data in zip(self._indicator, self._val_loader):
            val_img, val_target, file_path, _, group = self._unzip_data(val_data, self._device)

            val_img_dims = val_img.shape
            if self._num_iter != 0:
                for ITER in range(self._num_iter):
                    if ITER == 0:
                        Y_0_val = torch.ones(val_img_dims[0], self._model.num_classes, val_img_dims[2], val_img_dims[3]).to(self.device, non_blocking=True)
                        val_concat = torch.cat([Y_0_val, val_img], dim = 1)
                    else:
                        val_concat = torch.cat([val_logits, val_img], dim=1)
                    val_logits = self._model(val_concat)
            else:
                val_logits = self._model(val_img)

            onehot_target = class2one_hot(val_target.squeeze(1), self.num_classes)
            val_loss = self._sup_criterion(val_logits.softmax(1), onehot_target, disable_assert=True)

            self.meters["loss"].add(val_loss.item())
            self.meters["dice"].add(val_logits.max(1)[1], val_target.squeeze(1), group_name=group)
            report_dict = self.meters.tracking_status()
            self._indicator.set_postfix_dict(report_dict)
        return report_dict, self.meters["dice"].summary()["DSC_mean"]

    @staticmethod
    def _unzip_data(data, device):
        image, target, filename, partition, group = preprocess_input_with_single_transformation(data, device)
        return image, target, filename, partition, group


class InferenceEpocher(EvalEpocher):

    def set_save_dir(self, save_dir):
        self._save_dir = save_dir

    def _configure_meters(self, meters: MeterInterface) -> MeterInterface:
        meters = super()._configure_meters(meters)
        C = self.num_classes
        report_axis = list(range(1, C))
        meters.register_meter("hd", SurfaceMeter(C=C, report_axises=report_axis, metername="hausdorff"))
        return meters

    def _run(self, *args, **kwargs) -> Tuple[EpochResultDict, float]:
        self._model.eval()
        assert self._model.training is False, self._model.training
        for i, val_data in zip(self._indicator, self._val_loader):
            val_img, val_target, file_path, _, group = self._unzip_data(val_data, self._device)
            val_logits = self._model(val_img)
            # write image
            write_img_target(val_img, val_target, self._save_dir, file_path)
            write_predict(val_logits, self._save_dir, file_path, )
            onehot_target = class2one_hot(val_target.squeeze(1), self.num_classes)

            val_loss = self._sup_criterion(val_logits.softmax(1), onehot_target, disable_assert=True)

            self.meters["loss"].add(val_loss.item())
            self.meters["dice"].add(val_logits.max(1)[1], val_target.squeeze(1), group_name=group)
            with ExceptionIgnorer(RuntimeError):
                self.meters["hd"].add(val_logits.max(1)[1], val_target.squeeze(1))
            report_dict = self.meters.tracking_status()
            self._indicator.set_postfix_dict(report_dict)
        return report_dict, self.meters["dice"].summary()["DSC_mean"]


class TrainEpocher(_num_class_mixin, _Epocher):

    def __init__(self, model: Union[Model, nn.Module], optimizer: T_optim, labeled_loader: T_loader,
                 unlabeled_loader: T_loader, sup_criterion: T_loss, reg_weight: float, num_batches: int, cur_epoch=0,
                 device="cpu", feature_position=None, feature_importance=None) -> None:
        super().__init__(model, num_batches=num_batches, cur_epoch=cur_epoch, device=device)
        self._optimizer = optimizer
        self._labeled_loader = labeled_loader
        self._unlabeled_loader = unlabeled_loader
        self._sup_criterion = sup_criterion
        self._reg_weight = reg_weight
        self._affine_transformer = TensorRandomFlip(axis=[1, 2], threshold=0.8)
        assert isinstance(feature_position, list) and isinstance(feature_position[0], str), feature_position
        assert isinstance(feature_importance, list) and isinstance(feature_importance[0],
                                                                   (int, float)), feature_importance
        self._feature_position = feature_position
        self._feature_importance = feature_importance

    def _configure_meters(self, meters: MeterInterface) -> MeterInterface:
        C = self.num_classes
        report_axis = list(range(1, C))
        meters.register_meter("lr", AverageValueMeter())
        meters.register_meter("sup_loss", AverageValueMeter())
        meters.register_meter("reg_loss", AverageValueMeter())
        meters.register_meter("sup_dice", UniversalDice(C, report_axises=report_axis, ))
        return meters

    def _run(self, *args, **kwargs) -> EpochResultDict:
        self.meters["lr"].add(get_lrs_from_optimizer(self._optimizer)[0])
        self._model.train()
        assert self._model.training, self._model.training
        report_dict = {}
        with FeatureExtractor(self._model, self._feature_position) as self._fextractor:
            for i, labeled_data, unlabeled_data in zip(self._indicator, self._labeled_loader, self._unlabeled_loader):
                seed = random.randint(0, int(1e7))
                labeled_image, labeled_target, labeled_filename, _, label_group = \
                    self._unzip_data(labeled_data, self._device)
                unlabeled_image, unlabeled_target, *_ = self._unzip_data(unlabeled_data, self._device)
                with FixRandomSeed(seed):
                    unlabeled_image_tf = torch.stack([self._affine_transformer(x) for x in unlabeled_image], dim=0)
                assert unlabeled_image_tf.shape == unlabeled_image.shape, \
                    (unlabeled_image_tf.shape, unlabeled_image.shape)

                predict_logits = self._model(torch.cat([labeled_image, unlabeled_image, unlabeled_image_tf], dim=0))
                label_logits, unlabel_logits, unlabel_tf_logits = \
                    torch.split(
                        predict_logits,
                        [len(labeled_image), len(unlabeled_image), len(unlabeled_image_tf)],
                        dim=0
                    )
                with FixRandomSeed(seed):
                    unlabel_logits_tf = torch.stack([self._affine_transformer(x) for x in unlabel_logits], dim=0)
                assert unlabel_logits_tf.shape == unlabel_tf_logits.shape, \
                    (unlabel_logits_tf.shape, unlabel_tf_logits.shape)
                # supervised part
                onehot_target = class2one_hot(labeled_target.squeeze(1), self.num_classes)
                sup_loss = self._sup_criterion(label_logits.softmax(1), onehot_target)
                # regularized part
                reg_loss = self.regularization(
                    unlabeled_tf_logits=unlabel_tf_logits,
                    unlabeled_logits_tf=unlabel_logits_tf,
                    seed=seed,
                    unlabeled_image=unlabeled_image,
                    unlabeled_image_tf=unlabeled_image_tf,
                )
                total_loss = sup_loss + self._reg_weight * reg_loss
                # gradient backpropagation
                self._optimizer.zero_grad()
                total_loss.backward()
                self._optimizer.step()
                # recording can be here or in the regularization method
                with torch.no_grad():
                    self.meters["sup_loss"].add(sup_loss.item())
                    self.meters["sup_dice"].add(label_logits.max(1)[1], labeled_target.squeeze(1),
                                                group_name=label_group)
                    self.meters["reg_loss"].add(reg_loss.item())
                    report_dict = self.meters.tracking_status()
                    self._indicator.set_postfix_dict(report_dict)
        return report_dict

    @staticmethod
    def _unzip_data(data, device):
        (image, target), _, filename, partition, group = \
            preprocess_input_with_twice_transformation(data, device)
        return image, target, filename, partition, group

    def regularization(self, *args, **kwargs):
        return torch.tensor(0, dtype=torch.float, device=self._device)


class FullEpocher(_num_class_mixin, _Epocher):

    def __init__(self, num_iter: int, model: Union[Model, nn.Module], optimizer: T_optim, labeled_loader: T_loader,
                 sup_criterion: T_loss, cur_epoch=0,
                 device="cpu") -> None:
        super().__init__(model = model, num_batches = len(labeled_loader), cur_epoch = cur_epoch, device = device)
        self._num_iter = num_iter
        self._optimizer = optimizer
        self._labeled_loader = labeled_loader
        self._sup_criterion = sup_criterion

    def _configure_meters(self, meters: MeterInterface) -> MeterInterface:
        C = self.num_classes
        report_axis = list(range(1, C))
        meters.register_meter("lr", AverageValueMeter())
        meters.register_meter("sup_loss", AverageValueMeter())
        meters.register_meter("sup_dice", UniversalDice(C, report_axises=report_axis, ))
        return meters

    def _run(self, *args, **kwargs) -> EpochResultDict:
        self.meters["lr"].add(get_lrs_from_optimizer(self._optimizer)[0])
        self._model.train()
        assert self._model.training, self._model.training
        report_dict = {}

        for i, labeled_data in zip(self._indicator, self._labeled_loader):
            labeled_image, labeled_target, labeled_filename, _, label_group = \
                self._unzip_data(labeled_data, self._device)
                #(5, 1, 224, 224) -> labeled_image.shape
            labeled_image_dims = labeled_image.shape
            if self._num_iter != 0:
                for ITER in range(self._num_iter):
                    if ITER == 0:
                        Y_0 = torch.ones(labeled_image_dims[0],
                                          self._model.num_classes,
                                          labeled_image_dims[2], labeled_image_dims[3]).to(self.device, non_blocking = True)
                        concat = torch.cat([Y_0, labeled_image], dim = 1)
                    else:
                        concat = torch.cat([predict_logits, labeled_image], dim=1)
                    predict_logits = self._model(concat)
            else:
                predict_logits = self._model(labeled_image)
                #(5, 4, 224, 224) -> predict_logits.shape

            # supervised part
            onehot_target = class2one_hot(labeled_target.squeeze(1), self.num_classes)
            sup_loss = self._sup_criterion(predict_logits.softmax(1), onehot_target)

            total_loss = sup_loss
            # gradient backpropagation
            self._optimizer.zero_grad()
            total_loss.backward()
            self._optimizer.step()
            # recording can be here or in the regularization method
            with torch.no_grad():
                self.meters["sup_loss"].add(sup_loss.item())
                self.meters["sup_dice"].add(predict_logits.max(1)[1], labeled_target.squeeze())
                report_dict = self.meters.tracking_status()
                self._indicator.set_postfix_dict(report_dict)
        return report_dict

    @staticmethod
    def _unzip_data(data, device):
        (image, target), _, filename, partition, group = \
            preprocess_input_with_twice_transformation(data, device)
        return image, target, filename, partition, group