"""
            ls = []
            for j in range(len(val_img)):
                for k in range(len(seg)):
                    if file_path[j] == seg[k][:-4]:
                        pred = np.load(mask_path+seg[k])
                        pred = torch.from_numpy(pred).cuda()
                        ls.append(torch.cat([pred, val_img[j]]))
            new_input = torch.stack(ls)

"""
# TODO:With loss after each iter and produce gif
# TODO:With loss at end

import numpy as np
import random
import torch
from contrastyou.epocher._utils import preprocess_input_with_single_transformation  # noqa
from contrastyou.epocher._utils import preprocess_input_with_twice_transformation
from contrastyou.epocher._utils import preprocess_input_with_twice_transformation_for_exp2  # noqa
from contrastyou.epocher._utils import write_predict, write_img_target  # noqa
from contrastyou.trainer._utils import ClusterHead  # noqa
from deepclustering2.augment.tensor_augment import TensorRandomFlip
from deepclustering2.decorator import FixRandomSeed
from deepclustering2.epoch import _Epocher  # noqa
from deepclustering2.meters2 import EpochResultDict, AverageValueMeter, UniversalDice, MeterInterface, SurfaceMeter
from deepclustering2.models import Model
from deepclustering2.optim import get_lrs_from_optimizer
from deepclustering2.type import T_loader, T_loss, T_optim
from deepclustering2.utils import class2one_hot, ExceptionIgnorer
from semi_seg._utils import FeatureExtractor
from torch import nn
from torch.utils.data import DataLoader
from typing import Union, Tuple


class _num_class_mixin:
    _model: nn.Module

    @property
    def num_classes(self):
        return self._model.num_classes


class IterativeEvalEpocher(_num_class_mixin, _Epocher):

    def __init__(self, alpha: float, num_iter: int, model: Union[Model, nn.Module], val_loader: T_loader,
                 sup_criterion: T_loss, cur_epoch=0, device="cpu") -> None:
        assert isinstance(val_loader, DataLoader), \
            f"val_loader should be an instance of DataLoader, given {val_loader.__class__.__name__}."
        super().__init__(model, num_batches=len(val_loader), cur_epoch=cur_epoch, device=device)
        self._alpha = alpha
        self._num_iter = num_iter
        self._val_loader = val_loader
        self._sup_criterion = sup_criterion

    def _configure_meters(self, meters: MeterInterface) -> MeterInterface:
        C = self.num_classes
        report_axis = list(range(1, C))
        meters.register_meter("loss", AverageValueMeter())
        meters.register_meter("dice", UniversalDice(C, report_axises=report_axis, ))
        num_iters = self._num_iter
        assert num_iters >= 1
        for i in range(num_iters):
            meters.register_meter(f"itrdice_{i}", UniversalDice(C, report_axises=report_axis, ))
            meters.register_meter(f"itrloss_{i}", AverageValueMeter())
        return meters

    @torch.no_grad()
    def _run(self, *args, **kwargs) -> Tuple[EpochResultDict, float]:
        self._model.train()
        report_dict = EpochResultDict()
        save_dir_base = '/home/saksham/Iterative-learning/.data/ACDC_contrast/evolution_val/'

        for i, val_data in zip(self._indicator, self._val_loader):
            val_img, val_target, file_path, _, group = self._unzip_data(val_data, self._device)
            val_img_dims = val_img.shape
            # write_img_target(val_img[:,-1,:,:].reshape(val_img_dims[0], 1, 224, 224), val_target, save_dir_base, file_path)
            alpha = self._alpha
            # uniform_dis = torch.ones(val_img_dims[0], self._model.num_classes, *val_img_dims[2:],
            #                         device=self.device).softmax(1)
            aggregated_simplex = None
            loss_list = []
            onehot_target = class2one_hot(val_target.squeeze(1), self.num_classes)
            for ITER in range(self._num_iter):
                # concat = torch.cat([aggregated_simplex, uniform_dis], dim=1)
                # concat = torch.cat([val_img, aggregated_simplex], dim=1)
                if ITER == 0:
                    concat = val_img
                else:
                    concat = torch.cat([val_img[:, -1, :, :].reshape(val_img_dims[0], 1, 224, 224),
                                        aggregated_simplex], dim=1)
                cur_predict = self._model(concat).softmax(1)

                if ITER == 0:
                    aggregated_simplex = cur_predict
                    save_dir = save_dir_base + str(self._cur_epoch) + '/iter0/'
                    # write_predict(cur_predict, save_dir, file_path)
                else:
                    aggregated_simplex = alpha * aggregated_simplex + (1 - alpha) * cur_predict
                    save_dir = save_dir_base + str(self._cur_epoch) + '/iter1/'
                    # write_predict(cur_predict, save_dir, file_path)

                iter_loss = self._sup_criterion(aggregated_simplex, onehot_target,
                                                            disable_assert=True)

                with torch.no_grad():  # todo:  changer the evaler like the iterative epocher
                    self.meters[f"itrdice_{ITER}"].add(cur_predict.max(1)[1], val_target.squeeze())
                    self.meters[f"itrloss_{ITER}"].add(iter_loss.item())
                loss_list.append(iter_loss)

            if len(loss_list) == 0:
                raise RuntimeError("no loss there.")
            total_loss = sum(loss_list)

            self.meters["loss"].add(total_loss.item())
            self.meters["dice"].add(aggregated_simplex.max(1)[1], val_target.squeeze(1), group_name=group)
            report_dict = self.meters.tracking_status()
            self._indicator.set_postfix_dict(report_dict)
        return report_dict, self.meters["dice"].summary()["DSC_mean"]

    @staticmethod
    def _unzip_data(data, device):
        image, target, filename, partition, group = preprocess_input_with_single_transformation(data, device)
        return image, target, filename, partition, group


class FullEvalEpocher(_num_class_mixin, _Epocher):

    def __init__(self, model: Union[Model, nn.Module], val_loader: T_loader, sup_criterion: T_loss,
                 test_loader: T_loader = None, id=1,
                 cur_epoch=0, device="cpu") -> None:
        if id == 1:
            loader = val_loader
        else:
            loader = test_loader
        assert isinstance(val_loader, DataLoader), \
            f"val_loader should be an instance of DataLoader, given {val_loader.__class__.__name__}."
        super().__init__(model, num_batches=len(loader), cur_epoch=cur_epoch, device=device)

        self._test_loader = test_loader
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
        # mask_path = PROJECT_PATH + '/.data/ACDC_contrast/val/val_masks/'
        # seg = os.listdir(mask_path)
        for i, val_data in zip(self._indicator, self._val_loader):
            val_img, val_target, file_path, _, group = self._unzip_data(val_data, self._device)
            val_img[:, :4] = torch.zeros_like(val_img[:, :4])
            # insert block at the top here
            predict_logits = self._model(val_img).softmax(1)

            onehot_target = class2one_hot(val_target.squeeze(1), self.num_classes)
            val_loss = self._sup_criterion(predict_logits, onehot_target, disable_assert=True)

            self.meters["loss"].add(val_loss.item())
            self.meters["dice"].add(predict_logits.max(1)[1], val_target.squeeze(1), group_name=group)
            report_dict = self.meters.tracking_status()
            self._indicator.set_postfix_dict(report_dict)
        return report_dict, self.meters["dice"].summary()["DSC_mean"]

    @staticmethod
    def _unzip_data(data, device):
        image, target, filename, partition, group = preprocess_input_with_single_transformation(data, device)
        return image, target, filename, partition, group


class InferenceEpocher(IterativeEvalEpocher):

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
            val_logits = self._model(val_img).softmax(1)
            # write image
            write_img_target(val_img, val_target, self._save_dir, file_path)
            for i in range(len(val_img)):
                filename = self._save_dir + '/preds/' + file_path[i] + '.png'
                arr = val_logits[i].cpu().detach().numpy()
                np.save(filename, arr)
            # write_predict(val_logits, self._save_dir, file_path, )

            onehot_target = class2one_hot(val_target.squeeze(1), self.num_classes)

            val_loss = self._sup_criterion(val_logits.softmax(1), onehot_target, disable_assert=True)

            self.meters["loss"].add(val_loss.item())
            self.meters["dice"].add(val_logits.max(1)[1], val_target.squeeze(1), group_name=group)
            with ExceptionIgnorer(RuntimeError):
                self.meters["hd"].add(val_logits.max(1)[1], val_target.squeeze(1))
            report_dict = self.meters.tracking_status()
            self._indicator.set_postfix_dict(report_dict)
        return report_dict, self.meters["dice"].summary()["DSC_mean"]

    def _unzip_data(self, data, device):
        image, target, filename, partition, group = preprocess_input_with_single_transformation(data, device)
        return image, target, filename, partition, group


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


class IterativeEpocher(_num_class_mixin, _Epocher):

    def __init__(self, alpha: float, num_iter: int, model: Union[Model, nn.Module], optimizer: T_optim,
                 labeled_loader: T_loader,
                 sup_criterion: T_loss, cur_epoch=0, num_batches=100,
                 device="cpu") -> None:
        super().__init__(model=model, num_batches=num_batches, cur_epoch=cur_epoch, device=device)
        self._alpha = alpha
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
        num_iters = self._num_iter
        assert num_iters >= 1
        for i in range(num_iters):
            meters.register_meter(f"itrdice_{i}", UniversalDice(C, report_axises=report_axis, ))
            meters.register_meter(f"itrloss_{i}", AverageValueMeter())
        return meters

    def _run(self, *args, **kwargs) -> EpochResultDict:
        self.meters["lr"].add(get_lrs_from_optimizer(self._optimizer)[0])
        self._model.train()
        assert self._model.training, self._model.training
        report_dict = {}

        for i, labeled_data in zip(self._indicator, self._labeled_loader):
            labeled_image, labeled_target, labeled_filename, _, label_group = \
                self._unzip_data(labeled_data, self._device)
            # (5, 1, 224, 224) -> labeled_image.shape
            labeled_image_dims = labeled_image.shape
            alpha = self._alpha
            aggregated_simplex = None
            onehot_target = class2one_hot(labeled_target.squeeze(1), self.num_classes)
            loss_list = []
            for ITER in range(self._num_iter):
                if ITER == 0:
                    concat = labeled_image
                else:
                    concat = torch.cat([labeled_image[:, -1, :, :].reshape(labeled_image_dims[0], 1, 224, 224),
                                        aggregated_simplex], dim=1)
                cur_predict = self._model(concat).softmax(1)

                if ITER == 0:
                    aggregated_simplex = cur_predict
                else:
                    aggregated_simplex = alpha * aggregated_simplex.detach() + (
                        1 - alpha) * cur_predict  # todo: try to play with this
                cur_loss = self._sup_criterion(aggregated_simplex, onehot_target)
                loss_list.append(cur_loss)

                with torch.no_grad():
                    self.meters[f"itrdice_{ITER}"].add(cur_predict.max(1)[1], labeled_target.squeeze())
                    self.meters[f"itrloss_{ITER}"].add(cur_loss.item())

            # supervised part
            if len(loss_list) == 0:
                raise RuntimeError("no loss there.")
            total_loss = sum(loss_list)
            # gradient backpropagation

            self._optimizer.zero_grad()
            total_loss.backward()
            self._optimizer.step()
            # recording can be here or in the regularization method
            with torch.no_grad():
                self.meters["sup_loss"].add(total_loss.item())
                self.meters["sup_dice"].add(aggregated_simplex.max(1)[1], labeled_target.squeeze())
                report_dict = self.meters.tracking_status()
                self._indicator.set_postfix_dict(report_dict)
        return report_dict

    @staticmethod
    def _unzip_data(data, device):
        image, target, filename, partition, group = \
            preprocess_input_with_single_transformation(data, device)
        return image, target, filename, partition, group

class InverseIterativeEvalEpocher(_num_class_mixin, _Epocher):

    def __init__(self, memory_bank, alpha: float, num_iter: int, model: Union[Model, nn.Module], val_loader: T_loader,
                 sup_criterion: T_loss, cur_epoch=0, device="cpu") -> None:
        assert isinstance(val_loader, DataLoader), \
            f"val_loader should be an instance of DataLoader, given {val_loader.__class__.__name__}."
        super().__init__(model, num_batches=len(val_loader), cur_epoch=cur_epoch, device=device)
        self._alpha = alpha
        self._mem_bank = memory_bank
        self._num_iter = num_iter
        self._val_loader = val_loader
        self._sup_criterion = sup_criterion

    def _configure_meters(self, meters: MeterInterface) -> MeterInterface:
        C = self.num_classes
        report_axis = list(range(1, C))
        num_iters = self._num_iter
        assert num_iters >= 1
        for i in range(num_iters):
            meters.register_meter(f"itrdice_{i}", UniversalDice(C, report_axises=report_axis, ))
            meters.register_meter(f"itrloss_{i}", AverageValueMeter())
        return meters

    @torch.no_grad()
    def _run(self, *args, **kwargs) -> Tuple[EpochResultDict, float]:
        self._model.train()
        report_dict = EpochResultDict()
        save_dir_base = '/home/saksham/Iterative-learning/.data/ACDC_contrast/evolution_val/'

        for ITER in range(self._num_iter):
            for i, val_data in zip(self._indicator, self._val_loader):
                val_img, val_target, file_path, _, group = self._unzip_data(val_data, self._device)
                val_img_dims = val_img.shape
                # write_img_target(val_img[:,-1,:,:].reshape(val_img_dims[0], 1, 224, 224), val_target, save_dir_base, file_path)
                alpha = self._alpha
                onehot_target = class2one_hot(val_target.squeeze(1), self.num_classes)

                # concat = torch.cat([aggregated_simplex, uniform_dis], dim=1)
                # concat = torch.cat([val_img, aggregated_simplex], dim=1)
                if ITER == 0:
                    concat = val_img
                else:
                    cur_batch_prev_pred = []
                    for file in file_path:
                        cur_batch_prev_pred.append(self._mem_bank[file])
                    cur_batch_stack = torch.stack(cur_batch_prev_pred)
                    concat = torch.cat([cur_batch_stack,
                                        val_img[:, -1, :, :].reshape(val_img_dims[0], 1, 224, 224)], dim=1)
                    assert None not in (concat.cpu().__array__())
                cur_predict = self._model(concat).softmax(1)

                if ITER == 0:
                    aggregated_simplex = cur_predict
                    save_dir = save_dir_base + str(self._cur_epoch) + '/iter0/'
                    # write_predict(cur_predict, save_dir, file_path)
                else:
                    aggregated_simplex = alpha * cur_batch_stack.detach() + (1 - alpha) * cur_predict
                    save_dir = save_dir_base + str(self._cur_epoch) + '/iter1/'
                    # write_predict(cur_predict, save_dir, file_path)
                for j in range(val_img.shape[0]):
                    self._mem_bank[file_path[j]] = aggregated_simplex[j].detach()

                iter_loss = self._sup_criterion(aggregated_simplex, onehot_target,
                                                            disable_assert=True)

            self.meters[f"itrloss_{ITER}"].add(iter_loss.item())
            self.meters[f"itrdice_{ITER}"].add(aggregated_simplex.max(1)[1], val_target.squeeze(),
                                               group_name=group)
            report_dict = self.meters.tracking_status()
            self._indicator.set_postfix_dict(report_dict)
        return report_dict, self.meters[f"itrdice_{ITER}"].summary()["DSC_mean"]

    @staticmethod
    def _unzip_data(data, device):
        image, target, filename, partition, group = preprocess_input_with_single_transformation(data, device)
        return image, target, filename, partition, group

class InverseIterativeEpocher(_num_class_mixin, _Epocher):

    def __init__(self, memory_bank, alpha: float, num_iter, model: Union[Model, nn.Module], optimizer: T_optim,
                 labeled_loader: T_loader,
                 sup_criterion: T_loss, cur_epoch=0, num_batches=100,
                 device="cpu") -> None:
        super().__init__(model=model, num_batches=num_batches, cur_epoch=cur_epoch, device=device)
        self._alpha = alpha
        self._num_iter = num_iter
        self._mem_bank = memory_bank
        self._optimizer = optimizer
        self._labeled_loader = labeled_loader
        self._sup_criterion = sup_criterion

    def _configure_meters(self, meters: MeterInterface) -> MeterInterface:
        C = self.num_classes
        report_axis = list(range(1, C))
        meters.register_meter("lr", AverageValueMeter())
        #meters.register_meter("sup_loss", AverageValueMeter())
        #meters.register_meter("sup_dice", UniversalDice(C, report_axises=report_axis, ))
        num_iters = self._num_iter
        assert num_iters >= 1
        for i in range(num_iters):
            meters.register_meter(f"itrdice_{i}", UniversalDice(C, report_axises=report_axis, ))
            meters.register_meter(f"itrloss_{i}", AverageValueMeter())
        return meters

    def _run(self, *args, **kwargs) -> EpochResultDict:
        self.meters["lr"].add(get_lrs_from_optimizer(self._optimizer)[0])
        self._model.train()
        assert self._model.training, self._model.training
        report_dict = {}

        for ITER in range(self._num_iter):

            for i, labeled_data in zip(self._indicator, self._labeled_loader):
                labeled_image, labeled_target, labeled_filename, _, label_group = \
                    self._unzip_data(labeled_data, self._device)
                # (5, 1, 224, 224) -> labeled_image.shape
                labeled_image_dims = labeled_image.shape
                alpha = self._alpha
                onehot_target = class2one_hot(labeled_target.squeeze(1), self.num_classes)

                if ITER == 0:
                    concat = labeled_image
                else:
                    cur_batch_prev_pred = []
                    for file in labeled_filename:
                        cur_batch_prev_pred.append(self._mem_bank[file])
                    cur_batch_stack = torch.stack(cur_batch_prev_pred)

                    concat = torch.cat([cur_batch_stack,
                                        labeled_image[:, -1, :, :].reshape(labeled_image_dims[0], 1, 224, 224)], dim=1)

                cur_predict = self._model(concat).softmax(1)

                if ITER == 0:
                    aggregated_simplex = cur_predict
                else:
                    aggregated_simplex = alpha * cur_batch_stack.detach() + (
                        1 - alpha) * cur_predict  # todo: try to play with this
                for j in range(labeled_image.shape[0]):
                    self._mem_bank[labeled_filename[j]] = aggregated_simplex[j].detach()

                cur_loss = self._sup_criterion(aggregated_simplex, onehot_target,
                                               disable_assert = True)

            # supervised part

            # gradient backpropagation
                total_loss = cur_loss
                self._optimizer.zero_grad()
                cur_loss.backward(retain_graph = True)
                self._optimizer.step()
                # recording can be here or in the regularization method
                with torch.no_grad():
                    self.meters[f"itrloss_{ITER}"].add(total_loss.item())
                    self.meters[f"itrdice_{ITER}"].add(aggregated_simplex.max(1)[1], labeled_target.squeeze(),
                                                        group_name = label_group)
                    report_dict = self.meters.tracking_status()
                    self._indicator.set_postfix_dict(report_dict)
        return report_dict

    @staticmethod
    def _unzip_data(data, device):
        image, target, filename, partition, group = \
            preprocess_input_with_single_transformation(data, device)
        return image, target, filename, partition, group


class FullEpocher(_num_class_mixin, _Epocher):

    def __init__(self, model: Union[Model, nn.Module], optimizer: T_optim, labeled_loader: T_loader,
                 sup_criterion: T_loss, cur_epoch=0, num_batches=300,
                 device="cpu") -> None:
        super().__init__(model=model, num_batches=num_batches, cur_epoch=cur_epoch, device=device)

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
            labeled_image[:, :4] = torch.zeros_like(labeled_image[:, :4])

            predict_logits = self._model(labeled_image)  # bug 1

            onehot_target = class2one_hot(labeled_target.squeeze(1), self.num_classes)
            sup_loss = self._sup_criterion(predict_logits.softmax(1), onehot_target)

            # supervised part
            total_loss = sup_loss
            # gradient backpropagation
            self._optimizer.zero_grad()
            total_loss.backward()
            self._optimizer.step()
            # recording can be here or in the regularization method
            with torch.no_grad():
                self.meters["sup_loss"].add(sup_loss.item())
                self.meters["sup_dice"].add(predict_logits.detach().max(1)[1], labeled_target.detach().squeeze(1),
                                            group_name=label_group)  # to register the dice, you should put the label_group here.
                report_dict = self.meters.tracking_status()
                self._indicator.set_postfix_dict(report_dict)
        return report_dict

    @staticmethod
    def _unzip_data(data, device):
        (image, target), _, filename, partition, group = \
            preprocess_input_with_twice_transformation(data, device)
        return image, target, filename, partition, group

# class FullEpocherExp(_num_class_mixin, _Epocher):
#
#     def __init__(self, model: Union[Model, nn.Module], optimizer: T_optim, labeled_loader: T_loader,
#                  sup_criterion: T_loss, cur_epoch=0,
#                  device="cpu") -> None:
#         super().__init__(model=model, num_batches=len(labeled_loader), cur_epoch=cur_epoch, device=device)
#
#         self._optimizer = optimizer
#         self._labeled_loader = labeled_loader
#         self._sup_criterion = sup_criterion
#
#     def _configure_meters(self, meters: MeterInterface) -> MeterInterface:
#         C = self.num_classes
#         report_axis = list(range(1, C))
#         meters.register_meter("lr", AverageValueMeter())
#         meters.register_meter("sup_loss", AverageValueMeter())
#         meters.register_meter("sup_dice", UniversalDice(C, report_axises=report_axis, ))
#         return meters
#
#     def _run(self, *args, **kwargs) -> EpochResultDict:
#         self.meters["lr"].add(get_lrs_from_optimizer(self._optimizer)[0])
#         self._model.train()
#         assert self._model.training, self._model.training
#         report_dict = {}
#         mask_path = PROJECT_PATH + '/.data/ACDC_contrast/train/train_masks/'
#         seg = os.listdir(mask_path)
#         for i, labeled_data in zip(self._indicator, self._labeled_loader):
#             labeled_image, labeled_target, labeled_filename, _, label_group = \
#                 self._unzip_data(labeled_data, self._device)
#             breakpoint()
#             ls = []
#             for j in range(len(labeled_image)):
#                 for k in range(len(seg)):
#                     if labeled_filename[j] == seg[k][:-4]:
#                         pred = np.load(mask_path + seg[k])
#                         pred = torch.from_numpy(pred).cuda()
#                         ls.append(torch.cat([pred, labeled_image[j]]))
#             new_input = torch.stack(ls)
#             predict_logits = self._model(new_input).softmax(1)
#
#             onehot_target = class2one_hot(labeled_target.squeeze(1), self.num_classes)
#             sup_loss = self._sup_criterion(predict_logits, onehot_target)
#
#             # supervised part
#             total_loss = sup_loss
#             # gradient backpropagation
#             self._optimizer.zero_grad()
#             total_loss.backward()
#             self._optimizer.step()
#             # recording can be here or in the regularization method
#             with torch.no_grad():
#                 self.meters["sup_loss"].add(sup_loss.item())
#                 self.meters["sup_dice"].add(predict_logits.max(1)[1], labeled_target.squeeze(1))
#                 report_dict = self.meters.tracking_status()
#                 self._indicator.set_postfix_dict(report_dict)
#         return report_dict
#
#     @staticmethod
#     def _unzip_data(data, device):
#         (image, target), _, filename, partition, group = \
#             preprocess_input_with_twice_transformation_for_exp2(data, device)
#         return image, target, filename, partition, group
