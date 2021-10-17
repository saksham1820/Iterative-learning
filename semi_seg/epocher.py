import typing as t
from collections import defaultdict
from functools import lru_cache
from typing import Union, Tuple

import torch
from deepclustering2.epoch import _Epocher  # noqa
from deepclustering2.meters2 import EpochResultDict, AverageValueMeter, UniversalDice, MeterInterface
from deepclustering2.meters2.individual_meters import _Metric
from deepclustering2.models import Model
from deepclustering2.optim import get_lrs_from_optimizer
from deepclustering2.type import T_loader, T_loss, T_optim
from deepclustering2.utils import class2one_hot
from loguru import logger
from torch import nn
from torch.utils.data import DataLoader

from contrastyou.epocher.utils import write_predict, write_img_target  # noqa
from contrastyou.trainer._utils import ClusterHead  # noqa


class AverageValueDictionaryMeter(_Metric):
    def __init__(self) -> None:
        super().__init__()
        self._meter_dicts: t.Dict[str, AverageValueMeter] = defaultdict(AverageValueMeter)

    def reset(self):
        for k, v in self._meter_dicts.items():
            v.reset()

    def add(self, **kwargs):
        for k, v in kwargs.items():
            self._meter_dicts[k].add(v)

    def summary(self):
        return {k: v.summary() for k, v in self._meter_dicts.items()}

    def detailed_summary(self):
        return self.summary()


class AverageValueListMeter(AverageValueDictionaryMeter):
    def add(self, list_value: t.Iterable[float] = None, **kwargs):
        assert isinstance(list_value, t.Iterable)
        for i, v in enumerate(list_value):
            self._meter_dicts[str(i)].add(v)


class _num_class_mixin:
    _model: nn.Module

    @property
    def num_classes(self):
        return self._model.num_classes

    @property
    @lru_cache()
    def num_iters(self) -> int:
        try:
            return self._model.num_iters
        except Exception:
            logger.opt(exception=True).debug("num_iters error")
            return 0


'''
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
'''
"""
class InverseIterativeEpocher2(_UnzipMixin, _num_class_mixin, _Epocher):

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
        # meters.register_meter("sup_loss", AverageValueMeter())
        # meters.register_meter("sup_dice", UniversalDice(C, report_axises=report_axis, ))
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

            # convLSTM.......
            # 1 how to do the inference.
            # iter:
            #       epoch

            # inference.
            # mask
            # from
            # 0, 1, 2, 3, ....N:
            #
            # # checkpoints of N.
            # training: input: image + well - predoicted
            # mask -> better - predicted
            # mask
            #
            # inference: input: image + teacher
            # prediction -> better - predicted
            # mask  # avoid multiple checkpoints.

            # epcoh :
            #  iter # convlstm.... (stop.. reduce innov.)
            # another network that will judge the quality. # merged this model with the curent model.
            # how to decide what to improve and what not to improve. alpha as a map. (still in this direction.)
            # yeh... (alpha dynamic).... next week...
            # under-fitting. Working on the code issues.

            # we have to store checkpints.
            # 2. long dependency.  iter:
            # epoch using convLSTM .......
            # Boosting using convlstm. ...
            # only focus on the region not right.

            # getting the training work
            # start with it, improve the baseline.
            # overfitting. the student gets the prediction of the teacher.
            # training only one or two last layer of the network.
            # weak model for boosting........ (Good idea).

            onehot_target = class2one_hot(labeled_target.squeeze(1), self.num_classes)

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
                                           disable_assert=True)

            # supervised part

            # gradient backpropagation
            total_loss = cur_loss
            self._optimizer.zero_grad()
            cur_loss.backward()
            self._optimizer.step()
            # recording can be here or in the regularization method
            with torch.no_grad():
                self.meters[f"itrloss_{ITER}"].add(total_loss.item())
                self.meters[f"itrdice_{ITER}"].add(aggregated_simplex.max(1)[1], labeled_target.squeeze(),
                                                   group_name=label_group)
                report_dict = self.meters.tracking_status()
                self._indicator.set_postfix_dict(report_dict)
        return report_dict
"""
"""
class TrainEpocher(_num_class_mixin, _Epocher):

    def __init__(self, model: nn.Module, optimizer: T_optim, labeled_loader: T_loader, unlabeled_loader: T_loader,
                 sup_criterion: T_loss, reg_weight: float, num_batches: int, cur_epoch=0,
                 device="cpu", **kwargs) -> None:
        super().__init__(model=model, num_batches=num_batches, cur_epoch=cur_epoch, device=device, **kwargs)
        self._optimizer = optimizer
        self._labeled_loader = labeled_loader
        self._unlabeled_loader = unlabeled_loader
        self._sup_criterion = sup_criterion
        self._reg_weight = reg_weight

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
"""


# this is a unzip mixin
class _UnzipMixin:
    @staticmethod
    def _unzip_data(data, device):
        image, target, filename, partition, group, pretrained_prediction = \
            data[0].to(device), data[1].to(device), data[2], data[3], data[4], data[5].to(device)
        return image, target, filename, partition, group, pretrained_prediction


class AugmentMixin:

    def __init__(self, *, augment, **kwargs) -> None:
        super().__init__(**kwargs)
        self._augment = augment


# fully supervision
class FullEvalEpocher(_UnzipMixin, AugmentMixin, _num_class_mixin, _Epocher):

    def __init__(self, model: Union[Model, nn.Module], loader: T_loader, sup_criterion: T_loss,
                 cur_epoch=0, device="cpu", **kwargs) -> None:
        assert isinstance(loader, DataLoader), \
            f"val_loader should be an instance of DataLoader, given {loader.__class__.__name__}."
        super().__init__(model=model, num_batches=len(loader), cur_epoch=cur_epoch, device=device, **kwargs)
        self._loader = loader
        self._sup_criterion = sup_criterion
        assert model.num_iters == 0

    def _configure_meters(self, meters: MeterInterface) -> MeterInterface:
        C = self.num_classes
        report_axis = list(range(1, C))
        meters.register_meter("loss", AverageValueMeter())
        meters.register_meter("dice", UniversalDice(C, report_axises=report_axis))
        return meters

    @torch.no_grad()
    def _run(self, *args, **kwargs) -> Tuple[EpochResultDict, float]:
        self._model.eval()
        report_dict = EpochResultDict()
        for i, data in zip(self._indicator, self._loader):
            image, target, filename, _, group_name, teacher_pred = \
                self._unzip_data(data, self._device)
            (image_, teacher_pred_), target_ = self._augment(
                images=(image, teacher_pred.float()), targets=target.float())
            target_ = target_.long()

            predict_logits = self._model(image_)
            onehot_target = class2one_hot(target_.squeeze(1), self.num_classes)

            val_loss = self._sup_criterion(predict_logits.softmax(1), onehot_target, disable_assert=True)

            self.meters["loss"].add(val_loss.item())
            self.meters["dice"].add(predict_logits.max(1)[1], target_.squeeze(1), group_name=group_name)
            report_dict = self.meters.tracking_status()
            self._indicator.set_postfix_dict(report_dict)
        return report_dict, self.meters["dice"].summary()["DSC_mean"]


class FullEpocher(AugmentMixin, _UnzipMixin, _num_class_mixin, _Epocher):

    def __init__(self, model: Union[Model, nn.Module], optimizer: T_optim, labeled_loader: T_loader,
                 sup_criterion: T_loss, cur_epoch=0, num_batches=300, device="cpu", **kwargs) -> None:
        super().__init__(model=model, num_batches=num_batches, cur_epoch=cur_epoch, device=device, **kwargs)

        self._optimizer = optimizer
        self._labeled_loader = labeled_loader
        self._sup_criterion = sup_criterion
        assert model.num_iters == 0

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
            labeled_image, labeled_target, labeled_filename, _, label_group, teacher_pred = \
                self._unzip_data(labeled_data, self._device)

            (labeled_image_, teacher_pred_), labeled_target_ = self._augment(
                images=(labeled_image, teacher_pred.float()), targets=labeled_target.float())
            labeled_target_ = labeled_target_.long()

            predict_logits = self._model(labeled_image_)

            onehot_target = class2one_hot(labeled_target_.squeeze(1), self.num_classes)
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
                self.meters["sup_dice"].add(predict_logits.max(1)[1], labeled_target_.squeeze(1),
                                            group_name=label_group)
                report_dict = self.meters.tracking_status()
                self._indicator.set_postfix_dict(report_dict)
        return report_dict


# iterative one
class IterativeEvalEpocher(AugmentMixin, _UnzipMixin, _num_class_mixin, _Epocher):

    def __init__(self, model: nn.Module, loader: T_loader, sup_criterion: T_loss, lstm_criterion: T_loss, cur_epoch=0,
                 device="cpu",
                 **kwargs) -> None:
        assert isinstance(loader, DataLoader), \
            f"val_loader should be an instance of DataLoader, given {loader.__class__.__name__}."
        super().__init__(model=model, num_batches=len(loader), cur_epoch=cur_epoch, device=device, **kwargs)
        self._loader = loader
        self._sup_criterion = sup_criterion
        self._lstm_criterion = lstm_criterion
        assert model.num_iters > 0

    def _configure_meters(self, meters: MeterInterface) -> MeterInterface:
        C = self.num_classes
        report_axis = list(range(1, C))
        meters.register_meter("sup_loss", AverageValueMeter())
        meters.register_meter("dice", UniversalDice(C, report_axises=report_axis, ))
        meters.register_meter("iloss", AverageValueListMeter())
        for i in range(self.num_iters):
            meters.register_meter(f"idsc{i}", UniversalDice(C, report_axises=report_axis, ))
        return meters

    @torch.no_grad()
    def _run(self, *args, **kwargs) -> Tuple[EpochResultDict, float]:
        self._model.eval()
        report_dict = EpochResultDict()

        for i, data in zip(self._indicator, self._loader):
            image, target, filename, _, group_name, teacher_pred = \
                self._unzip_data(data, self._device)

            (image_, teacher_pred_), target_ = self._augment(
                images=(image, teacher_pred.float()), targets=target.float())
            target_ = target_.long()
            onehot_target = class2one_hot(target_.squeeze(1), self.num_classes)

            logits, corrected_logits, errors = self._model(image_)
            sup_loss = self._sup_criterion(logits.softmax(1), onehot_target, disable_assert=True)
            lstm_loss = self._lstm_criterion(corrected_logits, onehot_target)

            with torch.no_grad():
                self.meters["sup_loss"].add(sup_loss.item())
                self.meters["dice"].add(logits.max(1)[1], target_.squeeze(1), group_name=group_name)
                self.meters["iloss"].add(lstm_loss.tolist())

                for _i in range(self.num_iters):
                    self.meters[f"idsc{_i}"].add(corrected_logits[:, _i].max(1)[1], target_.squeeze(),
                                                 group_name=group_name)

            total_loss = sup_loss + lstm_loss.mean()

            report_dict = self.meters.tracking_status()
            self._indicator.set_postfix_dict(report_dict)
        return report_dict, self.meters["dice"].summary()["DSC_mean"]


class IterativeEpocher(_UnzipMixin, AugmentMixin, _num_class_mixin, _Epocher):

    def __init__(self, model: Union[Model, nn.Module], optimizer: T_optim,
                 labeled_loader: T_loader,
                 sup_criterion: T_loss, lstm_criterion: T_loss, cur_epoch=0, num_batches=100,
                 device="cpu", **kwargs) -> None:
        super().__init__(model=model, num_batches=num_batches, cur_epoch=cur_epoch, device=device, **kwargs)

        self._optimizer = optimizer
        self._labeled_loader = labeled_loader
        self._sup_criterion = sup_criterion
        self._lstm_criterion = lstm_criterion

    def _configure_meters(self, meters: MeterInterface) -> MeterInterface:
        C = self.num_classes
        report_axis = list(range(1, C))
        meters.register_meter("lr", AverageValueMeter())
        meters.register_meter("lstm", AverageValueListMeter())
        meters.register_meter("sup_loss", AverageValueMeter())
        meters.register_meter("sup_dice", UniversalDice(C, report_axises=report_axis, ))
        for i in range(self.num_iters):
            meters.register_meter(f"idsc{i}", UniversalDice(C, report_axises=report_axis, ))
        return meters

    def _run(self, *args, **kwargs) -> EpochResultDict:
        self.meters["lr"].add(get_lrs_from_optimizer(self._optimizer)[0])
        self._model.train()
        assert self._model.training, self._model.training
        report_dict = {}

        for i, labeled_data in zip(self._indicator, self._labeled_loader):
            labeled_image, labeled_target, labeled_filename, _, label_group, teacher_pred = \
                self._unzip_data(labeled_data, self._device)
            # (5, 1, 224, 224) -> labeled_image.shape
            (labeled_image_, teacher_pred_), labeled_target_ = self._augment(
                images=(labeled_image, teacher_pred.float()), targets=labeled_target.float())
            labeled_target_ = labeled_target_.long()

            onehot_target = class2one_hot(labeled_target_.squeeze(1), self.num_classes)
            logits, corrected_logits, errors = self._model(labeled_image_)
            cur_loss = self._sup_criterion(logits.softmax(1), onehot_target)
            lstm_loss = self._lstm_criterion(corrected_logits, onehot_target)

            with torch.no_grad():
                self.meters['sup_loss'].add(cur_loss.item())
                self.meters["sup_dice"].add(logits.max(1)[1], labeled_target_.squeeze(), group_name=label_group)
                self.meters[f"lstm"].add(lstm_loss.tolist())
                for _i in range(self.num_iters):
                    self.meters[f"idsc{_i}"].add(corrected_logits[:, _i].max(1)[1], labeled_target_.squeeze(),
                                                 group_name=label_group)

            total_loss = sum(cur_loss + lstm_loss)
            # gradient backpropagation

            self._optimizer.zero_grad()
            total_loss.backward()
            self._optimizer.step()
            # recording can be here or in the regularization method
            with torch.no_grad():
                report_dict = self.meters.tracking_status()
                self._indicator.set_postfix_dict(report_dict)
        return report_dict


"""
class InverseIterativeEvalEpocher(AugmentMixin, _UnzipMixin, _num_class_mixin, _Epocher):

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


class InverseIterativeEpocher(AugmentMixin, _UnzipMixin, _num_class_mixin, _Epocher):

    def __init__(self, memory_bank, alpha: float, num_iter, model: Union[Model, nn.Module], optimizer: T_optim,
                 labeled_loader: T_loader,
                 sup_criterion: T_loss, cur_epoch=0, num_batches=100,
                 device="cpu", **kwargs) -> None:
        super().__init__(model=model, num_batches=num_batches, cur_epoch=cur_epoch, device=device, **kwargs)
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
        # meters.register_meter("sup_loss", AverageValueMeter())
        # meters.register_meter("sup_dice", UniversalDice(C, report_axises=report_axis, ))
        num_iters = self._num_iter
        # assert num_iters >= 1
        # for i in range(num_iters):
        #     meters.register_meter(f"itrdice_{i}", UniversalDice(C, report_axises=report_axis, ))
        #     meters.register_meter(f"itrloss_{i}", AverageValueMeter())
        return meters

    def _run(self, *args, **kwargs) -> EpochResultDict:
        self.meters["lr"].add(get_lrs_from_optimizer(self._optimizer)[0])
        self._model.train()
        assert self._model.training, self._model.training
        report_dict = {}

        for ITER in range(self._num_iter):

            for i, labeled_data in zip(self._indicator, self._labeled_loader):
                labeled_image, labeled_target, labeled_filename, _, label_group, teacher_pred = \
                    self._unzip_data(labeled_data, self._device)

                # data augmentation
                labeled_image_, (labeled_target_, teacher_pred_) = self._augment(
                    images=labeled_image, targets=(labeled_target.float(), teacher_pred.float()))
                labeled_target_ = labeled_target_.long()

                # (5, 1, 224, 224) -> labeled_image.shape
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
                                               disable_assert=True)

                # supervised part

                # gradient backpropagation
                total_loss = cur_loss
                self._optimizer.zero_grad()
                cur_loss.backward()
                self._optimizer.step()
                # recording can be here or in the regularization method
                with torch.no_grad():
                    self.meters[f"itrloss_{ITER}"].add(total_loss.item())
                    self.meters[f"itrdice_{ITER}"].add(aggregated_simplex.max(1)[1], labeled_target.squeeze(),
                                                       group_name=label_group)
                    report_dict = self.meters.tracking_status()
                    self._indicator.set_postfix_dict(report_dict)
        return report_dict
"""
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
