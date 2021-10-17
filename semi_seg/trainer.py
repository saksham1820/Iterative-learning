import os
import typing as t
from abc import abstractmethod
from pathlib import Path
from typing import Tuple

import torch
from deepclustering2 import optim
from deepclustering2.meters2 import EpochResultDict, StorageIncomeDict
from deepclustering2.schedulers import GradualWarmupScheduler
from deepclustering2.trainer import Trainer
from deepclustering2.type import T_loader, T_loss
from torch import nn, Tensor

from contrastyou import PROJECT_PATH
from contrastyou.arch.unet_convlstm import LSTMErrorLoss
from semi_seg.epocher import FullEpocher, IterativeEpocher, IterativeEvalEpocher, \
    FullEvalEpocher

__all__ = ["trainer_zoos", "FullTrainer", "IterativeTrainer"]

type_augment = t.Callable[[Tensor, Tensor], t.Tuple[Tensor, Tensor]]


class TensorAugmentPlugin:

    def __init__(self, *, tra_augment: type_augment, val_augment: type_augment, **kwargs) -> None:
        self._tra_tensor_augment: type_augment = tra_augment
        self._val_tensor_augment: type_augment = val_augment
        super().__init__(**kwargs)


class BaseTrainer(TensorAugmentPlugin, Trainer, ):
    RUN_PATH = str(Path(PROJECT_PATH) / "semi_seg" / "runs")  # noqa

    def __init__(self, *, model: nn.Module, labeled_loader: T_loader, val_loader: T_loader,
                 sup_criterion: T_loss, save_dir: str = "base", max_epoch: int = 100, num_batches: int = 100,
                 device: str = "cpu", configuration, **kwargs):
        super().__init__(model=model, save_dir=save_dir, max_epoch=max_epoch, num_batches=num_batches, device=device,
                         configuration=configuration, **kwargs)
        self._labeled_loader = labeled_loader
        self._val_loader = val_loader
        self._sup_criterion = sup_criterion

    def init(self):
        self._init()
        self._init_optimizer()
        self._init_scheduler(self._optimizer)

    def _init(self):
        pass

    def _init_scheduler(self, optimizer):
        scheduler_dict = self._config.get("Scheduler", None)
        if scheduler_dict is None:
            return
        else:
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self._optimizer,
                T_max=self._config["Trainer"]["max_epoch"] - self._config["Scheduler"]["warmup_max"],
                eta_min=1e-7
            )
            scheduler = GradualWarmupScheduler(optimizer, scheduler_dict["multiplier"],
                                               total_epoch=scheduler_dict["warmup_max"],
                                               after_scheduler=scheduler)
            self._scheduler = scheduler

    def _init_optimizer(self):
        optim_dict = self._config["Optim"]
        self._optimizer = optim.__dict__[optim_dict["name"]](
            params=self._model.parameters(),
            **{k: v for k, v in optim_dict.items() if k != "name"}
        )

    @abstractmethod
    def _run_epoch(self, *args, **kwargs) -> EpochResultDict:
        ...

    @abstractmethod
    def _eval_epoch(self, *, loader: T_loader, **kwargs) -> Tuple[EpochResultDict, float]:
        ...

    def _start_training(self):
        for self._cur_epoch in range(self._start_epoch, self._max_epoch):
            train_result: EpochResultDict
            eval_result: EpochResultDict
            cur_score: float
            train_result = self.run_epoch()
            with torch.no_grad():
                eval_result, cur_score = self.eval_epoch(loader=self._val_loader)

            if hasattr(self, "_scheduler"):
                self._scheduler.step()
            storage_per_epoch = StorageIncomeDict(tra=train_result, val=eval_result)  # , test=test_result)
            self._storage.put_from_dict(storage_per_epoch, self._cur_epoch)
            self._writer.add_scalar_with_StorageDict(storage_per_epoch, self._cur_epoch)
            # save_checkpoint
            self.save(cur_score)
            # save storage result on csv file.
            self._storage.to_csv(self._save_dir)

    def inference(self, checkpoint=None):  # noqa
        if checkpoint is None:
            self.load_state_dict_from_path(os.path.join(self._save_dir, "best.pth"), strict=True)
        else:
            checkpoint = Path(checkpoint)
            if checkpoint.is_file():
                if not checkpoint.suffix == ".pth":
                    raise FileNotFoundError(checkpoint)
            else:
                assert checkpoint.exists()
                checkpoint = checkpoint / "best.pth"
            self.load_state_dict_from_path(str(checkpoint), strict=True)
        evaler = InferenceEpocher(self._model, val_loader=self._val_loader,  # test_loader=self._test_loader,
                                  sup_criterion=self._sup_criterion, id=1,
                                  cur_epoch=self._cur_epoch, device=self._device)
        evaler.set_save_dir(self._save_dir)
        result, cur_score = evaler.run()
        return result, cur_score


class FullTrainer(BaseTrainer):

    def __init__(self, *, model: nn.Module, labeled_loader: T_loader, val_loader: T_loader, sup_criterion: T_loss,
                 save_dir: str = "base", max_epoch: int = 100, num_batches: int = 100, device: str = "cpu",
                 configuration, **kwargs):
        super().__init__(model=model, labeled_loader=labeled_loader, val_loader=val_loader, sup_criterion=sup_criterion,
                         save_dir=save_dir, max_epoch=max_epoch, num_batches=num_batches, device=device,
                         configuration=configuration, **kwargs)
        assert self._model.num_iters == 0

    def _run_epoch(self, *args, **kwargs) -> EpochResultDict:
        trainer = FullEpocher(model=self._model, optimizer=self._optimizer, labeled_loader=self._labeled_loader,
                              sup_criterion=self._sup_criterion, device=self._device, cur_epoch=self._cur_epoch,
                              num_batches=self._num_batches, augment=self._tra_tensor_augment)
        result = trainer.run()
        return result

    def _eval_epoch(self, *, loader: T_loader, **kwargs) -> Tuple[EpochResultDict, float]:
        evaler = FullEvalEpocher(model=self._model, loader=loader,
                                 sup_criterion=self._sup_criterion,
                                 cur_epoch=self._cur_epoch, device=self._device, augment=self._val_tensor_augment)
        result, cur_score = evaler.run()
        return result, cur_score


class IterativeTrainer(BaseTrainer):

    def __init__(self, *, model: nn.Module, labeled_loader: T_loader, val_loader: T_loader, sup_criterion: T_loss,
                 save_dir: str = "base", max_epoch: int = 100, num_batches: int = 100, device: str = "cpu",
                 configuration, **kwargs):
        super().__init__(model=model, labeled_loader=labeled_loader, val_loader=val_loader, sup_criterion=sup_criterion,
                         save_dir=save_dir, max_epoch=max_epoch, num_batches=num_batches, device=device,
                         configuration=configuration, **kwargs)
        self._lstm_criterion = LSTMErrorLoss()
        assert self._model.num_iters > 0

    def _run_epoch(self, *args, **kwargs) -> EpochResultDict:
        trainer = IterativeEpocher(model=self._model,
                                   optimizer=self._optimizer, labeled_loader=self._labeled_loader,
                                   sup_criterion=self._sup_criterion, device=self._device,
                                   num_batches=self._num_batches, lstm_criterion=self._lstm_criterion,
                                   cur_epoch=self._cur_epoch, augment=self._tra_tensor_augment, **kwargs)
        result = trainer.run()
        return result

    def _eval_epoch(self, *, loader: T_loader, **kwargs) -> Tuple[EpochResultDict, float]:
        evaler = IterativeEvalEpocher(model=self._model, loader=loader,
                                      sup_criterion=self._sup_criterion,
                                      cur_epoch=self._cur_epoch, device=self._device,
                                      lstm_criterion=self._lstm_criterion, augment=self._val_tensor_augment,
                                      )
        result, cur_score = evaler.run()
        return result, cur_score


"""
class InverseIterativeTrainer(BaseTrainer):

    def __init__(self, *, memory_bank, alpha: float, num_iter: int, model: nn.Module, labeled_loader: T_loader,
                 val_loader: T_loader,
                 sup_criterion: T_loss, save_dir: str = "base", max_epoch: int = 100,
                 num_batches: int = 100, device: str = "cpu", configuration=None, **kwargs):
        super().__init__(model=model, labeled_loader=labeled_loader, unlabeled_loader=None,
                         val_loader=val_loader, sup_criterion=sup_criterion, save_dir=save_dir,
                         max_epoch=max_epoch, num_batches=num_batches, device=device,
                         configuration=configuration,
                         **kwargs)
        self._alpha = alpha
        self._memory_bank = memory_bank
        self._num_iter = num_iter
        self._labeled_loader = labeled_loader
        self._val_loader = val_loader
        self._sup_criterion = sup_criterion

    def init(self):
        self._init_optimizer()
        self._init_scheduler(self._optimizer)

    def _run_epoch(self, *args, **kwargs) -> EpochResultDict:
        trainer = InverseIterativeEpocher(memory_bank=self._memory_bank, alpha=self._alpha,
                                          num_iter=self._num_iter, model=self._model,
                                          optimizer=self._optimizer, labeled_loader=self._labeled_loader,
                                          sup_criterion=self._sup_criterion, device=self._device,
                                          num_batches=self._num_batches,
                                          cur_epoch=self._cur_epoch,
                                          augment=self._tra_tensor_augment)
        result = trainer.run()
        return result

    def _eval_epoch(self, *, loader: T_loader, **kwargs) -> Tuple[EpochResultDict, float]:
        mem_bank = {}
        temp = list(loader)
        for i in range(len(temp)):
            for file in temp[i][1]:
                mem_bank[file] = None
        evaler = InverseIterativeEvalEpocher(model=self._model, val_loader=loader,
                                             sup_criterion=self._sup_criterion, memory_bank=mem_bank,
                                             cur_epoch=self._cur_epoch, device=self._device,
                                             num_iter=config["Iterations"]["num_iter"],
                                             alpha=config["Aggregator"]["alpha"])
        result, cur_score = evaler.run()
        return result, cur_score
"""

trainer_zoos = {
    "full": FullTrainer,
    "iterative": IterativeTrainer,
    # "inv_iterative": InverseIterativeTrainer
}
