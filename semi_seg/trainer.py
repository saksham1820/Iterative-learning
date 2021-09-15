import os
import torch
from contrastyou import PROJECT_PATH
from deepclustering2 import optim
from deepclustering2.configparser import ConfigManger
from deepclustering2.meters2 import EpochResultDict, StorageIncomeDict
from deepclustering2.schedulers import GradualWarmupScheduler
from deepclustering2.trainer import Trainer
from deepclustering2.type import T_loader, T_loss
from pathlib import Path
from semi_seg.epocher import FullEpocher, IterativeEpocher, TrainEpocher, EvalEpocher, InferenceEpocher, FullEvalEpocher
from torch import nn
from typing import Tuple

cmanager = ConfigManger(Path(PROJECT_PATH) / "config/semi.yaml")
config = cmanager.config

__all__ = ["trainer_zoos"]


class SemiTrainer(Trainer):
    RUN_PATH = str(Path(PROJECT_PATH) / "semi_seg" / "runs")  # noqa

    feature_positions = ["Up_conv4", "Up_conv3"]

    def __init__(self, *, alpha: float = config["Aggregator"]["alpha"],
                 num_iter: int = config["Iterations"]["num_iter"],
                 model: nn.Module, labeled_loader: T_loader, unlabeled_loader: T_loader,
                 val_loader: T_loader,
                 sup_criterion: T_loss, save_dir: str = "base",
                 max_epoch: int = 100, num_batches: int = 100, device: str = "cpu", configuration=None, **kwargs):
        super().__init__(model, save_dir, max_epoch, num_batches, device, configuration)

        self._labeled_loader = labeled_loader
        self._unlabeled_loader = unlabeled_loader
        self._val_loader = val_loader
        # self._test_loader = test_loader
        self._sup_criterion = sup_criterion

    def init(self):
        self._init()
        self._init_optimizer()
        self._init_scheduler(self._optimizer)

    def _init(self):
        self.set_feature_positions(self._config["Trainer"]["feature_names"])
        feature_importance = self._config["Trainer"]["feature_importance"]
        assert isinstance(feature_importance, list), type(feature_importance)
        feature_importance = [float(x) for x in feature_importance]
        self._feature_importance = [x / sum(feature_importance) for x in feature_importance]
        assert len(self._feature_importance) == len(self.feature_positions)

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

    def _run_epoch(self, *args, **kwargs) -> EpochResultDict:
        trainer = TrainEpocher(
            self._model, self._optimizer, self._labeled_loader, self._unlabeled_loader,
            self._sup_criterion, 0, self._num_batches, self._cur_epoch, self._device,
            feature_position=self.feature_positions, feature_importance=self._feature_importance
        )
        result = trainer.run()
        return result

    def _eval_epoch(self, *, loader: T_loader, **kwargs) -> Tuple[EpochResultDict, float]:
        evaler = EvalEpocher(model=self._model, val_loader=loader,
                             sup_criterion=self._sup_criterion,
                             cur_epoch=self._cur_epoch, device=self._device,
                             num_iter=config["Iterations"]["num_iter"], alpha=config["Aggregator"]["alpha"])
        result, cur_score = evaler.run()
        return result, cur_score

    def _start_training(self):
        for self._cur_epoch in range(self._start_epoch, self._max_epoch):
            train_result: EpochResultDict
            eval_result: EpochResultDict
            cur_score: float
            train_result = self.run_epoch()
            with torch.no_grad():
                eval_result, cur_score = self.eval_epoch(loader=self._val_loader)
            #    test_result = {}
            #    if self._test_loader is not None:
            #        test_result, _ = self.eval_epoch(loader=self._test_loader)
            # update lr_scheduler
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

    @classmethod
    def set_feature_positions(cls, feature_positions):
        cls.feature_positions = feature_positions


class IterativeTrainer(SemiTrainer):

    def __init__(self, *, alpha: float, num_iter: int, model: nn.Module, labeled_loader: T_loader, val_loader: T_loader,
                 sup_criterion: T_loss, save_dir: str = "base", max_epoch: int = 100,
                 num_batches: int = 100, device: str = "cpu", configuration=None, **kwargs):
        super().__init__(model=model, labeled_loader=labeled_loader, unlabeled_loader=None,
                         val_loader=val_loader, sup_criterion=sup_criterion, save_dir=save_dir,
                         max_epoch=max_epoch, num_batches=num_batches, device=device,
                         configuration=configuration,
                         **kwargs)
        self._alpha = alpha
        self._num_iter = num_iter
        self._labeled_loader = labeled_loader
        self._val_loader = val_loader
        self._sup_criterion = sup_criterion

    def init(self):
        self._init_optimizer()
        self._init_scheduler(self._optimizer)

    def _run_epoch(self, *args, **kwargs) -> EpochResultDict:
        trainer = IterativeEpocher(alpha=self._alpha, num_iter=self._num_iter, model=self._model,
                                   optimizer=self._optimizer, labeled_loader=self._labeled_loader,
                                   sup_criterion=self._sup_criterion, device=self._device,
                                   num_batches=self._num_batches,
                                   cur_epoch=self._cur_epoch)
        result = trainer.run()
        return result


class FullTrainer(SemiTrainer):

    def __init__(self, *, model: nn.Module, labeled_loader: T_loader,
                 val_loader: T_loader, test_loader: T_loader,
                 sup_criterion: T_loss, save_dir: str = "base", max_epoch: int = 100,
                 num_batches: int = 100, device: str = "cpu", configuration=None, **kwargs):
        super().__init__(model=model, labeled_loader=labeled_loader, unlabeled_loader=None, test_loader=test_loader,
                         val_loader=val_loader, sup_criterion=sup_criterion, save_dir=save_dir,
                         max_epoch=max_epoch, num_batches=num_batches, device=device,
                         configuration=configuration,
                         **kwargs)

        self._labeled_loader = labeled_loader
        self._val_loader = val_loader
        self._sup_criterion = sup_criterion

    def init(self):
        self._init_optimizer()
        self._init_scheduler(self._optimizer)

    def _run_epoch(self, *args, **kwargs) -> EpochResultDict:
        trainer = FullEpocher(model=self._model, optimizer=self._optimizer, labeled_loader=self._labeled_loader,
                              sup_criterion=self._sup_criterion, device=self._device, cur_epoch=self._cur_epoch,
                              num_batches=self._num_batches)
        result = trainer.run()
        return result

    def _eval_epoch(self, *, loader: T_loader, **kwargs) -> Tuple[EpochResultDict, float]:
        evaler = FullEvalEpocher(model=self._model, val_loader=loader,
                                 # todo: remove the iterative thing for the Full trainer.
                                 sup_criterion=self._sup_criterion,
                                 cur_epoch=self._cur_epoch, device=self._device)
        result, cur_score = evaler.run()
        return result, cur_score


trainer_zoos = {
    "full": FullTrainer,
    "iterative": IterativeTrainer
}
