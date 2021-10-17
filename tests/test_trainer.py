import os
from unittest import TestCase

import torch
from deepclustering2.loss import KL_div
from deepclustering2.utils import load_yaml
from torch.utils.data import DataLoader

from contrastyou import DATA_PATH, CONFIG_PATH
from contrastyou.arch import UNet, LSTM_Corrected_Unet
from contrastyou.arch.unet_convlstm import LSTMErrorLoss
from contrastyou.dataloader.acdc_dataset import ACDCDatasetWithTeacherPrediction
from semi_seg.augment import TensorAugment
from semi_seg.trainer import FullTrainer, IterativeTrainer


class TestFullTrainer(TestCase):
    def setUp(self) -> None:
        super().setUp()

        self._tra_set = ACDCDatasetWithTeacherPrediction(
            root_dir=DATA_PATH, mode="train", verbose=True,
        )
        self._val_set = ACDCDatasetWithTeacherPrediction(
            root_dir=DATA_PATH, mode="val", verbose=True,
        )
        self._tra_loader = DataLoader(self._tra_set, batch_size=4, shuffle=True, num_workers=6, persistent_workers=True)
        self._val_loader = DataLoader(self._val_set, batch_size=4, shuffle=False, num_workers=6)

        self._sup_criterion = KL_div()
        self._model = UNet(input_dim=1, num_classes=4)
        self._optimizer = torch.optim.Adam(self._model.parameters(), )
        self._config = load_yaml(os.path.join(CONFIG_PATH, "semi.yaml"))

    def test_run(self):
        trainer = FullTrainer(model=self._model, labeled_loader=self._tra_loader, val_loader=self._val_loader,
                              sup_criterion=self._sup_criterion, save_dir="tmp", max_epoch=2, num_batches=100,
                              device="cuda", configuration=self._config, tra_augment=TensorAugment.pretrain,
                              val_augment=TensorAugment.val)
        trainer.init()

        trainer.start_training()


class TestIterativeTrainer(TestCase):
    def setUp(self) -> None:
        super().setUp()

        self._tra_set = ACDCDatasetWithTeacherPrediction(
            root_dir=DATA_PATH, mode="train", verbose=True,
        )
        self._val_set = ACDCDatasetWithTeacherPrediction(
            root_dir=DATA_PATH, mode="val", verbose=True,
        )
        self._tra_loader = DataLoader(self._tra_set, batch_size=4, shuffle=True, num_workers=6, persistent_workers=True)
        self._val_loader = DataLoader(self._val_set, batch_size=4, shuffle=False, num_workers=6)

        self._sup_criterion = KL_div()
        self._lstm_criterion = LSTMErrorLoss()
        self._model = LSTM_Corrected_Unet(input_dim=1, num_classes=4)
        self._optimizer = torch.optim.Adam(self._model.parameters(), )
        self._config = load_yaml(os.path.join(CONFIG_PATH, "semi.yaml"))

    def test_run(self):
        trainer = IterativeTrainer(model=self._model, labeled_loader=self._tra_loader, val_loader=self._val_loader,
                                   sup_criterion=self._sup_criterion, save_dir="tmp", max_epoch=2, num_batches=100,
                                   device="cuda", configuration=self._config, tra_augment=TensorAugment.pretrain,
                                   val_augment=TensorAugment.val)
        trainer.init()

        trainer.start_training()
