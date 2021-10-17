from unittest import TestCase

import torch
from deepclustering2.loss import KL_div
from torch.utils.data import DataLoader

from contrastyou import DATA_PATH
from contrastyou.arch import UNet, LSTM_Corrected_Unet
from contrastyou.arch.unet_convlstm import LSTMErrorLoss
from contrastyou.dataloader.acdc_dataset import ACDCDatasetWithTeacherPrediction
from semi_seg.augment import TensorAugment
from semi_seg.epocher import FullEvalEpocher, FullEpocher, IterativeEvalEpocher, IterativeEpocher


class TestFullEpocher(TestCase):
    def setUp(self) -> None:
        super().setUp()
        self._dataset = ACDCDatasetWithTeacherPrediction(
            root_dir=DATA_PATH, mode="train", verbose=True,
        )
        self._dataloader = DataLoader(self._dataset, batch_size=4, shuffle=True, num_workers=6)

        self._sup_criterion = KL_div()
        self._model = UNet(input_dim=1, num_classes=4)
        self._optimizer = torch.optim.Adam(self._model.parameters(), )

    def test_val_epocher(self):
        test_epocher = FullEvalEpocher(model=self._model, loader=self._dataloader, sup_criterion=KL_div(),
                                       cur_epoch=0, device="cuda", augment=TensorAugment.val)

        report_dict, best_score = test_epocher.run()

    def test_train_epocher(self):
        tra_epocher = FullEpocher(model=self._model, labeled_loader=self._dataloader, sup_criterion=KL_div(),
                                  cur_epoch=0, device="cuda", augment=TensorAugment.pretrain, optimizer=self._optimizer)

        report_dict = tra_epocher.run()


class TestIterativeEpocher(TestCase):

    def setUp(self) -> None:
        super().setUp()
        self._dataset = ACDCDatasetWithTeacherPrediction(
            root_dir=DATA_PATH, mode="train", verbose=True,
        )
        self._dataloader = DataLoader(self._dataset, batch_size=4, shuffle=True, num_workers=6)

        self._sup_criterion = KL_div()
        self._model = LSTM_Corrected_Unet(input_dim=1, num_classes=4)
        self._optimizer = torch.optim.Adam(self._model.parameters(), )
        self._lstm_criterion = LSTMErrorLoss()

    def test_val_epocher(self):
        test_epocher = IterativeEvalEpocher(model=self._model, loader=self._dataloader,
                                            sup_criterion=self._sup_criterion,
                                            cur_epoch=0, device="cuda", augment=TensorAugment.val,
                                            lstm_criterion=self._lstm_criterion)
        report_dict, best_score = test_epocher.run()

    def test_tra_epocher(self):
        test_epocher = IterativeEpocher(model=self._model, labeled_loader=self._dataloader,
                                        sup_criterion=self._sup_criterion,
                                        cur_epoch=0, device="cuda", augment=TensorAugment.pretrain,
                                        lstm_criterion=self._lstm_criterion, optimizer=self._optimizer)
        report_dict  = test_epocher.run()
