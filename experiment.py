import os
import typing as t
import warnings

warnings.simplefilter("ignore")

from deepclustering2.configparser import ConfigManger
from deepclustering2.dataloader.sampler import InfiniteRandomSampler
from deepclustering2.loss import KL_div
from deepclustering2.utils import set_benchmark
import torch

from torch.utils.data import DataLoader

from contrastyou import DATA_PATH, CONFIG_PATH
from contrastyou.dataloader.acdc_dataset import ACDCDatasetWithTeacherPrediction, ACDCDataset
from semi_seg.augment import TensorAugment
from semi_seg.trainer import trainer_zoos
from utils import extract_dataset_based_on_num_patients
from contrastyou.augment.helper import fix_all_seed_within_context


def get_model(trainer_name: str, config: t.Dict[str, t.Dict[str, t.Any]]):
    seed = config.get("RandomSeed", 1)
    with fix_all_seed_within_context(seed=seed):
        if trainer_name == "full":
            from contrastyou.arch.unet_convlstm import UNet
            return UNet(**config["Arch"])
        elif trainer_name == "iterative":
            if config["Arch"]["name"] == "type1":
                from contrastyou.arch.unet_convlstm import LSTM_Corrected_Unet as UNet
                return UNet(**config["Arch"], seq_len=config["Iterations"]["num_iter"])
            elif config["Arch"]["name"] == "type2":
                from contrastyou.arch.iterative import RefinementModel
                return RefinementModel(**config["Arch"], seq_len=config["Iterations"]["num_iter"])
            else:
                raise NotImplemented(config["Arch"]["name"])
        else:
            raise NotImplemented(trainer_name)


con_manager = ConfigManger(os.path.join(CONFIG_PATH, "semi.yaml"))
config = con_manager.config
set_benchmark(config.get("RandomSeed", 1))
torch.use_deterministic_algorithms(True)

tra_transforms = TensorAugment.pretrain
val_transforms = TensorAugment.val

train_dataset: 'ACDCDataset' = ACDCDatasetWithTeacherPrediction(
    root_dir=DATA_PATH, mode="train", verbose=True,
)
(train_dataset,) = extract_dataset_based_on_num_patients(10, dataset=train_dataset, seed=0)
val_dataset: 'ACDCDataset' = ACDCDatasetWithTeacherPrediction(root_dir=DATA_PATH, mode="val", verbose=True)

shuffle = config["LabeledData"].get("shuffle", True)
sampler = InfiniteRandomSampler(train_dataset, shuffle=shuffle)
train_loader = DataLoader(
    train_dataset, **{k: v for k, v in config["LabeledData"].items() if k != "shuffle"},
    sampler=sampler, pin_memory=True, persistent_workers=True
)
val_loader = DataLoader(
    val_dataset, **config["ValidationData"],
    pin_memory=True
)

trainer_name = config["Trainer"].pop("name")

model = get_model(trainer_name, config)
Trainer = trainer_zoos[trainer_name]

trainer = Trainer(
    model=model, labeled_loader=train_loader, val_loader=val_loader, sup_criterion=KL_div(),
    configuration=con_manager.config, **{k: v for k, v in config["Trainer"].items() if k != "freeze_grad"},
    tra_augment=tra_transforms, val_augment=val_transforms
)

trainer.init()

checkpoint = config.get("Checkpoint", None)
if checkpoint: trainer.load_state_dict_from_path(checkpoint, strict=False)

if config["Trainer"].get("freeze_grad"):
    try:
        model.disable_grad(*config["Trainer"]["freeze_grad"])
    except AttributeError:
        model._unet.disable_grad(*config["Trainer"]["freeze_grad"])

trainer.start_training()
