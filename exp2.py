import os
import warnings

from deepclustering2.configparser import ConfigManger
from deepclustering2.dataloader.sampler import InfiniteRandomSampler
from deepclustering2.loss import KL_div
from deepclustering2.utils import gethash
from deepclustering2.utils import set_benchmark
from scipy.sparse import issparse  # noqa
from torch.utils.data import DataLoader

from contrastyou import DATA_PATH, CONFIG_PATH
from contrastyou.arch import LSTM_Corrected_Unet as UNet
from contrastyou.dataloader._seg_datset import ContrastBatchSampler  # noqa
from contrastyou.dataloader.acdc_dataset import ACDCDatasetWithTeacherPrediction, ACDCDataset
from semi_seg.augment import TensorAugment
from semi_seg.trainer import trainer_zoos
from utils import extract_dataset_based_on_num_patients

warnings.simplefilter("ignore")

# load configure from yaml and argparser
cmanager = ConfigManger(os.path.join(CONFIG_PATH, "semi.yaml"))
config = cmanager.config
cur_githash = gethash(__file__)

tra_transforms = TensorAugment.pretrain
val_transforms = TensorAugment.val

train_dataset: 'ACDCDataset' = ACDCDatasetWithTeacherPrediction(
    root_dir=DATA_PATH, mode="train", verbose=True,
)
subtra_set_1 = extract_dataset_based_on_num_patients(10, dataset=train_dataset, seed=0)
val_dataset: 'ACDCDataset' = ACDCDatasetWithTeacherPrediction(root_dir=DATA_PATH, mode="val", verbose=True)

sampler = InfiniteRandomSampler(train_dataset, shuffle=True)
train_loader = DataLoader(train_dataset,
                          batch_size=config["LabeledData"]["batch_size"],
                          num_workers=config["LabeledData"]["num_workers"],
                          sampler=sampler,
                          pin_memory=False)

val_loader = DataLoader(val_dataset,
                        batch_size=config["ValidationData"]["batch_size"],
                        num_workers=config["ValidationData"]["num_workers"],
                        pin_memory=False)

# set reproducibility
set_benchmark(config.get("RandomSeed", 1))
trainer_name = config["Trainer"].pop("name")
Trainer = trainer_zoos[trainer_name]
model = UNet(**config["Arch"], seq_len=config["Iterations"]["num_iter"])
trainer = Trainer(
    model=model, labeled_loader=train_loader, val_loader=val_loader, sup_criterion=KL_div(), test_loader=None,
    memory_bank=None,
    num_iter=config["Iterations"]['num_iter'], alpha=config["Aggregator"]["alpha"],
    configuration={**cmanager.config, **{"GITHASH": cur_githash}}, **config["Trainer"],
    tra_augment=tra_transforms, val_augment=val_transforms
)

trainer.init()

checkpoint = config.get("Checkpoint", None)
if checkpoint is not None:
    trainer.load_state_dict_from_path(checkpoint, strict=False)
trainer.start_training()
# trainer.inference(checkpoint=checkpoint)
