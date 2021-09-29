from deepclustering2.dataloader.sampler import InfiniteRandomSampler
from scipy.sparse import issparse  # noqa

_ = issparse  # noqa
import warnings
from deepclustering2.loss import KL_div

from pathlib import Path
from contrastyou import PROJECT_PATH
from contrastyou.arch import UNet

from contrastyou.dataloader._seg_datset import ContrastBatchSampler  # noqa
from deepclustering2.configparser import ConfigManger
from deepclustering2.utils import gethash
from deepclustering2.utils import set_benchmark
from semi_seg.trainer import trainer_zoos

from contrastyou import DATA_PATH
from contrastyou.dataloader.acdc_dataset import ACDCDataset
from semi_seg.augment import ACDCStrongTransforms
import os
from torch.utils.data import DataLoader

warnings.simplefilter("ignore")

# load configure from yaml and argparser
cmanager = ConfigManger(Path(PROJECT_PATH) / "config/semi.yaml")
config = cmanager.config
cur_githash = gethash(__file__)

tra_transforms = ACDCStrongTransforms.pretrain
val_transforms = ACDCStrongTransforms.val

val_dataset = ACDCDataset(exp=True, tod='val', root_dir=DATA_PATH, mode="val", transforms=val_transforms, verbose=True)
train_dataset = ACDCDataset(exp=True, tod='train', root_dir=DATA_PATH, mode="train", transforms=val_transforms,
                            verbose=True)
ls = {'img': [], 'gt': []}
files_to_match = os.listdir(
    '/home/saksham/Iterative-learning/.data/ACDC_contrast/train/train_masks_from_teacher/preds')
for i in range(len(train_dataset._filenames['img'])):
    if train_dataset._filenames['img'][i].split('/')[-1] + '.npy' in files_to_match:
        ls['img'].append(train_dataset._filenames['img'][i])
        ls['gt'].append(train_dataset._filenames['gt'][i])

train_dataset._filenames['img'] = ls['img']
train_dataset._filenames['gt'] = ls['gt']

sampler = InfiniteRandomSampler(train_dataset, shuffle=True)

train_loader = DataLoader(train_dataset,
                          batch_size=config["LabeledData"]["batch_size"],
                          num_workers=config["LabeledData"]["num_workers"],
                          sampler=sampler, pin_memory=False)

val_loader = DataLoader(val_dataset,
                        batch_size=config["ValidationData"]["batch_size"],
                        num_workers=config["ValidationData"]["num_workers"],
                        pin_memory=False)
mem_bank = {}
for i in range(len(train_dataset)):
    mem_bank[train_dataset[i][-3]] = None


# set reproducibility
set_benchmark(config.get("RandomSeed", 1))
trainer_name = config["Trainer"].pop("name")
Trainer = trainer_zoos[trainer_name]
model = UNet(**config["Arch"])
trainer = Trainer(
    model=model, labeled_loader=train_loader, memory_bank = mem_bank,
    val_loader=val_loader, sup_criterion=KL_div(), test_loader=None,
    num_iter=config["Iterations"]['num_iter'], alpha=config["Aggregator"]["alpha"],
    configuration={**cmanager.config, **{"GITHASH": cur_githash}}, **config["Trainer"])

trainer.init()
checkpoint = config.get("Checkpoint", None)
if checkpoint is not None:
    trainer.load_state_dict_from_path(checkpoint, strict=False)
trainer.start_training()
# trainer.inference(checkpoint=checkpoint)
