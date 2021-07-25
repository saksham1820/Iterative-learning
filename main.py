
from scipy.sparse import issparse  # noqa

_ = issparse  # noqa

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

from torch.utils.data import DataLoader

# load configure from yaml and argparser
cmanager = ConfigManger(Path(PROJECT_PATH) / "config/semi.yaml")
config = cmanager.config
cur_githash = gethash(__file__)

tra_transforms = ACDCStrongTransforms.pretrain
val_transforms = ACDCStrongTransforms.val

tra_dataset = ACDCDataset(root_dir=DATA_PATH, mode="train", transforms=tra_transforms, verbose=True)
val_dataset = ACDCDataset(root_dir=DATA_PATH, mode="val", transforms=val_transforms, verbose=True)

train_loader = DataLoader(tra_dataset,
                          batch_size = config["LabeledData"]["batch_size"],
                          num_workers = config["LabeledData"]["num_workers"],
                          shuffle = True)
val_loader = DataLoader(val_dataset,
                        batch_size = config["ValidationData"]["batch_size"],
                        num_workers = config["ValidationData"]["num_workers"])


# set reproducibility
set_benchmark(config.get("RandomSeed", 1))

trainer_name = config["Trainer"].pop("name")
Trainer = trainer_zoos[trainer_name]
model = UNet(**config["Arch"])
trainer = Trainer(
    model=model, labeled_loader=train_loader, alpha = config["Aggregator"]["alpha"],
    num_iter = config["Iterations"]["num_iter"],
    val_loader=val_loader, sup_criterion=KL_div(),
    configuration={**cmanager.config, **{"GITHASH": cur_githash}},
    **config["Trainer"]
)
trainer.init()
checkpoint = config.get("Checkpoint", None)
if checkpoint is not None:
    trainer.load_state_dict_from_path(checkpoint, strict=False)
trainer.start_training()
# trainer.inference(checkpoint=checkpoint)

