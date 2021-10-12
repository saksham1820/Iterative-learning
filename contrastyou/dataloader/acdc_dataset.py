import numpy as np
import os
import re
import torch
from PIL import Image
from contrastyou.augment.helper import ToLabel, Identity
from contrastyou.augment.sequential_wrapper import SequentialWrapper
from contrastyou.dataloader._seg_datset import ContrastDataset
from copy import deepcopy
from deepclustering2.dataset import ACDCDataset as _ACDCDataset, ACDCSemiInterface as _ACDCSemiInterface
from pathlib import Path
from torch import Tensor
from torchvision import transforms
from typing import List, Tuple, Union

default_cpu_augment = SequentialWrapper(image_transform=transforms.ToTensor(), target_transform=ToLabel(),
                                        com_transform=Identity())


class ACDCDataset(ContrastDataset, _ACDCDataset):
    download_link = "https://drive.google.com/uc?id=1SMAS6R46BOafLKE9T8MDSVGAiavXPV-E"
    zip_name = "ACDC_contrast.zip"
    folder_name = "ACDC_contrast"

    def __init__(self, root_dir: str, mode: str, transforms: SequentialWrapper = None,
                 verbose=True, *args, **kwargs) -> None:
        super().__init__(root_dir, mode, ["img", "gt"], transforms, verbose)
        self._acdc_info = np.load(os.path.join(self._root_dir, "acdc_info.npy"), allow_pickle=True).item()
        assert isinstance(self._acdc_info, dict) and len(self._acdc_info) == 200

        self._transform = transforms or default_cpu_augment

    def __getitem__(self, index) -> Tuple[List[Tensor], str, str, str]:
        [img_png, target_png], filename_list = self._getitem_index(index)
        filename = Path(filename_list[0]).stem
        data = self._transform(images=[img_png], targets=[target_png])
        partition = self._get_partition(filename)
        group = self._get_group(filename)
        return data, filename, partition, group

    def _get_group(self, filename) -> Union[str, int]:
        return str(self._get_group_name(filename))

    def _get_partition(self, filename) -> Union[str, int]:
        # set partition
        max_len_given_group = self._acdc_info[self._get_group_name(filename)]
        cutting_point = max_len_given_group // 3
        cur_index = int(re.compile(r"\d+").findall(filename)[-1])
        if cur_index <= cutting_point - 1:
            return str(0)
        if cur_index <= 2 * cutting_point:
            return str(1)
        return str(2)

    def show_paritions(self) -> List[Union[str, int]]:
        return [self._get_partition(f) for f in list(self._filenames.values())[0]]

    def show_groups(self) -> List[Union[str, int]]:
        return [self._get_group(f) for f in list(self._filenames.values())[0]]


class ACDCDatasetWithTeacherPrediction(ACDCDataset):
    download_link = "https://drive.google.com/uc?id=1SMAS6R46BOafLKE9T8MDSVGAiavXPV-E"
    zip_name = "ACDC_contrast_with_prediction.zip"
    folder_name = "ACDC_contrast_with_prediction"

    def __init__(self, root_dir: str, mode: str, transforms: SequentialWrapper = None, verbose=True,
                 **kwargs) -> None:
        super().__init__(root_dir, mode, transforms, verbose, **kwargs)
        mask_name = "pred_arrays"

        self._mask_folder = os.path.join(self._root_dir, self._mode, mask_name)
        assert Path(self._mask_folder).exists() and Path(self._mask_folder).is_dir(), self._mask_folder

    def __getitem__(self, index) -> Tuple[Tensor, Tensor, str, str, str, Tensor]:
        [img_png, target_png], filename_list = self._getitem_index(index)
        filename = Path(filename_list[0]).stem

        mask_path = os.path.join(self._mask_folder, filename + ".png.npy")
        mask = np.load(mask_path)
        assert (0 <= mask).all() and (mask <= 1).all()
        assert np.allclose(mask.sum(0), np.ones_like(mask.sum(0)))

        mask_pil = [Image.fromarray((mask[i] * 255).astype(np.uint8)) for i in range(4)]
        data = self._transform(images=[img_png, mask_pil[0], mask_pil[1], mask_pil[2], mask_pil[3]],
                               targets=[target_png])
        image, prev_pred, target = data[0][0], torch.cat(data[0][1:], dim=0), data[-1][0]

        partition = self._get_partition(filename)
        group = self._get_group(filename)
        return image, target, filename, partition, group, prev_pred


class ACDCSemiInterface(_ACDCSemiInterface):

    def __init__(self, root_dir, labeled_data_ratio: float = 0.2, unlabeled_data_ratio: float = 0.8,
                 seed: int = 0, verbose: bool = True) -> None:
        super().__init__(root_dir, labeled_data_ratio, unlabeled_data_ratio, seed, verbose)
        self.DataClass = ACDCDataset


if __name__ == '__main__':
    from contrastyou import DATA_PATH
    import matplotlib.pyplot as plt

    dataset = ACDCDatasetWithTeacherPrediction(
        root_dir=DATA_PATH, mode="train", verbose=False)
    image, target, filename, partition, group, prev_pred = dataset[0]
    from torchvision import transforms

    gpu_transform = SequentialWrapper(
        com_transform=transforms.Compose([transforms.RandomCrop(size=128)]),
        image_transform=transforms.ColorJitter(brightness=10, contrast=0, saturation=0)
    )
    image_, target_ = gpu_transform(images=torch.stack([deepcopy(image) for _ in range(10)], dim=0),
                                    targets=torch.stack([target] * 10))
    from deepclustering2.viewer import multi_slice_viewer_debug

    for i in range(10):
        image_, target_ = gpu_transform(images=image, targets=target)
        multi_slice_viewer_debug([image, image_, ], block=False)
    plt.show()
