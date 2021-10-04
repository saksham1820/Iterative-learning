"""
        if self._exp:
            if self._tod == 'train':
                mask_path = self._root_dir + '/train/teacher/preds/' + filename + '.png.npy'
                mask = np.load(mask_path)
                mask_channel = [mask[i, :, :] for i in range(4)]
                mask_pil = [Image.fromarray((mask_channel[i] * 255).astype(np.uint8)) for i in range(4)]
                data = self._transform(imgs=[mask_pil[0], mask_pil[1], mask_pil[2], mask_pil[3], img_png],
                                       targets=[target_png])
                transformed_mask = torch.cat([data[0].pop(0) for _ in range(4)])
                transformed_mask_tf = torch.cat([data[1].pop(0) for _ in range(4)])
                data = [[torch.cat([transformed_mask, data[0][0]]), data[0][1]],
                        [torch.cat([transformed_mask_tf, data[1][0]]), data[1][1]]]
                breakpoint()
            elif self._tod == 'val':
                mask_path = self._root_dir + '/val/val_masks/' + filename + '.npy'
                mask = np.load(mask_path)
                mask_channel = [mask[i, :, :] for i in range(4)]
                mask_pil = [Image.fromarray((mask_channel[i] * 255).astype(np.uint8)) for i in range(4)]
                data = self._transform(imgs = [mask_pil[0], mask_pil[1], mask_pil[2], mask_pil[3], img_png],
                                       targets = [target_png])
                transformed_mask = torch.cat([data.pop(0) for _ in range(4)])
                data = [torch.cat([transformed_mask, data[0]]), data[1]]
        else:
"""#TODO:add this block below and then perform iterative segmentation
import os
import re
from pathlib import Path
from typing import List, Tuple, Union
import matplotlib.pyplot as plt
import torch
from PIL import Image
import numpy as np
from torch import Tensor

from contrastyou.augment.sequential_wrapper import SequentialWrapper
from contrastyou.dataloader._seg_datset import ContrastDataset
from deepclustering2.dataset import ACDCDataset as _ACDCDataset, ACDCSemiInterface as _ACDCSemiInterface


class ACDCDataset(ContrastDataset, _ACDCDataset):
    download_link = "https://drive.google.com/uc?id=1SMAS6R46BOafLKE9T8MDSVGAiavXPV-E"
    zip_name = "ACDC_contrast.zip"
    folder_name = "ACDC_contrast"

    def __init__(self, tod, exp, root_dir: str, mode: str, transforms: SequentialWrapper = SequentialWrapper(),
                 verbose=True, *args, **kwargs) -> None:
        super().__init__(root_dir, mode, ["img", "gt"], transforms, verbose)
        self._acdc_info = np.load(os.path.join(self._root_dir, "acdc_info.npy"), allow_pickle=True).item()
        assert isinstance(self._acdc_info, dict) and len(self._acdc_info) == 200
        self._tod = tod
        self._exp = exp
        self._transform = transforms

    def __getitem__(self, index) -> Tuple[List[Tensor], str, str, str]:
        [img_png, target_png], filename_list = self._getitem_index(index)
        filename = Path(filename_list[0]).stem
        #add the block here
        if self._exp:
            #
            # if self._tod == 'train':
            #     mask_path = self._root_dir + '/train/train_masks_from_teacher/preds/' + filename + '.png.npy'
            #     mask = np.load(mask_path)
            #     mask_channel = [mask[i, :, :] for i in range(4)]
            #     mask_pil = [Image.fromarray((mask_channel[i] * 255).astype(np.uint8)) for i in range(4)]
            #     data = self._transform(imgs=[mask_pil[0], mask_pil[1], mask_pil[2], mask_pil[3], img_png],
            #                            targets=[target_png])
            #     transformed_mask = torch.cat([data[0].pop(0) for _ in range(4)])
            #     transformed_mask_tf = torch.cat([data[1].pop(0) for _ in range(4)])
            #     data = [[torch.cat([transformed_mask, data[0][0]]), data[0][1]],
            #             [torch.cat([transformed_mask_tf, data[1][0]]), data[1][1]]]
            if self._tod == 'train':
                mask_path = self._root_dir + '/train/train_masks_from_teacher/preds/' + filename + '.png.npy'
                mask = np.load(mask_path)
                mask_channel = [mask[i, :, :] for i in range(4)]

                mask_pil = [Image.fromarray((mask_channel[i] * 255).astype(np.uint8)) for i in range(4)]
                data = self._transform(imgs = [mask_pil[0], mask_pil[1], mask_pil[2], mask_pil[3], img_png],
                                       targets = [target_png])
                transformed_mask = torch.cat([data.pop(0) for _ in range(4)])
                data = [torch.cat([transformed_mask, data[0]]), data[1]]

            elif self._tod == 'val':
                mask_path = self._root_dir + '/val/val_masks_from_teacher/' + filename + '.png.npy'
                mask = np.load(mask_path)
                mask_channel = [mask[i, :, :] for i in range(4)]

                mask_pil = [Image.fromarray((mask_channel[i] * 255).astype(np.uint8)) for i in range(4)]
                data = self._transform(imgs = [mask_pil[0], mask_pil[1], mask_pil[2], mask_pil[3], img_png],
                                       targets = [target_png])
                transformed_mask = torch.cat([data.pop(0) for _ in range(4)])
                data = [torch.cat([transformed_mask, data[0]]), data[1]]
        else:
            data = self._transform(imgs=[img_png], targets=[target_png])
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


class ACDCSemiInterface(_ACDCSemiInterface):

    def __init__(self, root_dir, labeled_data_ratio: float = 0.2, unlabeled_data_ratio: float = 0.8,
                 seed: int = 0, verbose: bool = True) -> None:
        super().__init__(root_dir, labeled_data_ratio, unlabeled_data_ratio, seed, verbose)
        self.DataClass = ACDCDataset
