import numpy as np
import os
import typing as t
from collections import OrderedDict
from contrastyou.augment import SequentialWrapper
from contrastyou.augment.helper import fix_all_seed_within_context
from copy import deepcopy as dcopy
from deepclustering2.utils import path2Path

if t.TYPE_CHECKING:
    from contrastyou.dataloader import ACDCDataset


def extract_data(train_dataset: 'ACDCDataset', pred_folder: str):
    ls = {'img': [], 'gt': []}
    files_to_match = os.listdir(
        pred_folder)
    for i in range(len(train_dataset._filenames['img'])):
        if train_dataset._filenames['img'][i].split('/')[-1] + '.npy' in files_to_match:
            ls['img'].append(train_dataset._filenames['img'][i])
            ls['gt'].append(train_dataset._filenames['gt'][i])
    train_dataset._filenames['img'] = ls['img']
    train_dataset._filenames['gt'] = ls['gt']
    return train_dataset


def get_stem(path):
    return path2Path(path).stem


def extract_sub_dataset_based_on_scan_names(dataset: 'ACDCDataset', group_names: t.Iterable[str],
                                            transforms: SequentialWrapper = None) -> 'ACDCDataset':
    available_group_names = sorted(set(dataset.get_group_list()))
    for g in group_names:
        assert g in available_group_names, (g, available_group_names)
    memory = dataset._filenames  # noqa
    get_scan_name = dataset._get_group_name  # noqa
    new_memory = OrderedDict()
    for sub_folder, path_list in memory.items():
        new_memory[sub_folder] = [x for x in path_list if get_scan_name(get_stem(x)) in group_names]

    new_dataset = dcopy(dataset)
    new_dataset._filenames = new_memory
    if transforms:
        new_dataset.transforms = transforms
    assert set(new_dataset.get_group_list()) == set(group_names)

    return new_dataset


def extract_dataset_based_on_num_patients(*num_patients: int, dataset: 'ACDCDataset', seed=0) -> t.Tuple[
    'ACDCDataset', ...]:
    total_patient_num = len(dataset.get_group_list())
    presumed_total_patient_num = sum(num_patients)
    assert total_patient_num >= presumed_total_patient_num

    scan_list = sorted(set(dataset.get_group_list()))
    with fix_all_seed_within_context(seed):
        scan_list_permuted = np.random.permutation(scan_list).tolist()

    def _sum_iter(ratio_list):
        sum_ = 0
        for i in ratio_list:
            yield sum_ + i
            sum_ += i

    def _two_element_iter(cut_list):
        previous = 0
        for r in cut_list:
            yield previous, r
            previous = r
        # yield previous, len(scan_list)

    cutting_points = [x for x in _sum_iter(num_patients)]

    sub_datasets = [extract_sub_dataset_based_on_scan_names(dataset, scan_list_permuted[x:y]) for x, y in
                    _two_element_iter(cutting_points)]
    assert sum([len(set(x.get_scan_list())) for x in sub_datasets]) == len(scan_list)
    return sub_datasets
