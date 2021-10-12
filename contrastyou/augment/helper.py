import numpy as np
import os
import random
import torch
import typing as t
from PIL import Image
from contextlib import contextmanager
from torch.utils.data.dataloader import DataLoader, _BaseDataLoaderIter  # noqa

try:
    from torch._six import container_abcs
except ImportError:
    import collections.abc as container_abcs
T = t.TypeVar("T")


class ToLabel(object):
    """
    PIL image to Label (long) with mapping (dict)
    """

    def __init__(self, mapping: t.Dict[int, int] = None) -> None:
        """
        :param mapping: Optional dictionary containing the mapping.
        """
        super().__init__()
        self.mapping_call = np.vectorize(lambda x: mapping[x]) if mapping else None

    def __call__(self, img: Image.Image):
        np_img = np.array(img)[None, ...].astype(np.float32)  # type: ignore
        if self.mapping_call:
            np_img = self.mapping_call(np_img)
        t_img = torch.from_numpy(np_img)
        return t_img.long()


class Identity:
    def __call__(self, x: T) -> T:
        return x


def fix_all_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


@contextmanager
def fix_all_seed_for_transforms(seed):
    random_state = random.getstate()
    np_state = np.random.get_state()
    torch_state = torch.random.get_rng_state()
    fix_all_seed(seed)
    try:
        yield
    finally:
        random.setstate(random_state)
        np.random.set_state(np_state)  # noqa
        torch.random.set_rng_state(torch_state)  # noqa


@contextmanager
def fix_all_seed_within_context(seed):
    random_state = random.getstate()
    np_state = np.random.get_state()
    torch_state = torch.random.get_rng_state()
    cuda_support = torch.cuda.is_available()
    if cuda_support:
        torch_cuda_state = torch.cuda.get_rng_state()
        torch_cuda_state_all = torch.cuda.get_rng_state_all()
    fix_all_seed(seed)
    try:
        yield
    finally:
        random.setstate(random_state)
        np.random.set_state(np_state)  # noqa
        torch.random.set_rng_state(torch_state)  # noqa
        if cuda_support:
            torch.cuda.set_rng_state(torch_cuda_state)  # noqa
            torch.cuda.set_rng_state_all(torch_cuda_state_all)  # noqa
