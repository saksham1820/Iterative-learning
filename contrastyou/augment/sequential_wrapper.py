import functools
import random
import warnings
from collections import OrderedDict
from contextlib import contextmanager
from functools import lru_cache
from typing import Callable, List, Tuple, Union, TypeVar, Iterable, overload, Type

from PIL import Image
from torch import Tensor
from torchvision.transforms import Compose, InterpolationMode

from contrastyou.augment.helper import fix_all_seed_for_transforms, ToLabel

_InputType = TypeVar("_InputType")
_OutputType = TypeVar("_OutputType")
T = TypeVar("T")

_TransformType = Callable[[_InputType], _OutputType]

_Pil2Pil_T = _TransformType[Image.Image, Image.Image]
_Pil2Tensor_T = _TransformType[Image.Image, Tensor]

_TypeList = List[T]
_Pil_List = _TypeList[Image.Image]

_ItemOrIterable = Union[T, Iterable[T]]
_Pil_or_Iterable = _ItemOrIterable[Image.Image]
_Tensor_or_Iterable = _ItemOrIterable[Tensor]

__all__ = ["SequentialWrapper", "SequentialWrapperTwice"]


def get_transform(transform: _TransformType) -> Iterable[_TransformType]:
    if isinstance(transform, Compose):
        for x in transform.transforms:
            yield from get_transform(x)
    else:
        yield transform


def is_tuple_or_list(item) -> bool:
    return isinstance(item, (list, tuple))


@lru_cache()
def get_interpolation(interp: str) -> InterpolationMode:
    return {"bilinear": InterpolationMode.BILINEAR, "nearest": InterpolationMode.NEAREST}[interp]


@lru_cache()
def get_interpolation_kornia(interp: str) -> InterpolationMode:
    from kornia.constants import Resample
    return {"bilinear": Resample.BILINEAR, "nearest": Resample.NEAREST}[interp]


@contextmanager
def switch_interpolation_kornia(transforms: _TransformType, *, interp: str):
    assert interp in ("bilinear", "nearest"), interp  # noqa
    previous_inters = OrderedDict()
    transforms_iter = get_transform(transforms)
    interpolation = get_interpolation_kornia(interp)
    for id_, t in enumerate(transforms_iter):
        if hasattr(t, "resample"):
            previous_inters[id_] = t.resample
            t.interpolation = interpolation
    try:
        yield
    finally:
        transforms_iter = get_transform(transforms)
        for id_, t in enumerate(transforms_iter):
            if hasattr(t, "resample"):
                t.interpolation = previous_inters[id_]


@contextmanager
def switch_interpolation_torchvision(transforms: _TransformType, *, interp: str):
    assert interp in ("bilinear", "nearest"), interp  # noqa
    previous_inters = OrderedDict()
    transforms_iter = get_transform(transforms)
    interpolation = get_interpolation(interp)
    for id_, t in enumerate(transforms_iter):
        if hasattr(t, "interpolation"):
            previous_inters[id_] = t.interpolation
            t.interpolation = interpolation
    try:
        yield
    finally:
        transforms_iter = get_transform(transforms)
        for id_, t in enumerate(transforms_iter):
            if hasattr(t, "interpolation"):
                t.interpolation = previous_inters[id_]


# todo: adding switch_interpolation for kornia library.


def random_int() -> int:
    return random.randint(0, int(1e5))


def transform_(image: _InputType, transform: _TransformType, seed: int) -> _OutputType:
    with fix_all_seed_for_transforms(seed):
        return transform(image)


def warning_suppress(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return func(*args, **kwargs)

    return wrapper


class SequentialWrapper:
    @overload
    def __init__(self, *, com_transform: _TransformType[Image.Image, Image.Image],
                 image_transform: _TransformType[Image.Image, Tensor],
                 target_transform: _TransformType[Image.Image, Tensor] = ToLabel(),
                 switch_interpo: Type[switch_interpolation_torchvision]) -> None:
        pass

    @overload
    def __init__(self, *, com_transform: _TransformType[Tensor, Tensor],
                 image_transform: _TransformType[Tensor, Tensor],
                 target_transform: _TransformType[Tensor, Tensor] = ToLabel(),
                 switch_interpo: Type[switch_interpolation_kornia]) -> None:
        pass

    def __init__(self, *, com_transform=None, image_transform, target_transform=ToLabel(),
                 switch_interpo=switch_interpolation_torchvision) -> None:
        """
        image -> comm_transform -> img_transform -> Tensor
        target -> comm_transform -> target_transform -> Tensor
        :param com_transform: common geo-transformation
        :param image_transform: transformation only applied for images
        :param target_transform: transformation only applied for targets
        """
        self._com_transform = com_transform
        self._image_transform = image_transform
        self._target_transform = target_transform

        self.switch_interpo = switch_interpo

    @overload
    def __call__(self, images: _Pil_or_Iterable, targets: _Pil_or_Iterable = None, com_seed: int = None,
                 img_seed: int = None, target_seed: int = None) -> Tuple[_Tensor_or_Iterable, _Tensor_or_Iterable]:
        pass

    @overload
    def __call__(self, images: _Tensor_or_Iterable, targets: _Tensor_or_Iterable = None, com_seed: int = None,
                 img_seed: int = None, target_seed: int = None) -> Tuple[_Tensor_or_Iterable, _Tensor_or_Iterable]:
        pass

    def __call__(self, images, targets=None, com_seed: int = None, img_seed: int = None, target_seed: int = None) -> \
        Tuple[_Tensor_or_Iterable, _Tensor_or_Iterable]:
        com_seed = com_seed or random_int()
        img_seed = img_seed or random_int()
        target_seed = target_seed or random_int()

        is_image_tuple = is_tuple_or_list(images)
        is_target_tuple = is_tuple_or_list(targets)

        if not is_image_tuple:
            images = [images, ]
        if not is_target_tuple and targets is not None:
            targets = [targets, ]

        image_list_after_transform, target_list_after_transform = images, targets or []

        if self._com_transform:
            # comm is the optional
            with self.switch_interpo(self._com_transform, interp="bilinear"):
                image_list_after_transform = [transform_(image, self._com_transform, com_seed)
                                              for image in image_list_after_transform]
            if targets is not None:
                with self.switch_interpo(self._com_transform, interp="nearest"):
                    target_list_after_transform = [transform_(target, self._com_transform, com_seed)
                                                   for target in target_list_after_transform]

        image_list_after_transform = [transform_(image, self._image_transform, img_seed)
                                      for image in image_list_after_transform]

        if targets is not None:
            with self.switch_interpo(self._target_transform, interp="nearest"):
                target_list_after_transform = [transform_(target, self._target_transform, target_seed)
                                               for target in target_list_after_transform]
        if not is_image_tuple:
            image_list_after_transform = image_list_after_transform[0]
        if not is_target_tuple and targets is not None:
            target_list_after_transform = target_list_after_transform[0]

        return image_list_after_transform, target_list_after_transform

    def __repr__(self):
        return (
            f"comm_transform:{self._com_transform}\n"
            f"img_transform:{self._image_transform}.\n"
            f"target_transform: {self._target_transform}"
        )


class SequentialWrapperTwice(SequentialWrapper):

    def __init__(self, *, com_transform=None, image_transform, target_transform, total_freedom=True) -> None:
        """
        :param total_freedom: if True, the two-time generated images are using different seeds for all aspect,
                              otherwise, the images are used different random seed only for img_seed
        """
        super().__init__(com_transform=com_transform, image_transform=image_transform,
                         target_transform=target_transform)
        self._total_freedom = total_freedom

    def __call__(self, images, targets=None, seed: int = None, **kwargs) -> \
        Tuple[_Tensor_or_Iterable, _Tensor_or_Iterable]:
        seed = seed or random_int()

        with fix_all_seed_for_transforms(seed):
            comm_seed1, comm_seed2 = random_int(), random_int()
            img_seed1, img_seed2 = random_int(), random_int()
            target_seed1, target_seed2 = random_int(), random_int()

            if self._total_freedom:
                images1, targets1 = super(SequentialWrapperTwice, self).__call__(images, targets,
                                                                                 comm_seed1, img_seed1,
                                                                                 target_seed1)
                images2, targets2 = super(SequentialWrapperTwice, self).__call__(images, targets,
                                                                                 comm_seed2, img_seed2,
                                                                                 target_seed2)
                return [*images1, *images2], [*targets1, *targets2]

            images1, targets1 = super(SequentialWrapperTwice, self).__call__(images, targets,
                                                                             comm_seed1, img_seed1,
                                                                             target_seed1)
            images2, targets2 = super(SequentialWrapperTwice, self).__call__(images, targets,
                                                                             comm_seed1, img_seed2,
                                                                             target_seed1)
            return [*images1, *images2], [*targets1, *targets2]
