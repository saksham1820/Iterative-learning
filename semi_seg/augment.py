from deepclustering2.augment import pil_augment
from torchvision import transforms

from contrastyou.augment.sequential_wrapper import SequentialWrapperTwice, SequentialWrapper, \
    switch_interpolation_kornia
from contrastyou.trainer._utils import Identical


class ACDCStrongTransforms:
    pretrain = SequentialWrapperTwice(
        com_transform=pil_augment.Compose([
            pil_augment.RandomRotation(45),
            pil_augment.RandomVerticalFlip(),
            pil_augment.RandomHorizontalFlip(),
            pil_augment.RandomCrop(224),

        ]),
        image_transform=pil_augment.Compose([
            transforms.ColorJitter(brightness=[0.5, 1.5], contrast=[0.5, 1.5], saturation=[0.5, 1.5]),
            transforms.ToTensor()
        ]),
        target_transform=pil_augment.Compose([
            pil_augment.ToLabel()
        ]),
        total_freedom=True
    )
    label = SequentialWrapperTwice(
        com_transform=pil_augment.Compose([
            pil_augment.RandomCrop(224),
            pil_augment.RandomRotation(30),
        ]),
        image_transform=pil_augment.Compose([
            transforms.ToTensor()
        ]),
        target_transform=pil_augment.Compose([
            pil_augment.ToLabel()
        ]),
    )
    val = SequentialWrapper(
        target_transform=pil_augment.ToLabel(),
        com_transform=pil_augment.CenterCrop(224),
        image_transform=pil_augment.ToTensor()
    )

    trainval = SequentialWrapperTwice(
        com_transform=pil_augment.Compose([
            pil_augment.RandomCrop(224),

        ]),
        image_transform=pil_augment.Compose([
            transforms.ToTensor()
        ]),
        target_transform=pil_augment.Compose([
            pil_augment.ToLabel()
        ]),
        total_freedom=True
    )


class ACDCTensorTransforms:
    pretrain = SequentialWrapper(
        com_transform=transforms.Compose([
            transforms.RandomRotation(45),
            transforms.RandomVerticalFlip(),
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(224),

        ]),
        image_transform=transforms.Compose([
            transforms.ColorJitter(brightness=[0.5, 1.5], ),
        ]),
        target_transform=Identical(),

    )
    label = SequentialWrapper(
        com_transform=transforms.Compose([
            transforms.RandomCrop(224),
            transforms.RandomRotation(30),
        ]),
        target_transform=Identical(),
        image_transform=Identical()

    )
    val = SequentialWrapper(
        com_transform=transforms.CenterCrop(224),
        target_transform=Identical(),
        image_transform=Identical()
    )

    trainval = SequentialWrapper(
        com_transform=transforms.Compose([
            transforms.RandomCrop(224),

        ]),
        target_transform=Identical(),
        image_transform=Identical()
    )


from kornia.augmentation import RandomRotation, RandomVerticalFlip, RandomHorizontalFlip, RandomCrop, \
    CenterCrop, ColorJitter


class TensorAugment:
    pretrain = SequentialWrapper(
        com_transform=transforms.Compose([
            RandomRotation(45, p=0.9),
            RandomVerticalFlip(),
            RandomHorizontalFlip(),
            RandomCrop(size=(224, 224), ),
        ]),
        image_transform=transforms.Compose([
            # ColorJitter(brightness=[0.5, 1.5], ),
        ]),
        target_transform=Identical(),
        switch_interpo=switch_interpolation_kornia
    )
    label = SequentialWrapper(
        com_transform=transforms.Compose([
            RandomCrop(size=(224, 224), ),
            RandomRotation(30),
        ]),
        target_transform=Identical(),
        image_transform=Identical(),
        switch_interpo=switch_interpolation_kornia
    )
    val = SequentialWrapper(
        com_transform=CenterCrop(224),
        target_transform=Identical(),
        image_transform=Identical(),
        switch_interpo=switch_interpolation_kornia
    )
