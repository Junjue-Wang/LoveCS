from albumentations import HorizontalFlip, VerticalFlip, RandomRotate90, Normalize, RandomCrop, RandomScale
from albumentations import OneOf, Compose
import ever as er
from ever.api.preprocess.albu import RandomDiscreteScale

TARGET_SET = 'ToSpaceborne'
NUM_CLASSES = 4
source_dir = dict(
    image_dir=[
        './cross_sensor/src_domain/images/',
    ],
    mask_dir=[
        './cross_sensor/src_domain/masks/',
    ],
)
target_dir = dict(
    image_dir=[
        './cross_sensor/tgt_domain/images/',
    ],
    mask_dir=[
        './cross_sensor/tgt_domain/masks/',
    ],
)


SOURCE_DATA_CONFIG = dict(
    image_dir=source_dir['image_dir'],
    mask_dir=source_dir['mask_dir'],
    transforms=Compose([
        RandomDiscreteScale([0.5, 0.75, 1.25, 1.5, 1.75]), 
        RandomCrop(512, 512),
        OneOf([
            HorizontalFlip(True),
            VerticalFlip(True),
            RandomRotate90(True)
        ], p=0.75),
        Normalize(mean=(123.675, 116.28, 103.53),
                  std=(58.395, 57.12, 57.375),
                  max_pixel_value=1, always_apply=True),
        er.preprocess.albu.ToTensor()

    ]),
    CV=dict(k=10, i=-1),
    training=True,
    batch_size=8,
    num_workers=2,
    reclassify=True,
)


TARGET_DATA_CONFIG = dict(
    image_dir=target_dir['image_dir'],
    mask_dir=target_dir['mask_dir'],
    transforms=Compose([
        RandomDiscreteScale([0.5, 0.75, 1.25, 1.5, 1.75]), 
        RandomCrop(512, 512),
        OneOf([
            HorizontalFlip(True),
            VerticalFlip(True),
            RandomRotate90(True)
        ], p=0.75),
        Normalize(mean=(123.675, 116.28, 103.53),
                  std=(58.395, 57.12, 57.375),
                  max_pixel_value=1, always_apply=True),
        er.preprocess.albu.ToTensor()

    ]),
    CV=dict(k=10, i=-1),
    training=True,
    batch_size=8,
    num_workers=2,
    reclassify=False,
)

EVAL_DATA_CONFIG = dict(
    image_dir=target_dir['image_dir'],
    mask_dir=target_dir['mask_dir'],
    transforms=Compose([
        Normalize(mean=(123.675, 116.28, 103.53),
                  std=(58.395, 57.12, 57.375),
                  max_pixel_value=1, always_apply=True),
        er.preprocess.albu.ToTensor()

    ]),
    CV=dict(k=10, i=-1),
    training=False,
    batch_size=1,
    reclassify=True,
    num_workers=0,
    drop_last=False,
)
