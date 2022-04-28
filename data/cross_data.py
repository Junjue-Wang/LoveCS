from torch.utils.data import Dataset, DataLoader
import glob
import os
from skimage.io import imread
from albumentations.pytorch import ToTensorV2
from albumentations import HorizontalFlip, VerticalFlip, RandomRotate90, Normalize
from albumentations import OneOf, Compose
from collections import OrderedDict
from ever.interface import ConfigurableMixin
from torch.utils.data import SequentialSampler, RandomSampler
from ever.api.data import CrossValSamplerGenerator
import numpy as np
import logging
from utils.tools import seed_worker

logger = logging.getLogger(__name__)



LABEL_MAP = OrderedDict(
    Background=-1,
    Building=1,
    Road=2,
    Water=3,
    Barren=0,
    Forest=0,
    Agricultural=0
)


def reclassify(cls):
    new_cls = np.ones_like(cls, dtype=np.int64) * -1
    for idx, label in enumerate(LABEL_MAP.values()):
        new_cls = np.where(cls == idx, np.ones_like(cls)*label, new_cls)
    return new_cls



class CSDataset(Dataset):
    def __init__(self, image_dir, mask_dir, transforms=None, reclassify=False):
        self.rgb_filepath_list = []
        self.cls_filepath_list= []
        if isinstance(image_dir, list) and isinstance(mask_dir, list):
            for img_dir_path, mask_dir_path in zip(image_dir, mask_dir):
                self.batch_generate(img_dir_path, mask_dir_path)
        elif isinstance(image_dir, list) and not isinstance(mask_dir, list):
            for img_dir_path in image_dir:
                self.batch_generate(img_dir_path, mask_dir)
        else:
            self.batch_generate(image_dir, mask_dir)

        self.transforms = transforms

        self._reclassify = reclassify


    def batch_generate(self, image_dir, mask_dir):
        rgb_filepath_list = glob.glob(os.path.join(image_dir, '*.tif'))
        rgb_filepath_list += glob.glob(os.path.join(image_dir, '*.png'))

        
        rgb_filename_list = [os.path.split(fp)[-1] for fp in rgb_filepath_list]
        cls_filepath_list = []
        if mask_dir is not None:
            for fname in rgb_filename_list:
                maskp = os.path.join(mask_dir, fname)
                if os.path.exists(maskp):
                    cls_filepath_list.append(maskp)
                else:
                    rgb_filepath_list.remove(os.path.join(image_dir, fname))
        logger.info('Dataset images: %d' % len(rgb_filepath_list))
        self.rgb_filepath_list += rgb_filepath_list
        self.cls_filepath_list += cls_filepath_list

    def __getitem__(self, idx):
        image = imread(self.rgb_filepath_list[idx])
        mask = imread(self.cls_filepath_list[idx]).astype(np.long) -1
        if self._reclassify:
            mask = reclassify(mask)
        if self.transforms is not None:
            blob = self.transforms(image=image, mask=mask)
            image = blob['image']
            mask = blob['mask']

        return image, dict(cls=mask, fname=os.path.basename(self.rgb_filepath_list[idx]))

    def __len__(self):
        return len(self.rgb_filepath_list)



class CSLoader(DataLoader, ConfigurableMixin):
    def __init__(self, config):
        ConfigurableMixin.__init__(self, config)
        dataset = CSDataset(self.config.image_dir, self.config.mask_dir, self.config.transforms, reclassify=self.config.reclassify)

        if self.config.CV.i != -1:
            CV = CrossValSamplerGenerator(dataset, distributed=True, seed=2333)
            sampler_pairs = CV.k_fold(self.config.CV.k)
            train_sampler, val_sampler = sampler_pairs[self.config.CV.i]
            if self.config.training:
                sampler = train_sampler
            else:
                sampler = val_sampler
        else:
            sampler = RandomSampler(dataset) if self.config.training else SequentialSampler(
                dataset)

        super(CSLoader, self).__init__(dataset,
                                       self.config.batch_size,
                                       sampler=sampler,
                                       num_workers=self.config.num_workers,
                                       pin_memory=True,
                                       drop_last=self.config.drop_last
                                       )
    def set_default_config(self):
        self.config.update(dict(
            image_dir=None,
            mask_dir=None,
            batch_size=4,
            num_workers=4,
            scale_size=None,
            reclassify=False,
            drop_last=True,
            transforms=Compose([
                OneOf([
                    HorizontalFlip(True),
                    VerticalFlip(True),
                    RandomRotate90(True),
                ], p=0.75),
                Normalize(mean=(), std=(), max_pixel_value=1, always_apply=True),
                ToTensorV2()
            ]),
        ))
