import os
import typing as tp

import cv2
import numpy as np
import pandas as pd

from augmentations import get_validation_augmentation, get_training_augmentation
from config import Config


def _onehot(size, target):
    vec = np.zeros(size)
    vec[target] = 1
    return vec


def preprocess(im: np.ndarray) -> np.ndarray:
    im = im.astype(np.float32)
    im /= 255
    im = np.transpose(im, (2, 0, 1))
    im -= np.array([0.485, 0.456, 0.406])[:, None, None]
    im /= np.array([0.229, 0.224, 0.225])[:, None, None]
    return im


_qal2target = {
    75: 0,
    90: 1,
    95: 2,
}


class ClassificationDataset:

    def __init__(self,
                 df,
                 augmentation=None,
                 preprocess=None,
                 mode='train',
                 data_root=None,
                 use_qual=False,
                 ):
        self.df = df
        self.augmentation = augmentation
        self.preprocess = preprocess
        self.mode = mode
        self.data_root = data_root
        self.use_qual = use_qual

    def __getitem__(self, idx):
        sample = self.df.iloc[idx]
        img_name, kind, label = sample['image_name'], sample['kind'], sample['label']
        if self.use_qual:
            qual = sample['quality']
            qal_targ = _onehot(3, _qal2target[qual])
        im_path = os.path.join(self.data_root, kind, img_name)
        image = cv2.imread(im_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if self.augmentation:
            image = self.augmentation(image=image)['image']
        if self.preprocess:
            image = self.preprocess(image)
        if self.mode == 'test':
            return image
        target = _onehot(4, label)
        if self.use_qual:
            return image, target, qal_targ
        return image, target

    def __len__(self):
        return len(self.df)


def get_train_val_datasets(config: Config) -> tp.Tuple[ClassificationDataset, ClassificationDataset]:
    df = pd.read_csv(config.df_folds_path)
    df_train, df_val = df[df['fold'] != config.fold_num], df[df['fold'] == config.fold_num]
    df_train = df_train.reset_index(drop=True)
    df_val = df_val.reset_index(drop=True)

    train_dataset = ClassificationDataset(df_train, get_training_augmentation(), preprocess=preprocess, mode='train',
                                          data_root=config.data_path, use_qual=config.use_qual)
    val_dataset = ClassificationDataset(df_val, get_validation_augmentation(), preprocess=preprocess, mode='val',
                                        data_root=config.data_path, use_qual=False)

    return train_dataset, val_dataset
