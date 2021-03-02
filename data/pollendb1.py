"""POLLEN Dataset Classes

Original author: Francisco Massa
https://github.com/fmassa/vision/blob/voc_dataset/torchvision/datasets/voc.py

Updated by: Ellis Brown, Max deGroot
"""
import os.path as osp
import pickle

import cv2
import numpy as np
import torch
import torch.utils.data as data

from .config import HOME

POLLEN_CLASSES = ('grain',)  # always index 0

# note: if you used our download scripts, this should be right
POLLEN_ROOT = osp.join(HOME, "data/POLLENDB1/")


class POLLENDetection(data.Dataset):
    """POLLEN Detection Dataset Object

    input is image, target is annotation

    Arguments:
        root (string): filepath to POLLENDB1 folder.
        image_set (string): imageset to use (eg. 'train', 'val', 'test')
        transform (callable, optional): transformation to perform on the
            input image
        target_transform (callable, optional): transformation to perform on the
            target `annotation`
            (eg: take in caption string, return tensor of word indices)
        dataset_name (string, optional): which dataset to load
            (default: 'POLLENDB1')
    """

    def __init__(
        self,
        root,
        image_sets=[('2019', 'train')],
        transform=None,
        target_transform=None,
        dataset_name='POLLENDB1',
    ):
        self.root = root
        self.image_set = image_sets
        self.transform = transform
        self.target_transform = target_transform
        self.dims = np.array([640, 512, 640, 512])
        self.name = dataset_name
        self._annopath = osp.join('%s', 'Annotations', '%s.pkl')
        self._imgpath = osp.join('%s', 'JPEGImages', '%s')
        self.ids = list()
        for (year, name) in image_sets:
            rootpath = osp.join(self.root, 'POLLEN' + year)
            with open(self._annopath % (rootpath, name), 'rb') as fp:
                self.targets = pickle.load(fp)
                self.ids.extend([(rootpath, i) for i in self.targets.keys()])

    def __getitem__(self, index):
        im, gt, h, w = self.pull_item(index)

        return im, gt

    def __len__(self):
        return len(self.ids)

    def pull_item(self, index):
        img_id = self.ids[index]

        img = cv2.imread(self._imgpath % img_id)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        height, width, _ = img.shape

        target = self.targets[img_id[1]]
        target = target.astype(float)
        target[:, :4] /= self.dims

        if self.transform is not None:
            img, boxes, labels = self.transform(img, target[:, :4], target[:, 4])

            target = np.hstack((boxes, np.expand_dims(labels, axis=1)))
        return img, target, height, width
