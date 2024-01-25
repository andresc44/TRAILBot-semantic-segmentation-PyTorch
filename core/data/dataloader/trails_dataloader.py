"""Hiking Trails Semantic Segmentation Dataset."""
import os
import torch
import numpy as np

from PIL import Image
from .segbase import SegmentationDataset

class TrailsSegmentation(SegmentationDataset):
    """Trails Semantic Segmentation Dataset.

    Parameters
    ----------
    root : string
        Path to Trails folder. Default is './datasets/trails'
    split: string
        'train', 'val' or 'test'
    mode: string
        'train', 'val', 'test', or 'testval'
        indicates how to transform data
    transform : callable, optional
        A function that transforms the image
    Examples
    --------
    >>> from torchvision import transforms
    >>> import torch.utils.data as data
    >>> # Transforms for Normalization
    >>> input_transform = transforms.Compose([
    >>>     transforms.ToTensor(),
    >>>     transforms.Normalize((.485, .456, .406), (.229, .224, .225)),
    >>> ])
    >>> # Create Dataset
    >>> trainset = TrailsSegmentation(split='train', transform=input_transform)
    >>> # Create Training Loader
    >>> train_data = data.DataLoader(
    >>>     trainset, 4, shuffle=True,
    >>>     num_workers=4)
    """
    NUM_CLASS = 2

    def __init__(self, root='../core/data/datasets/trail_dataset', split='train', mode=None, transform=None, **kwargs):
        super(TrailsSegmentation, self).__init__(root, split, mode, transform, **kwargs)
        assert os.path.exists(self.root)
        self.images, self.masks = _get_trails_pairs(self.root, self.split)
        assert (len(self.images) == len(self.masks))
        if len(self.images) == 0:
            raise RuntimeError("Found 0 images in subfolders of:" + root + "\n")

    def __getitem__(self, index):
        img = Image.open(self.images[index]).convert('RGB')
        if self.mode == 'test':
            if self.transform is not None:
                img = self.transform(img)
            return img, os.path.basename(self.images[index])
        mask = Image.open(self.masks[index])
        # synchrosized transform
        if self.mode == 'train':
            img, mask = self._sync_transform(img, mask)
        elif self.mode == 'val':
            img, mask = self._val_sync_transform(img, mask)
        else:
            assert self.mode == 'testval'
            img, mask = self._img_transform(img), self._mask_transform(mask)
        # general resize, normalize and toTensor
        if self.transform is not None:
            img = self.transform(img)
        return img, mask, os.path.basename(self.images[index])

    def _mask_transform(self, mask):
        target = np.array(mask).astype('int32')
        target[target > 0] = 1
        return torch.from_numpy(target).long()

    def __len__(self):
        return len(self.images)

    @property
    def pred_offset(self):
        return 0


def _get_trails_pairs(folder, split='train'):
    def get_path_pairs(img_folder, mask_folder):
        img_paths = []
        mask_paths = []
        for root, _, files in os.walk(img_folder):
            for filename in files:
                if filename.endswith('.jpg'):
                    imgpath = os.path.join(root, filename)
                    maskname = filename.replace('.jpg', '.png')
                    maskpath = os.path.join(mask_folder, maskname)
                    if os.path.isfile(imgpath) and os.path.isfile(maskpath):
                        img_paths.append(imgpath)
                        mask_paths.append(maskpath)
                    else:
                        print('cannot find the mask or image:', imgpath, maskpath)
        print('Found {} images in the folder {}'.format(len(img_paths), img_folder))
        return img_paths, mask_paths

    if split == 'train':
        img_folder = os.path.join(folder, 'Training/Images')
        mask_folder = os.path.join(folder, 'Training/Masks')
        img_paths, mask_paths = get_path_pairs(img_folder, mask_folder)
    elif split == 'test':
        img_folder = os.path.join(folder, 'Testing/Images')
        mask_folder = os.path.join(folder, 'Testing/Masks')
        img_paths, mask_paths = get_path_pairs(img_folder, mask_folder)
    elif split == 'val':
        img_folder = os.path.join(folder, 'Validating/Images')
        mask_folder = os.path.join(folder, 'Validating/Masks')
        img_paths, mask_paths = get_path_pairs(img_folder, mask_folder)
    return img_paths, mask_paths


if __name__ == '__main__':
    dataset = TrailsSegmentation(base_size=280, crop_size=256)