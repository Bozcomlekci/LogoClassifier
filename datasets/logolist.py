import os

import numpy as np

from .base_dataset import BaseDataset
from .builder import DATASETS

#Create a new dataset upon BaseDataset described in:
#https://mmclassification.readthedocs.io/en/latest/#tutorials/new_dataset.html

def has_file_allowed_extension(filename, extensions):
    filename_lower = filename.lower()
    return any(filename_lower.endswith(ext) for ext in extensions)


def find_folders(root):
    folders = [
        d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d))
    ]
    folders.sort()
    folder_to_idx = {folders[i]: i for i in range(len(folders))}
    return folder_to_idx


def get_samples(root, folder_to_idx, extensions):
    samples = []
    root = os.path.expanduser(root)
    for folder_name in sorted(os.listdir(root)):
        _dir = os.path.join(root, folder_name)
        if not os.path.isdir(_dir):
            continue

        for _, _, fns in sorted(os.walk(_dir)):
            for fn in sorted(fns):
                if has_file_allowed_extension(fn, extensions):
                    path = os.path.join(folder_name, fn)
                    item = (path, folder_to_idx[folder_name])
                    samples.append(item)
    return samples


@DATASETS.register_module()
class Logolist(BaseDataset):
    IMG_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif')
    CLASSES = [
        'microsoft',
        'ikea',
        'oracle',
        'stripe',
        'netflix',
        'visa',
        'hsbc',
        'spotify',
        'square',
        'fedex',
        'ups',
        'dhl',
        'jpmorgan',
        'airbnb',
        'facebook',
        'usps',
        'samsung',
        'icbc',
        'apple',
        'google',
        'chase bank',
        'pwc',
        'at&t',
        'ibm',
        'mastercard',
        'american express',
        'alibaba',
        'bank of america',
        'adobe',
        'amazon',
        'ebay',
        'other_class'
    ]

    def load_annotations(self):
        if self.ann_file is None:
            folder_to_idx = find_folders(self.data_prefix)
            samples = get_samples(
                self.data_prefix,
                folder_to_idx,
                extensions=self.IMG_EXTENSIONS)
            if len(samples) == 0:
                raise (RuntimeError('Found 0 files in subfolders of: '
                                    f'{self.data_prefix}. '
                                    'Supported extensions are: '
                                    f'{",".join(self.IMG_EXTENSIONS)}'))

            self.folder_to_idx = folder_to_idx
        elif isinstance(self.ann_file, str):
            with open(self.ann_file) as f:
                samples = [x.strip().split(' ') for x in f.readlines()]
        else:
            raise TypeError('ann_file must be a str or None')
        self.samples = samples
        data_infos = []
        for sample in self.samples:
            filename = sample[0]
            gt_label = sample[1]
            info = {'img_prefix': self.data_prefix}
            info['img_info'] = {'filename': filename}
            info['gt_label'] = np.array(gt_label, dtype=np.int64)
            data_infos.append(info)
        return data_infos
