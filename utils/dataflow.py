"""Data related."""
import importlib
import os
from PIL import Image
import torch
from torchvision import datasets, transforms
from utils.autoaug import RandAugment
from utils.transforms import Lighting
from utils.transforms import CenterCropPadding
from utils.transforms import RandomResizedCropPadding
from utils.lmdb_dataset import ImageFolderLMDB

import numpy as np
import io
from functools import partial
from mmcv.parallel import collate


class_dict = {}
f = open('./data/map_clsloc.txt', 'r').readlines()
for line in f:
    name, id, describe = line.split()
    class_dict[name] = int(id) - 1

from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.data import create_transform

def build_transform(is_train, FLAGS):
    
    input_size = FLAGS.image_size
    resize_im = input_size > 32
    if is_train:

        transform = create_transform(
            input_size=input_size,
            is_training=True,
            color_jitter=0.4,
            auto_augment='rand-m9-mstd0.5-inc1',
            interpolation='bicubic',
            re_prob=0.25,
            re_mode='pixel',
            re_count=1,
        )
        if not resize_im:
            transform.transforms[0] = transforms.RandomCrop(input_size, padding=4)
        return transform

    t = []
    if resize_im:
        size = int((256 / 224) * input_size)
        t.append(
            transforms.Resize(size, interpolation=3),  # to maintain same ratio w.r.t. 224 images
        )
        t.append(transforms.CenterCrop(input_size))

    t.append(transforms.ToTensor())
    t.append(transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD))
    return transforms.Compose(t)


class DataPrefetcher():
    """Prefetch data to GPU.
    Modified from https://github.com/NVIDIA/apex/blob/f5cd5ae937f168c763985f627bbf850648ea5f3f/examples/imagenet/main_amp.py#L256
    """

    def __init__(self, loader):
        self.loader = iter(loader)
        self.stream = torch.cuda.Stream()
        # With Amp, it isn't necessary to manually convert data to half.
        # if args.fp16:
        #     self.mean = self.mean.half()
        #     self.std = self.std.half()
        self.stop = False
        self.preload()

    def preload(self):
        try:
            self.next_input, self.next_target = next(self.loader)
        except StopIteration:
            self.stop = True
            self.next_input = None
            self.next_target = None
            return
        with torch.cuda.stream(self.stream):
            self.next_input = self.next_input.cuda(non_blocking=True)
            self.next_target = self.next_target.cuda(non_blocking=True)
            # With Amp, it isn't necessary to manually convert data to half.
            # if args.fp16:
            #     self.next_input = self.next_input.half()
            # else:
            self.next_input = self.next_input.float()

    def __next__(self):
        torch.cuda.current_stream().wait_stream(self.stream)
        if self.stop:
            raise StopIteration
        input = self.next_input
        target = self.next_target
        self.preload()
        return input, target

    def __iter__(self):
        return self

    def __len__(self):
        return len(self.loader)


class DataPrefetcherKeypoint():
    """Prefetch data to GPU.
    Modified from https://github.com/NVIDIA/apex/blob/f5cd5ae937f168c763985f627bbf850648ea5f3f/examples/imagenet/main_amp.py#L256
    """

    def __init__(self, loader):
        self.loader = iter(loader)
        self.stream = torch.cuda.Stream()
        # With Amp, it isn't necessary to manually convert data to half.
        # if args.fp16:
        #     self.mean = self.mean.half()
        #     self.std = self.std.half()
        self.stop = False
        self.preload()

    def preload(self):
        try:
            self.next_input, self.next_target, self.next_target_weight, self.next_meta = next(self.loader)
        except StopIteration:
            self.stop = True
            self.next_input = None
            self.next_target = None
            self.next_target_weight = None
            self.next_meta = None
            return
        with torch.cuda.stream(self.stream):
            self.next_input = self.next_input.cuda(non_blocking=True)
            self.next_target = self.next_target.cuda(non_blocking=True)
            self.next_target_weight = self.next_target_weight.cuda(non_blocking=True)
            # With Amp, it isn't necessary to manually convert data to half.
            # if args.fp16:
            #     self.next_input = self.next_input.half()
            # else:
            self.next_input = self.next_input.float()

    def __next__(self):
        torch.cuda.current_stream().wait_stream(self.stream)
        if self.stop:
            raise StopIteration
        input = self.next_input
        target = self.next_target
        target_weight = self.next_target_weight
        meta = self.next_meta
        self.preload()
        return input, target, target_weight, meta

    def __iter__(self):
        return self

    def __len__(self):
        return len(self.loader)


class FakeData(datasets.vision.VisionDataset):
    """Fake data used to benchmark data pipeline."""

    def __init__(self,
                 size=1000,
                 image_size=(3, 224, 224),
                 num_classes=10,
                 transform=None,
                 target_transform=None):
        super(FakeData, self).__init__(None)
        self.transform = transform
        self.target_transform = target_transform
        self.size = size
        self.num_classes = num_classes
        self.image_size = image_size
        self.transform = transform
        self.target_transform = target_transform

        self.img = torch.zeros(image_size)
        self.target = 0

    def __getitem__(self, index):
        if index >= len(self):
            raise IndexError("{} index out of range".format(
                self.__class__.__name__))
        return self.img, self.target

    def __len__(self):
        return self.size


def data_transforms(FLAGS):
    """Get transform of dataset."""
    if FLAGS.data_transforms in [
        'imagenet1k_basic', 'imagenet1k_inception', 'imagenet1k_mobile'
    ]:
        if FLAGS.data_transforms == 'imagenet1k_inception':
            mean = [0.5, 0.5, 0.5]
            std = [0.5, 0.5, 0.5]
            crop_scale = 0.08
            jitter_param = 0.4
            lighting_param = 0.1
        elif FLAGS.data_transforms == 'imagenet1k_basic':
            mean = [0.485, 0.456, 0.406]
            std = [0.229, 0.224, 0.225]
            crop_scale = 0.08
            jitter_param = 0.4
            lighting_param = 0.1
        elif FLAGS.data_transforms == 'imagenet1k_mobile':
            mean = [0.485, 0.456, 0.406]
            std = [0.229, 0.224, 0.225]
            crop_scale = 0.25
            jitter_param = 0.4
            lighting_param = 0.1
        if not FLAGS.prune_params.bn_prune_filter:
            randaug = [RandAugment(2, 12)]
        else:
            randaug = []
        train_transforms = transforms.Compose(randaug + [
            transforms.RandomResizedCrop(224, scale=(crop_scale, 1.0)),
            transforms.ColorJitter(brightness=jitter_param,
                                   contrast=jitter_param,
                                   saturation=jitter_param),
            Lighting(lighting_param),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ])
        val_transforms = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ])
        test_transforms = val_transforms
    elif FLAGS.data_transforms in [
        'imagenet1k_mnas_bilinear', 'imagenet1k_mnas_bicubic'
    ]:
        if FLAGS.data_transforms == 'imagenet1k_mnas_bilinear':
            resize_method = Image.BILINEAR
        elif FLAGS.data_transforms == 'imagenet1k_mnas_bicubic':
            resize_method = Image.BICUBIC
        mean = [0.485, 0.456, 0.406]  # RGB
        std = [0.229, 0.224, 0.225]
        crop_padding = 32

        train_transforms = transforms.Compose([
            RandomResizedCropPadding(224,
                                     scale=(0.08, 1.0),
                                     min_object_covered=0.1,
                                     ratio=(3. / 4., 4. / 3.),
                                     log_ratio=False,
                                     interpolation=resize_method,
                                     crop_padding=crop_padding),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ])
        val_transforms = transforms.Compose([
            CenterCropPadding(224, crop_padding),
            transforms.Resize(224, resize_method),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ])
        test_transforms = val_transforms
    else:
        try:
            transforms_lib = importlib.import_module(FLAGS.data_transforms)
            return transforms_lib.data_transforms()
        except ImportError:
            raise NotImplementedError(
                'Data transform {} is not yet implemented.'.format(
                    FLAGS.data_transforms))
    return train_transforms, val_transforms, test_transforms


def dataset(train_transforms, val_transforms, test_transforms, FLAGS):
    """Get dataset for classification."""
    if FLAGS.dataset == 'imagenet1k':
        if not FLAGS.test_only or FLAGS.bn_calibration:
            train_set = datasets.ImageFolder(os.path.join(
                FLAGS.dataset_dir, 'train'),
                transform=train_transforms)
        else:
            train_set = None
        val_set = datasets.ImageFolder(os.path.join(FLAGS.dataset_dir, 'val'),
                                       transform=val_transforms)
        test_set = None
    
    elif FLAGS.dataset == 'imagenet1k_fake':
        train_set = FakeData(size=1281167,
                             image_size=(3, FLAGS.image_size, FLAGS.image_size),
                             num_classes=1000)
        val_set = FakeData(size=50000,
                           image_size=(3, FLAGS.image_size, FLAGS.image_size),
                           num_classes=1000)
        test_set = None
    
    elif FLAGS.dataset == 'imagenet1k_lmdb':
        if not FLAGS.test_only or FLAGS.bn_calibration:
            if not FLAGS.use_hdfs:
                train_set = ImageFolderLMDB(os.path.join(FLAGS.dataset_dir,
                                                         'train'),
                                            transform=train_transforms)

        else:
            train_set = None
        if not FLAGS.use_hdfs:
            val_set = ImageFolderLMDB(os.path.join(FLAGS.dataset_dir, 'val'),
                                      transform=val_transforms)

        test_set = None
    
    elif FLAGS.dataset == 'cifar10':
        if not FLAGS.test_only or FLAGS.bn_calibration:
            transform = build_transform(is_train=True, FLAGS=FLAGS)
            train_set = datasets.CIFAR10(FLAGS.dataset_dir, train=True, transform=transform, download=True)
        else:
            train_set = None
        val_set = datasets.CIFAR10(FLAGS.dataset_dir, train=False, transform=transform, download=True)
        test_set = None 
    
    else:
        try:
            dataset_lib = importlib.import_module(FLAGS.dataset)
            return dataset_lib.dataset(train_transforms, val_transforms,
                                       test_transforms)
        except ImportError:
            raise NotImplementedError(
                'Dataset {} is not yet implemented.'.format(FLAGS.dataset))
    return train_set, val_set, test_set



def collate_fn(list_data):
    batched_pts_list, batched_gt_bboxes_list = [], []
    batched_labels_list, batched_names_list = [], []
    batched_difficulty_list = []
    batched_img_list, batched_calib_list = [], []
    for data_dict in list_data:
        pts, gt_bboxes_3d = data_dict['pts'], data_dict['gt_bboxes_3d']
        gt_labels, gt_names = data_dict['gt_labels'], data_dict['gt_names']
        difficulty = data_dict['difficulty']
        image_info, calbi_info = data_dict['image_info'], data_dict['calib_info']

        batched_pts_list.append(torch.from_numpy(pts))
        batched_gt_bboxes_list.append(torch.from_numpy(gt_bboxes_3d))
        batched_labels_list.append(torch.from_numpy(gt_labels))
        batched_names_list.append(gt_names) # List(str)
        batched_difficulty_list.append(torch.from_numpy(difficulty))
        batched_img_list.append(image_info)
        batched_calib_list.append(calbi_info)
    
    rt_data_dict = dict(
        batched_pts=batched_pts_list,
        batched_gt_bboxes=batched_gt_bboxes_list,
        batched_labels=batched_labels_list,
        batched_names=batched_names_list,
        batched_difficulty=batched_difficulty_list,
        batched_img_info=batched_img_list,
        batched_calib_info=batched_calib_list
    )

    return rt_data_dict


def data_loader(train_set, val_set, test_set, FLAGS):
    """Get data loader."""

    def _build_loader(dset, batch_size, shuffle, sampler=None):
        if FLAGS.dataset!= 'kitti':
            return torch.utils.data.DataLoader(
                dset,
                batch_size=batch_size,
                shuffle=shuffle,
                pin_memory=True,
                num_workers=FLAGS.data_loader_workers,
                sampler=sampler,
                drop_last=FLAGS.get('drop_last', False))
        
        elif FLAGS.dataset == 'kitti':
            collate = collate_fn
            return torch.utils.data.DataLoader(
                dset,
                batch_size=batch_size,
                shuffle=shuffle,
                # pin_memory=True,
                num_workers=FLAGS.data_loader_workers,
                sampler=sampler,
                collate_fn=collate,
                drop_last=FLAGS.get('drop_last', False))


    if FLAGS.use_distributed:
        if not FLAGS.test_only or FLAGS.bn_calibration:
            train_sampler = torch.utils.data.distributed.DistributedSampler(train_set)
            train_shuffle = False
        val_sampler = None
    else:
        if not FLAGS.test_only or FLAGS.bn_calibration:
            train_sampler = None
            train_shuffle = True
        val_sampler = None

    if FLAGS.data_loader == 'imagenet1k_basic':
        if not FLAGS.test_only:
            if not FLAGS.use_hdfs:
                train_loader = _build_loader(train_set,
                                             FLAGS._loader_batch_size,
                                             train_shuffle,
                                             sampler=train_sampler)
        else:
            train_loader = None
        if FLAGS.bn_calibration:
            if not FLAGS.use_hdfs:
                calib_loader = _build_loader(train_set,
                                             1,
                                            #  FLAGS._loader_batch_size_calib,
                                             train_shuffle,
                                             sampler=train_sampler)
        else:
            calib_loader = None
        
        # build the validation dataloader
        if not FLAGS.use_hdfs:
            if FLAGS.model_kwparams.task == 'segmentation' and FLAGS.dataset != 'coco':
                val_loader = torch.utils.data.DataLoader(val_set,
                                                         batch_size=1,
                                                         shuffle=False,
                                                         pin_memory=True,
                                                         num_workers=FLAGS.data_loader_workers,
                                                         collate_fn=partial(collate, samples_per_gpu=1),
                                                         worker_init_fn=None,
                                                         sampler=None if FLAGS.single_gpu_test else val_sampler,
                                                         drop_last=FLAGS.get('drop_last', False))
            elif FLAGS.dataset == 'coco':
                val_loader = torch.utils.data.DataLoader(val_set,
                                                         batch_size=FLAGS._loader_batch_size * 8,
                                                         shuffle=False,
                                                         pin_memory=True,
                                                         num_workers=FLAGS.data_loader_workers,
                                                         sampler=None,
                                                         drop_last=FLAGS.get('drop_last', False))
            else: # kitti
                val_loader = _build_loader(val_set,
                                           1, # FLAGS._loader_batch_size,
                                           False,
                                           sampler=val_sampler)
        test_loader = val_loader
    else:
        try:
            data_loader_lib = importlib.import_module(FLAGS.data_loader)
            return data_loader_lib.data_loader(train_set, val_set, test_set)
        except ImportError:
            raise NotImplementedError(
                'Data loader {} is not yet implemented.'.format(
                    FLAGS.data_loader))
    return train_loader, calib_loader, val_loader, test_loader
