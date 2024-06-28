import cv2
import PIL
import torch
import os, sys
import numpy as np
from glob import glob
from PIL import Image
import torch.utils.data
import torch.nn.functional as nnF
import torchvision.transforms as tfs
from torchvision import transforms, utils


def get_data_loaders(cfgs):
    batch_size = cfgs.get('batch_size', 64)
    num_workers = cfgs.get('num_workers', 4)

    run_train = cfgs.get('run_train', False)
    train_val_data_dir = cfgs.get('train_val_data_dir', './data')
    run_test = cfgs.get('run_test', False)
    test_data_dir = cfgs.get('test_data_dir', './data/test')

    train_loader = val_loader = test_loader = None
        
    get_train_loader = lambda **kargs: get_paired_image_loader(batch_size=batch_size, num_workers=num_workers, **kargs)
    
    get_loader = lambda **kargs: get_image_loader(**kargs, batch_size=batch_size, num_workers=num_workers)

    if run_train:
        train_data_dir = os.path.join(train_val_data_dir, "train")
        high_data_dir = os.path.join(train_data_dir, 'high')
        low_data_dir = os.path.join(train_data_dir, 'low')
        val_data_dir = os.path.join(train_val_data_dir, "val")
        high_val_dir = os.path.join(val_data_dir, 'high')
        low_val_dir = os.path.join(val_data_dir, 'low')
        assert os.path.isdir(train_data_dir), "Training data directory does not exist: %s" %train_data_dir
        assert os.path.isdir(val_data_dir), "Validation data directory does not exist: %s" %val_data_dir
        print(f"Loading training data from {train_data_dir}")
        train_loader = get_train_loader(data_dir_high=high_data_dir, data_dir_low=low_data_dir)
        print(f"Loading validation data from {val_data_dir}")
        val_loader = get_train_loader(data_dir_high=high_val_dir, data_dir_low=low_val_dir, is_validation=True)
    if run_test:
        assert os.path.isdir(test_data_dir), "Testing data directory does not exist: %s" %test_data_dir
        print(f"Loading testing data from {test_data_dir}")
        test_loader = get_loader(data_dir=test_data_dir)

    return train_loader, val_loader, test_loader


IMG_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif', '.tiff', 'webp')
def is_image_file(filename):
    return filename.lower().endswith(IMG_EXTENSIONS)

class paired_faces_data(torch.utils.data.Dataset):
    def __init__(self, data_hr, data_lr, is_validation):
        self.hr_imgs = [os.path.join(data_hr, i) for i in os.listdir(data_hr) if os.path.isfile(os.path.join(data_hr, i))]
        self.lr_imgs = [os.path.join(data_lr, i) for i in os.listdir(data_lr) if os.path.isfile(os.path.join(data_lr, i))]
        self.lr_len = len(self.lr_imgs)
        self.lr_shuf = np.arange(self.lr_len)
        self.is_validation = is_validation
        np.random.shuffle(self.lr_shuf)
        self.lr_idx = 0
        self.preproc = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

    def __len__(self):
        return len(self.hr_imgs)

    def __getitem__(self, index):
        data = {}
        hr = Image.open(self.hr_imgs[index]).convert('RGB')
        lr = Image.open(self.lr_imgs[self.lr_shuf[self.lr_idx]]).convert('RGB')
        fname = os.path.basename(self.hr_imgs[index])
        self.lr_idx += 1
        if self.lr_idx >= self.lr_len:
            self.lr_idx = 0
            np.random.shuffle(self.lr_shuf)
            
        hflip = not self.is_validation and np.random.rand()>0.5
        if hflip:
            hr = tfs.functional.hflip(hr)
            lr = tfs.functional.hflip(lr)
        
        data["z"] = torch.randn(1, 64, dtype=torch.float32)
        data["lr"] = self.preproc(lr)
        data["hr"] = self.preproc(hr)
        data["hr_down"] = nnF.avg_pool2d(data["hr"], 4, 4)
        data['img_name'] = fname
        return data
    
    def get_noise(self, n):
        return torch.randn(n, 1, 64, dtype=torch.float32)

def get_paired_image_loader(data_dir_high, data_dir_low, batch_size=16, num_workers=4, is_validation=False):
    dataset = paired_faces_data(data_dir_high, data_dir_low, is_validation=is_validation)
    loader = torch.utils.data.DataLoader(
        dataset=dataset, 
        batch_size=batch_size, 
        shuffle=not is_validation, 
        num_workers=num_workers, 
        pin_memory=True
    )
    return loader

class faces_data(torch.utils.data.Dataset):
    def __init__(self, data_dir, transform):
        assert data_dir, print('no datasets specified')
        self.transform = transform
        self.img_list = []
        self.data_dir = data_dir
        list_name = (glob(os.path.join(self.data_dir, "*.png")))
        list_name.sort()
        for filename in list_name:
            self.img_list.append(filename)
    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, index):
        data = {}
        inp16 = Image.open(self.img_list[index]).convert('RGB')
        fname = os.path.basename(self.img_list[index])
        data['lr'] = self.transform(inp16)
        data['img_name'] = fname
        data["z"] = torch.randn(1, 64, dtype=torch.float32)
        return data

def get_image_loader(data_dir, batch_size=1, num_workers=4):

    transform = transforms.Compose([
            transforms.ToTensor(),transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    dataset = faces_data(data_dir, transform)
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    return loader


## simple image dataset ##
def make_dataset(dir):
    assert os.path.isdir(dir), '%s is not a valid directory' % dir

    images = []
    for root, _, fnames in sorted(os.walk(dir)):
        for fname in sorted(fnames):
            if is_image_file(fname):
                fpath = os.path.join(root, fname)
                images.append(fpath)
    return images


class ImageDataset(torch.utils.data.Dataset):
    def __init__(self, data_dir, image_size=256, crop=None, is_validation=False):
        super(ImageDataset, self).__init__()
        self.root = data_dir
        self.paths = make_dataset(data_dir)
        self.size = len(self.paths)
        self.image_size = image_size
        self.crop = crop
        self.is_validation = is_validation

    def transform(self, img, hflip=False):
        if self.crop is not None:
            if isinstance(self.crop, int):
                img = tfs.CenterCrop(self.crop)(img)
            else:
                assert len(self.crop) == 4, 'Crop size must be an integer for center crop, or a list of 4 integers (y0,x0,h,w)'
                img = tfs.functional.crop(img, *self.crop)
        img = tfs.functional.resize(img, (self.image_size, self.image_size))
        if hflip:
            img = tfs.functional.hflip(img)
        return tfs.functional.to_tensor(img)

    def __getitem__(self, index):
        fpath = self.paths[index % self.size]
        img = Image.open(fpath).convert('RGB')
        hflip = not self.is_validation and np.random.rand()>0.5
        return self.transform(img, hflip=hflip)

    def __len__(self):
        return self.size

    def name(self):
        return 'ImageDataset'



## paired AB image dataset ##
def make_paied_dataset(dir, AB_dnames=None, AB_fnames=None):
    A_dname, B_dname = AB_dnames or ('A', 'B')
    dir_A = os.path.join(dir, A_dname)
    dir_B = os.path.join(dir, B_dname)
    assert os.path.isdir(dir_A), '%s is not a valid directory' % dir_A
    assert os.path.isdir(dir_B), '%s is not a valid directory' % dir_B

    images = []
    for root_A, _, fnames_A in sorted(os.walk(dir_A)):
        for fname_A in sorted(fnames_A):
            if is_image_file(fname_A):
                path_A = os.path.join(root_A, fname_A)
                root_B = root_A.replace(dir_A, dir_B, 1)
                if AB_fnames is not None:
                    fname_B = fname_A.replace(*AB_fnames)
                else:
                    fname_B = fname_A
                path_B = os.path.join(root_B, fname_B)
                if os.path.isfile(path_B):
                    images.append((path_A, path_B))
    return images


class PairedDataset(torch.utils.data.Dataset):
    def __init__(self, data_dir, image_size=256, crop=None, is_validation=False, AB_dnames=None, AB_fnames=None):
        super(PairedDataset, self).__init__()
        self.root = data_dir
        self.paths = make_paied_dataset(data_dir, AB_dnames=AB_dnames, AB_fnames=AB_fnames)
        self.size = len(self.paths)
        self.image_size = image_size
        self.crop = crop
        self.is_validation = is_validation

    def transform(self, img, hflip=False):
        if self.crop is not None:
            if isinstance(self.crop, int):
                img = tfs.CenterCrop(self.crop)(img)
            else:
                assert len(self.crop) == 4, 'Crop size must be an integer for center crop, or a list of 4 integers (y0,x0,h,w)'
                img = tfs.functional.crop(img, *self.crop)
        img = tfs.functional.resize(img, (self.image_size, self.image_size))
        if hflip:
            img = tfs.functional.hflip(img)
        return tfs.functional.to_tensor(img)

    def __getitem__(self, index):
        path_A, path_B = self.paths[index % self.size]
        img_A = Image.open(path_A).convert('RGB')
        img_B = Image.open(path_B).convert('RGB')
        hflip = not self.is_validation and np.random.rand()>0.5
        return self.transform(img_A, hflip=hflip), self.transform(img_B, hflip=hflip)

    def __len__(self):
        return self.size

    def name(self):
        return 'PairedDataset'







