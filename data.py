import os.path
import logging
import cv2
import h5py
import numpy as np
import torch
import torch.utils.data as data
import random
import matplotlib.pyplot as plt

# create dataloader
def create_dataloader(dataset, dataset_opt, opt=None, sampler=None):
    phase = dataset_opt['phase']
    if phase == 'train':
        if opt['dist']:
            world_size = torch.distributed.get_world_size()
            num_workers = dataset_opt['n_workers']
            assert dataset_opt['batch_size'] % world_size == 0
            batch_size = dataset_opt['batch_size'] // world_size
            shuffle = False
        else:
            num_workers = dataset_opt['n_workers'] * len(opt['gpu_ids'])
            batch_size = dataset_opt['batch_size']
            shuffle = True
        return torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle,
                                           num_workers=num_workers, sampler=sampler, drop_last=True,
                                           pin_memory=False)
    else:
        return torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0,
                                           pin_memory=False)


# create dataset
def create_dataset(dataset_opt):
    mode = dataset_opt['mode']
    # datasets for image restoration
    if mode == 'LQGT':
        dataset = LQGTDataset(dataset_opt)
    else:
        raise NotImplementedError('Dataset [{:s}] is not recognized.'.format(mode))

    logger = logging.getLogger('base')
    logger.info('Dataset [{:s} - {:s}] is created.'.format(dataset.__class__.__name__,
                                                           dataset_opt['name']))
    return dataset


##################################################################################
# utils
##################################################################################

IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
    '.mat'
]


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)


def _get_paths_from_images(path):
    """get image path list from image folder"""
    assert os.path.isdir(path), '{:s} is not a valid directory'.format(path)
    images = []
    for dirpath, _, fnames in sorted(os.walk(path)):
        for fname in sorted(fnames):
            if is_image_file(fname):
                img_path = os.path.join(dirpath, fname)
                images.append(img_path)
    assert images, '{:s} has no valid image file'.format(path)
    return images


def get_image_paths(data_type, dataroot):
    """get image path list
    support image files"""
    paths, sizes = None, None
    if dataroot is not None:
        if data_type == 'img':
            paths = sorted(_get_paths_from_images(dataroot))
        else:
            raise NotImplementedError('data_type [{:s}] is not recognized.'.format(data_type))
    return paths, sizes


def read_mat(path, phase, key):
    """read image from mat
    return: Numpy float32, HWC, [0,1]"""
    with h5py.File(path, 'r') as hf:
        img = np.array(hf.get(key))
    # if phase != 'train':
    #     img = np.transpose(img, (2, 1, 0))
    img = np.transpose(img, (2, 1, 0))
    img = img.astype(np.float32) / 255.
    return img


def modcrop(img_in, scale):
    """img_in: Numpy, HWC or HW"""
    img = np.copy(img_in)
    if img.ndim == 2:
        H, W = img.shape
        H_r, W_r = H % scale, W % scale
        img = img[:H - H_r, :W - W_r]
    elif img.ndim == 3:
        H, W, C = img.shape
        H_r, W_r = H % scale, W % scale
        img = img[:H - H_r, :W - W_r, :]
    else:
        raise ValueError('Wrong img ndim: [{:d}].'.format(img.ndim))
    return img


def augment(img_list, hflip=True, rot=True):
    """horizontal flip OR rotate (0, 90, 180, 270 degrees)"""
    hflip = hflip and random.random() < 0.5
    vflip = rot and random.random() < 0.5
    rot90 = rot and random.random() < 0.5

    def _augment(img):
        if hflip:
            img = img[:, ::-1, :]
        if vflip:
            img = img[::-1, :, :]
        if rot90:
            img = img.transpose(1, 0, 2)
        return img

    return [_augment(img) for img in img_list]


##################################################################################
# Dataset
##################################################################################

class LQGTDataset(data.Dataset):
    """
    Read LQ (Low Quality, e.g. LR (Low Resolution), blurry, etc) and GT image pairs.
    """
    def __init__(self, opt):
        super(LQGTDataset, self).__init__()
        self.opt = opt
        self.data_type = self.opt['data_type']
        self.paths_LQ, self.paths_GT = None, None
        self.sizes_LQ, self.sizes_GT = None, None

        self.paths_GT, self.sizes_GT = get_image_paths(self.data_type, opt['dataroot_GT'])
        self.paths_LQ, self.sizes_LQ = get_image_paths(self.data_type, opt['dataroot_LQ'])
        assert self.paths_GT, 'Error: GT path is empty.'
        if self.paths_LQ and self.paths_GT:
            assert len(self.paths_LQ) == len(
                self.paths_GT
            ), 'GT and LQ datasets have different number of images - {}, {}.'.format(
                len(self.paths_LQ), len(self.paths_GT))

    def __getitem__(self, index):
        GT_path, LQ_path = None, None
        scale = self.opt['scale']
        GT_size = self.opt['GT_size']

        # get GT image
        GT_path = self.paths_GT[index]
        img_GT = read_mat(GT_path, self.opt['phase'], 'rad')
        if self.opt['phase'] != 'train':  # modcrop in the validation / test phase
            img_GT = modcrop(img_GT, scale)

        # get LQ image
        LQ_path = self.paths_LQ[index]
        img_LQ = read_mat(LQ_path, self.opt['phase'], 'im_LR')

        if self.opt['phase'] == 'train':
            # The dataloader will randomly crop the images to GT_size x GT_size patches for training.
            H, W, C = img_LQ.shape
            LQ_size = GT_size // scale

            # randomly crop
            rnd_h = random.randint(0, max(0, H - LQ_size))
            rnd_w = random.randint(0, max(0, W - LQ_size))
            img_LQ = img_LQ[rnd_h:rnd_h + LQ_size, rnd_w:rnd_w + LQ_size, :]
            rnd_h_GT, rnd_w_GT = int(rnd_h * scale), int(rnd_w * scale)
            img_GT = img_GT[rnd_h_GT:rnd_h_GT + GT_size, rnd_w_GT:rnd_w_GT + GT_size, :]

            # augmentation - flip, rotate
            img_LQ, img_GT = augment([img_LQ, img_GT], self.opt['use_flip'],
                                          self.opt['use_rot'])

        # HWC to CHW, numpy to tensor
        img_GT = torch.from_numpy(np.ascontiguousarray(np.transpose(img_GT, (2, 0, 1)))).float()
        img_LQ = torch.from_numpy(np.ascontiguousarray(np.transpose(img_LQ, (2, 0, 1)))).float()

        if LQ_path is None:
            LQ_path = GT_path
        return {'LQ': img_LQ, 'GT': img_GT, 'LQ_path': LQ_path, 'GT_path': GT_path}

    def __len__(self):
        return len(self.paths_GT)


##################################################################################
# Test code
##################################################################################

def test_dataloader():
    dataset = 'ICVL151_sub'  # ICVL151_sub

    opt = {}
    opt['dist'] = False
    opt['gpu_ids'] = [0]

    opt['name'] = dataset
    opt['dataroot_GT'] = 'E:/MMSR/datasets/train_data/mat'
    opt['dataroot_LQ'] = 'E:/MMSR/datasets/train_data/mat_bicLRx4'
    opt['mode'] = 'LQGT'
    opt['phase'] = 'train'
    opt['use_shuffle'] = True
    opt['n_workers'] = 0
    opt['batch_size'] = 16
    opt['GT_size'] = 128
    opt['scale'] = 4
    opt['use_flip'] = False
    opt['use_rot'] = False
    opt['data_type'] = 'img'  # img

    train_set = create_dataset(opt)
    train_loader = create_dataloader(train_set, opt, opt, None)

    print('start...')
    for i, data in enumerate(train_loader):
        if i > 5:
            break

        print(i)
        LQ = data['LQ']
        GT = data['GT']

        print('GT_size:',GT.size())
        print('GT_path:', data['GT_path'])
        print('LQ_size:', LQ.size())
        print('LQ_path:', data['LQ_path'])
        # plt.imshow(GT[1, 4, :, :])
        # plt.show()


if __name__ == "__main__":
    test_dataloader()