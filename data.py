import os.path
import torch
import cv2
import numpy as np
import torch.utils.data as data
from skimage.transform import resize
import random

# Image utils
IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
]


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)


def make_dataset(s_dir):
    images = []
    if os.path.isdir(s_dir):
        for root, _, fnames in sorted(os.walk(s_dir)):
            for fname in fnames:
                if is_image_file(fname):
                    path = os.path.join(root, fname)
                    images.append(path)
    elif s_dir.endswith('.txt'):
        with open(s_dir, 'r') as fid:
            lines = fid.readlines()
            images = [x.strip() for x in lines]
    else:
        raise ValueError('input need to be a dir or a *.txt file, however it is {}'.format(s_dir))

    return images


class LowLevelImageFolder(data.Dataset):
    """
    A loader for loading most of low level tasks
    """

    def __init__(self, params, img_loader=cv2.imread, is_train=True):
        self.is_train = is_train
        self.img_loader = img_loader
        self.root = params['data_root']
        self.image_names = make_dataset(os.path.join(self.root, 'train.txt' if is_train else 'eval.txt'))
        self.height = params['crop_image_height']
        self.width = params['crop_image_width']
        self.new_size = params['new_size']

    def __len__(self):
        return len(self.image_names)

    def data_argument(self, img, mode, random_pos, random_flip):

        h, w = img.shape[:2]
        if h > w:
            new_w = int(w / h * self.new_size)
            new_h = self.new_size
        else:
            new_h = int(h / w * self.new_size)
            new_w = self.new_size

        img = cv2.resize(img, (new_w, new_h))

        if random_flip > 0.5:
            img = np.fliplr(img)

        # img = rotate(img,random_angle, order = mode)
        img = img[random_pos[0]:random_pos[1], random_pos[2]:random_pos[3], :]
        img = resize(img, (self.height, self.width), order=mode)

        return img

    def load_images(self, index, use_da=True):
        image_name = self.image_names[index]
        img_in = np.float32(self.img_loader(os.path.join(self.root, 'input', image_name))) / 255.
        img_out = np.float32(self.img_loader(os.path.join(self.root, 'output', image_name))) / 255.

        ori_h, ori_w = img_in.shape[:2]

        if use_da:
            random_flip = random.random()
            random_start_y = random.randint(0, 9)
            random_start_x = random.randint(0, 9)

            random_pos = [random_start_y, random_start_y + ori_h - 10, random_start_x,
                          random_start_x + ori_w - 10]

            img_in = self.data_argument(img_in, 1, random_pos, random_flip)
            img_out = self.data_argument(img_out, 1, random_pos, random_flip)
        else:
            h1, w1 = img_in.shape[:2]
            h2, w2 = img_out.shape[:2]
            h, w = max(h1, h2), max(w1, w2)
            if h > w:
                new_w = int(w / h * self.new_size)
                new_h = self.new_size
            else:
                new_h = int(h / w * self.new_size)
                new_w = self.new_size

            img_in = cv2.resize(img_in, (new_w, new_h))
            img_out = cv2.resize(img_out, (new_w, new_h))

        img_in = img_in[:, :, ::-1].copy()
        img_out = img_out[:, :, ::-1].copy()

        return img_in, img_out, image_name

    def __getitem__(self, index):
        img_in, img_out, file_name = self.load_images(index, self.is_train)

        targets = {}
        img_in = torch.from_numpy(np.transpose(img_in, (2, 0, 1))).contiguous().float()
        targets['output'] = torch.from_numpy(np.transpose(img_out, (2, 0, 1))).contiguous().float()
        targets['name'] = file_name

        return img_in, targets
