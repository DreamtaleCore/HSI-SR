from torch.utils.data import DataLoader
from torch.autograd import Variable
from torch.optim import lr_scheduler
from torchvision import transforms
from data import create_dataset,create_dataloader
import torch
import os
import math
import torchvision.utils as vutils
import torch.utils.model_zoo as model_zoo
import yaml
import numpy as np
import torch.nn.init as init
import time
import easydict
import h5py
from torch.nn import functional as F
import torch.nn as nn


# Methods
# get_all_data_loaders      : primary data loader interface (load trainA, testA, trainB, testB)
# get_data_loader_list      : list-based data loader
# get_data_loader_folder    : folder-based data loader
# get_config                : load yaml file
# eformat                   :
# write_2images             : save output image
# prepare_sub_folder        : create checkpoints and images folders for saving outputs
# write_one_row_html        : write one row of the html file for output images
# write_html                : create the html file.
# write_loss
# slerp
# get_slerp_interp
# get_model_list
# load_vgg16
# vgg_preprocess
# get_scheduler
# weights_init


def to_number(data):
    if type(data) is torch.Tensor:
        return data.item()
    else:
        return data


def get_local_time():
    return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())


def get_reflection_data_loader(conf):
    for phase, dataset_opt in conf['datasets'].items():
        dataset_opt['data_type'] = 'img'
        dataset_opt['phase'] = phase
        if conf['distortion'] == 'sr':
            dataset_opt['scale'] = conf['scale']
        if phase == 'train':
            train_set = create_dataset(dataset_opt)
            train_loader = create_dataloader(train_set, dataset_opt, conf, None)
        elif phase == 'val':
            dataset_opt['GT_size'] = None
            val_set = create_dataset(dataset_opt)
            val_loader = create_dataloader(val_set, dataset_opt, conf, None)
        else:
            raise NotImplementedError('Phase [{:s}] is not recognized.'.format(phase))
    return train_loader, val_loader


def get_config(config):
    with open(config, 'r') as stream:
        return easydict.EasyDict(yaml.load(stream))


def eformat(f, prec):
    s = "%.*e" % (prec, f)
    mantissa, exp = s.split('e')
    # add 1 to digits as 1 is taken by sign +/-
    return "%se%d" % (mantissa, int(exp))


def __write_images(image_outputs, display_image_num, file_name):
    image_outputs = [images.expand(-1, 3, -1, -1) for images in image_outputs]  # expand gray-scale images to 3 channels
    image_tensor = torch.cat([images[:display_image_num] for images in image_outputs], 0)
    image_grid = vutils.make_grid(image_tensor.data, nrow=display_image_num, padding=0, normalize=False)
    vutils.save_image(image_grid, file_name, nrow=1)


def write_2images(image_outputs, display_image_num, image_directory, postfix):
    n = len(image_outputs)
    assert n == 3, 'length of sampled images should be 3'
    __write_images(image_outputs, display_image_num,
                   '%s/x-y_gt-y_pred_%s.jpg' % (image_directory, postfix))


def prepare_sub_folder(output_directory):
    image_directory = os.path.join(output_directory, 'images')
    if not os.path.exists(image_directory):
        print("Creating directory: {}".format(image_directory))
        os.makedirs(image_directory)
    checkpoint_directory = os.path.join(output_directory, 'checkpoints')
    if not os.path.exists(checkpoint_directory):
        print("Creating directory: {}".format(checkpoint_directory))
        os.makedirs(checkpoint_directory)
    return checkpoint_directory, image_directory


def write_one_row_html(html_file, iterations, img_filename, all_size):
    html_file.write("<h3>iteration [%d] (%s)</h3>" % (iterations, img_filename.split('/')[-1]))
    html_file.write("""
        <p><a href="%s">
          <img src="%s" style="width:%dpx">
        </a><br>
        <p>
        """ % (img_filename, img_filename, all_size))
    return


def write_html(filename, iterations, image_save_iterations, image_directory, all_size=1536):
    html_file = open(filename, "w")
    html_file.write('''
    <!DOCTYPE html>
    <html>
    <head>
      <title>Experiment name = %s</title>
      <meta http-equiv="refresh" content="30">
    </head>
    <body>
    ''' % os.path.basename(filename))
    html_file.write("<h3>current</h3>")
    write_one_row_html(html_file, iterations, '%s/gen_a2b_train_current.jpg' % (image_directory), all_size)
    write_one_row_html(html_file, iterations, '%s/gen_b2a_train_current.jpg' % (image_directory), all_size)
    for j in range(iterations, image_save_iterations - 1, -1):
        if j % image_save_iterations == 0:
            write_one_row_html(html_file, j, '%s/gen_a2b_test_%08d.jpg' % (image_directory, j), all_size)
            write_one_row_html(html_file, j, '%s/gen_b2a_test_%08d.jpg' % (image_directory, j), all_size)
            write_one_row_html(html_file, j, '%s/gen_a2b_train_%08d.jpg' % (image_directory, j), all_size)
            write_one_row_html(html_file, j, '%s/gen_b2a_train_%08d.jpg' % (image_directory, j), all_size)
    html_file.write("</body></html>")
    html_file.close()


def write_loss(iterations, trainer, train_writer):
    members = [attr for attr in dir(trainer) \
               if not callable(getattr(trainer, attr)) and not attr.startswith("__") and (
                       'loss' in attr or 'grad' in attr or 'nwd' in attr)]
    for m in members:
        train_writer.add_scalar(m, getattr(trainer, m), iterations + 1)


def slerp(val, low, high):
    """
    original: Animating Rotation with Quaternion Curves, Ken Shoemake
    https://arxiv.org/abs/1609.04468
    Code: https://github.com/soumith/dcgan.torch/issues/14, Tom White
    """
    omega = np.arccos(np.dot(low / np.linalg.norm(low), high / np.linalg.norm(high)))
    so = np.sin(omega)
    return np.sin((1.0 - val) * omega) / so * low + np.sin(val * omega) / so * high


def get_slerp_interp(nb_latents, nb_interp, z_dim):
    """
    modified from: PyTorch inference for "Progressive Growing of GANs" with CelebA snapshot
    https://github.com/ptrblck/prog_gans_pytorch_inference
    """

    latent_interps = np.empty(shape=(0, z_dim), dtype=np.float32)
    for _ in range(nb_latents):
        low = np.random.randn(z_dim)
        high = np.random.randn(z_dim)  # low + np.random.randn(512) * 0.7
        interp_vals = np.linspace(0, 1, num=nb_interp)
        latent_interp = np.array([slerp(v, low, high) for v in interp_vals],
                                 dtype=np.float32)
        latent_interps = np.vstack((latent_interps, latent_interp))

    return latent_interps[:, :, np.newaxis, np.newaxis]


# Get model list for resume
def get_model_list(dirname, key):
    if os.path.exists(dirname) is False:
        return None
    gen_models = [os.path.join(dirname, f) for f in os.listdir(dirname) if
                  os.path.isfile(os.path.join(dirname, f)) and key in f and ".pt" in f]
    if len(gen_models) == 0:
        return None
    gen_models.sort()
    last_model_name = gen_models[-1]
    return last_model_name


def get_scheduler(optimizer, hyperparameters, iterations=-1):
    if 'lr_policy' not in hyperparameters or hyperparameters['lr_policy'] == 'constant':
        scheduler = None  # constant scheduler
    elif hyperparameters['lr_policy'] == 'step':
        scheduler = lr_scheduler.StepLR(optimizer, step_size=hyperparameters['step_size'],
                                        gamma=hyperparameters['gamma'], last_epoch=iterations)
    else:
        return NotImplementedError('learning rate policy [%s] is not implemented', hyperparameters['lr_policy'])
    return scheduler


def weights_init(init_type='gaussian'):
    def init_fun(m):
        classname = m.__class__.__name__
        if (classname.find('Conv') == 0 or classname.find('Linear') == 0) and hasattr(m, 'weight'):
            # print m.__class__.__name__
            if init_type == 'gaussian':
                init.normal_(m.weight.data, 0.0, 0.02)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=math.sqrt(2))
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=math.sqrt(2))
            elif init_type == 'default':
                pass
            else:
                assert 0, "Unsupported initialization: {}".format(init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)

    return init_fun


class Timer:
    def __init__(self, msg):
        self.msg = msg
        self.start_time = None

    def __enter__(self):
        self.start_time = time.time()

    def __exit__(self, exc_type, exc_value, exc_tb):
        print(self.msg % (time.time() - self.start_time))


def pytorch03_to_pytorch04(state_dict_base):
    def __conversion_core(state_dict_base):
        state_dict = state_dict_base.copy()
        for key, value in state_dict_base.items():
            if key.endswith(('enc.model.0.norm.running_mean',
                             'enc.model.0.norm.running_var',
                             'enc.model.1.norm.running_mean',
                             'enc.model.1.norm.running_var',
                             'enc.model.2.norm.running_mean',
                             'enc.model.2.norm.running_var',
                             'enc.model.3.model.0.model.1.norm.running_mean',
                             'enc.model.3.model.0.model.1.norm.running_var',
                             'enc.model.3.model.0.model.0.norm.running_mean',
                             'enc.model.3.model.0.model.0.norm.running_var',
                             'enc.model.3.model.1.model.1.norm.running_mean',
                             'enc.model.3.model.1.model.1.norm.running_var',
                             'enc.model.3.model.1.model.0.norm.running_mean',
                             'enc.model.3.model.1.model.0.norm.running_var',
                             'enc.model.3.model.2.model.1.norm.running_mean',
                             'enc.model.3.model.2.model.1.norm.running_var',
                             'enc.model.3.model.2.model.0.norm.running_mean',
                             'enc.model.3.model.2.model.0.norm.running_var',
                             'enc.model.3.model.3.model.1.norm.running_mean',
                             'enc.model.3.model.3.model.1.norm.running_var',
                             'enc.model.3.model.3.model.0.norm.running_mean',
                             'enc.model.3.model.3.model.0.norm.running_var',
                             'dec.model.0.model.0.model.1.norm.running_mean',
                             'dec.model.0.model.0.model.1.norm.running_var',
                             'dec.model.0.model.0.model.0.norm.running_mean',
                             'dec.model.0.model.0.model.0.norm.running_var',
                             'dec.model.0.model.1.model.1.norm.running_mean',
                             'dec.model.0.model.1.model.1.norm.running_var',
                             'dec.model.0.model.1.model.0.norm.running_mean',
                             'dec.model.0.model.1.model.0.norm.running_var',
                             'dec.model.0.model.2.model.1.norm.running_mean',
                             'dec.model.0.model.2.model.1.norm.running_var',
                             'dec.model.0.model.2.model.0.norm.running_mean',
                             'dec.model.0.model.2.model.0.norm.running_var',
                             'dec.model.0.model.3.model.1.norm.running_mean',
                             'dec.model.0.model.3.model.1.norm.running_var',
                             'dec.model.0.model.3.model.0.norm.running_mean',
                             'dec.model.0.model.3.model.0.norm.running_var',
                             )):
                del state_dict[key]
        return state_dict

    state_dict = dict()
    state_dict['a'] = __conversion_core(state_dict_base['a'])
    state_dict['b'] = __conversion_core(state_dict_base['b'])
    return state_dict


def save_mat(path, mat):
    with h5py.File(path, 'w') as hf:
        hf.create_dataset('rad', data=mat)
        hf.close()


####################
# metric
####################

def calculate_rmse(sr_img, gt_img):
    """Calculate the relative RMSE"""
    sr_img = sr_img.astype(np.float64)
    gt_img = gt_img.astype(np.float64)
    rmse = np.sqrt(np.mean((sr_img - gt_img)**2))
    return rmse


def calculate_psnr(sr_img, gt_img):
    # img1 and img2 have range [0, MAX]
    sr_img = sr_img.astype(np.float64)
    gt_img = gt_img.astype(np.float64)
    mse = np.mean((sr_img - gt_img)**2)
    if mse == 0:
        return float('inf')
    return 20 * math.log10(np.max(gt_img) / math.sqrt(mse))


def calculate_ssim(img_res, img_gt):
    """Calculate the structural similarity"""
    num=img_gt.shape[2]
    ssimm=np.zeros(num)
    c1=0.0001
    c2=0.0009
    n=0
    for x in range(img_gt.shape[2]):
        z = np.reshape(img_res[:, :, x], [-1])
        sa = np.reshape(img_gt[:, :, x], [-1])
        y = [z, sa]
        cov = np.cov(y)
        oz = cov[0, 0]
        osa = cov[1, 1]
        ozsa = cov[0, 1]
        ez = np.mean(z)
        esa = np.mean(sa)
        ssimm[n] = ((2 * ez * esa + c1) * (2 * ozsa + c2)) / ((ez * ez + esa * esa + c1) * (oz + osa + c2))
        n = n + 1
    SSIM=np.mean(ssimm)
    return SSIM


def calculate_sam(sr_img, gt_img):
    """Calculate the spectral angle mapping"""
    sam = 0
    for i in range(gt_img.shape[0]):
        for j in range(gt_img.shape[1]):
            x = np.reshape(sr_img[i, j, :], [-1])
            y = np.reshape(gt_img[i, j, :], [-1])
            tem1 = np.dot(x, y)
            tem2 = (np.linalg.norm(x)) * (np.linalg.norm(y))
            sam += np.arccos(tem1 / tem2)

    sam = (sam / (gt_img.shape[0] * gt_img.shape[1])) * 180 / np.pi
    return sam


def calculate_ergas(sr_img, gt_img, resize_factor=4):
    """Error relative global dimension de synthesis (ERGAS)
    reference: https://github.com/amteodoro/SA-PnP-GMM/blob/9e8dffab223d88642545d1760669e2326efe0973/Misc/ERGAS.m
    """
    sr_img = sr_img.astype(np.float64)
    gt_img = gt_img.astype(np.float64)
    err = sr_img - gt_img
    ergas = 0
    for i in range(err.shape[2]):
        ergas += np.mean(err[:, :, i]**2) / (np.mean(sr_img[:, :, i]))**2

    ergas = (100. / float(resize_factor)) * np.sqrt(1. / err.shape[2] * ergas)
    return ergas


def evaluation_matrix(img_res, img_gt, resize_fact=4):
    matrix = {
        'RMSE': calculate_rmse(img_res, img_gt),
        'SSIM': calculate_ssim(img_res, img_gt),
        'ERGAS': calculate_ergas(img_res, img_gt, resize_factor=resize_fact),
        'SAM': calculate_sam(img_res, img_gt)
    }
    return matrix
