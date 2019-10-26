from torch.utils.data import DataLoader
from torch.autograd import Variable
from torch.optim import lr_scheduler
from torchvision import transforms
from data import LowLevelImageFolder
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

# Merge into network.py latter
class CriterionPixelWise(nn.Module):
    def __init__(self, ignore_index=255, use_weight=True, reduce=True):
        super(CriterionPixelWise, self).__init__()
        self.ignore_index = ignore_index
        self.criterion = torch.nn.CrossEntropyLoss(ignore_index=ignore_index, reduce=reduce)
        if not reduce:
            print("disabled the reduce.")

    def forward(self, pred_stu, pred_tea):
        pred_tea[0].detach()
        assert pred_stu[0].shape == pred_tea[0].shape, 'the output dim of teacher and student differ'
        N, C, W, H = pred_stu[0].shape
        softmax_pred_T = F.softmax(pred_tea[0].view(-1, C), dim=1)
        logsoftmax = nn.LogSoftmax(dim=1)
        loss = (torch.sum(- softmax_pred_T * logsoftmax(pred_stu[0].view(-1, C)))) / W / H
        return loss


def L2(f_):
    return (((f_**2).sum(dim=1))**0.5).reshape(f_.shape[0], 1, f_.shape[2], f_.shape[3]) + 1e-8


def similarity(feat):
    feat = feat.float()
    tmp = L2(feat).detach()
    feat = feat/tmp
    feat = feat.reshape(feat.shape[0],feat.shape[1], -1)
    return torch.einsum('icm,icn->imn', [feat, feat])


def sim_dis_compute(f_src, f_tar):
    sim_err = ((similarity(f_tar) - similarity(f_src)) ** 2) / ((f_tar.shape[-1] * f_tar.shape[-2]) ** 2) / f_tar.shape[0]
    sim_dis = sim_err.sum()
    return sim_dis


class CriterionPairWiseForWholeFeatAfterPool(nn.Module):
    def __init__(self, scale):
        """inter pair-wise loss from inter feature maps"""
        super(CriterionPairWiseForWholeFeatAfterPool, self).__init__()
        self.criterion = sim_dis_compute
        self.scale = scale

    def forward(self, preds_stu, preds_tea):
        loss = 0
        for k, v in preds_stu:
            feat_S = preds_stu[k]
            feat_T = preds_tea[k]
            feat_T.detach()

            total_w, total_h = feat_T.shape[2], feat_T.shape[3]
            patch_w, patch_h = int(total_w * self.scale), int(total_h * self.scale)
            maxpool = nn.MaxPool2d(kernel_size=(patch_w, patch_h), stride=(patch_w, patch_h), padding=0,
                                   ceil_mode=True)  # change
            loss += self.criterion(maxpool(feat_S), maxpool(feat_T))
        loss = loss / len(preds_stu)
        return loss


class CriterionMerge(nn.Module):
    def __init__(self, param):
        super(CriterionMerge, self).__init__()
        self.loss_pair = CriterionPairWiseForWholeFeatAfterPool(param.pairwise_scale)
        self.loss_l2 = L2
        pass

    def forward(self, stu_pred, tea_pred, y_out, param):
        l_pair = self.loss_pair(stu_pred, tea_pred)
        l_pixel = self.loss_l2(stu_pred['out'] - tea_pred['out'])

        loss = param.pairwise_w * l_pair + param.pixel_w * l_pixel
        return loss


def get_criterion(param):
    return CriterionMerge(param)


def to_number(data):
    if type(data) is torch.Tensor:
        return data.item()
    else:
        return data


def get_local_time():
    return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())


def get_reflection_data_loader(conf):
    train_set = LowLevelImageFolder(conf, is_train=True)
    eval_set = LowLevelImageFolder(conf, is_train=False)
    train_loader = DataLoader(train_set)
    eval_loader = DataLoader(eval_set)
    return train_loader, eval_loader


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
