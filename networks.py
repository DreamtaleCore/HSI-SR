"""
models for 'Hyperspectral Image Super-Resolution'
"""
import functools
import math
import numpy as np
from queue import Queue
import torch
from torch import nn
import torch.nn.init as init
from torch.autograd import Variable
import torch.nn.functional as F
from torch.autograd import grad as ta_grad

try:
    from itertools import izip as zip
except ImportError:  # will be 3.x series
    pass


# utils, will be placed in package utils
def initialize_weights(net_l, scale=1):
    if not isinstance(net_l, list):
        net_l = [net_l]
    for net in net_l:
        for m in net.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, a=0, mode='fan_in')
                m.weight.data *= scale  # for residual block
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                init.kaiming_normal_(m.weight, a=0, mode='fan_in')
                m.weight.data *= scale
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias.data, 0.0)


##################################################################################
# Interfaces
##################################################################################

def get_discriminator(dis_opt, train_mode=None):
    """Get a discriminator"""
    # multi-scale dis
    return Discriminator(dis_opt)

def get_teacher(opt_net, train_mode=None):
    """Get a teacher"""
    return SRResNet(in_nc=opt_net['in_nc'], out_nc=opt_net['out_nc'],
                                       nf=opt_net['nf'], nb=opt_net['nb'], upscale=opt_net['scale'])

def get_student(opt_net, train_mode=None):
    """Get a student"""
    return SRResNet(in_nc=opt_net['in_nc'], out_nc=opt_net['out_nc'],
                                       nf=opt_net['nf'], nb=opt_net['nb'], upscale=opt_net['scale'])


##################################################################################
# Generator(SRResNet)
##################################################################################

class SRResNet(nn.Module):
    ''' modified SRResNet'''

    def __init__(self, in_nc=3, out_nc=3, nf=64, nb=16, upscale=4):
        super(SRResNet, self).__init__()
        self.upscale = upscale

        self.conv_first = nn.Conv2d(in_nc, nf, 3, 1, 1, bias=True)
        basic_block = functools.partial(ResidualBlock_noBN, nf=nf)
        self.recon_trunk = self._make_layer(basic_block, nb)

        # upsampling
        if self.upscale == 2:
            self.upconv1 = nn.Conv2d(nf, nf * 4, 3, 1, 1, bias=True)
            self.pixel_shuffle = nn.PixelShuffle(2)
        elif self.upscale == 3:
            self.upconv1 = nn.Conv2d(nf, nf * 9, 3, 1, 1, bias=True)
            self.pixel_shuffle = nn.PixelShuffle(3)
        elif self.upscale == 4:
            self.upconv1 = nn.Conv2d(nf, nf * 4, 3, 1, 1, bias=True)
            self.upconv2 = nn.Conv2d(nf, nf * 4, 3, 1, 1, bias=True)
            self.pixel_shuffle = nn.PixelShuffle(2)

        self.HRconv = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.conv_last = nn.Conv2d(nf, out_nc, 3, 1, 1, bias=True)

        # activation function
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

        # initialization
        initialize_weights([self.conv_first, self.upconv1, self.HRconv, self.conv_last],
                                     0.1)
        if self.upscale == 4:
            initialize_weights(self.upconv2, 0.1)

    def _make_layer(self, block, n_layers):
        layers = []
        for _ in range(n_layers):
            layers.append(block())
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.lrelu(self.conv_first(x))
        fea = self.recon_trunk(out)

        if self.upscale == 4:
            out = self.lrelu(self.pixel_shuffle(self.upconv1(fea)))
            out = self.lrelu(self.pixel_shuffle(self.upconv2(out)))
        elif self.upscale == 3 or self.upscale == 2:
            out = self.lrelu(self.pixel_shuffle(self.upconv1(fea)))

        out = self.conv_last(self.lrelu(self.HRconv(out)))
        base = F.interpolate(x, scale_factor=self.upscale, mode='bilinear', align_corners=False)
        out += base

        out_feature = {
            'input': x,
            'mid': fea,
            'out': out
        }
        return out_feature


##################################################################################
# Discriminator
##################################################################################

class Discriminator(nn.Module):
    """Discriminator."""
    def __init__(self):
        super(Discriminator, self).__init__()


##################################################################################
# Basic Blocks
##################################################################################

# define a Residual block w/o BN
class ResidualBlock_noBN(nn.Module):
    '''Residual block w/o BN
    ---Conv-ReLU-Conv-+-
     |________________|
    '''
    def __init__(self, nf=64):
        super(ResidualBlock_noBN, self).__init__()
        self.conv1 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.conv2 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)

        # initialization
        initialize_weights([self.conv1, self.conv2], 0.1)

    def forward(self, x):
        identity = x
        out = F.relu(self.conv1(x), inplace=True)
        out = self.conv2(out)
        return identity + out


##################################################################################
# Loss function
##################################################################################

def loss_function():
    pass


##################################################################################
# Test codes
##################################################################################

def test_network():
    import data
    dataset = 'ICVL151_sub'  # ICVL151_sub
    opt = {}
    opt['dist'] = False
    opt['gpu_ids'] = [0]
    opt['name'] = dataset
    opt['dataroot_GT'] = '/home/gyj/workspace/MMSR/datasets/valid/mat'
    opt['dataroot_LQ'] = '/home/gyj/workspace/MMSR/datasets/valid/mat_bicLRx4'
    opt['mode'] = 'LQGT'
    opt['phase'] = 'val'
    opt['use_shuffle'] = True
    opt['n_workers'] = 0
    opt['batch_size'] = 16
    opt['GT_size'] = 128
    opt['scale'] = 4
    opt['use_flip'] = False
    opt['use_rot'] = False
    opt['data_type'] = 'img'  # img

    opt['in_nc'] = 31
    opt['out_nc'] = 31
    opt['nf'] = 64
    opt['nb'] = 16
    opt['upscale'] = 4

    train_set = data.create_dataset(opt)
    train_loader = data.create_dataloader(train_set, opt, opt, None)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = get_teacher(opt).to(device)
    criterion = nn.L1Loss().to(device)

    print('start...')
    for i, data in enumerate(train_loader):
        if i > 5:
            break

        print(i)
        LQ = data['LQ'].to(device)
        print('LQ size:', LQ.shape)
        GT = data['GT'].to(device)
        print('GT size:', GT.shape)

        fake = model(LQ)
        print('fake size:', fake['out'].shape)
        loss = criterion(fake['out'], GT)
        print('loss:', loss)


if __name__ == '__main__':
    test_network()
