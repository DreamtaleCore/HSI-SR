"""
Interface of DNN models for image translations
"""
import functools

import math
import numpy as np
from queue import Queue
from torch import nn
from torch.autograd import Variable
import torch
from torch.autograd import grad as ta_grad
import torch.nn.functional as F
from torchvision.models import vgg11, vgg19

try:
    from itertools import izip as zip
except ImportError:  # will be 3.x series
    pass


##################################################################################
# Interfaces
##################################################################################

def get_discriminator(dis_opt, train_mode=None):
    """Get a discriminator"""
    # multi-scale dis
    return MsImageDis(dis_opt)


def get_generator(gen_opt, train_mode=None):
    """Get a generator"""
    return DirectGenMS(gen_opt)


##################################################################################
# Discriminator
##################################################################################

class MsImageDis(nn.Module):
    # Multi-scale discriminator architecture
    def __init__(self, dis_opt):
        super(MsImageDis, self).__init__()
        self.dim = dis_opt.dis.dim
        self.norm = dis_opt.dis.norm
        self.activ = dis_opt.dis.activ
        self.pad_type = dis_opt.dis.pad_type
        self.gan_type = dis_opt.dis.gan_type
        self.n_layers = dis_opt.dis.n_layer
        self.use_grad = dis_opt.dis.use_grad
        self.input_dim = dis_opt.input_dim
        self.num_scales = dis_opt.dis.num_scales
        self.use_wasserstein = dis_opt.dis.use_wasserstein
        self.grad_w = dis_opt.grad_w
        self.downsample = nn.AvgPool2d(3, stride=2, padding=[1, 1], count_include_pad=False)
        self.models = nn.ModuleList()
        self.sigmoid_func = nn.Sigmoid()

        for _ in range(self.num_scales):
            cnns = self._make_net()
            if self.use_wasserstein:
                cnns += [nn.Sigmoid()]

            self.models.append(nn.Sequential(*cnns))

    def _make_net(self):
        dim = self.dim
        cnn_x = []
        cnn_x += [Conv2dBlock(self.input_dim, dim, 4, 2, 1, norm='none', activation=self.activ, pad_type=self.pad_type)]
        for i in range(self.n_layers - 1):
            cnn_x += [Conv2dBlock(dim, dim * 2, 4, 2, 1, norm=self.norm, activation=self.activ, pad_type=self.pad_type)]
            dim *= 2
        cnn_x += [nn.Conv2d(dim, 1, 1, 1, 0)]
        return cnn_x

    def forward(self, x):
        output = None
        for model in self.models:
            out = model(x)
            if output is not None:
                _, _, h, w = out.shape
                output = F.interpolate(output, size=(h, w), mode='bilinear', align_corners=True)
                output = output + out
            else:
                output = out

            x = self.downsample(x)

        output = output / len(self.models)
        output = self.sigmoid_func(output)

        return output

    def calc_dis_loss(self, input_fake, input_real):
        # calculate the loss to train D
        outs0 = self.forward(input_fake)
        outs1 = self.forward(input_real)
        loss = 0

        for it, (out0, out1) in enumerate(zip(outs0, outs1)):
            if self.gan_type == 'lsgan':
                loss += torch.mean((out0 - 0) ** 2) + torch.mean((out1 - 1) ** 2)
            elif self.gan_type == 'nsgan':
                all0 = Variable(torch.zeros_like(out0.data).cuda(), requires_grad=False)
                all1 = Variable(torch.ones_like(out1.data).cuda(), requires_grad=False)
                loss += torch.mean(F.binary_cross_entropy(F.sigmoid(out0), all0) +
                                   F.binary_cross_entropy(F.sigmoid(out1), all1))
            else:
                assert 0, "Unsupported GAN type: {}".format(self.gan_type)

            # Gradient penalty
            grad_loss = 0
            if self.use_grad:
                eps = Variable(torch.rand(1), requires_grad=True)
                eps = eps.expand(input_real.size())
                eps = eps.cuda()
                x_tilde = eps * input_real + (1 - eps) * input_fake
                x_tilde = x_tilde.cuda()
                pred_tilde = self.calc_gen_loss(x_tilde)
                gradients = ta_grad(outputs=pred_tilde, inputs=x_tilde,
                                    grad_outputs=torch.ones(pred_tilde.size()).cuda(),
                                    create_graph=True, retain_graph=True, only_inputs=True)[0]
                grad_loss = self.grad_w * gradients

                input_real = self.downsample(input_real)
                input_fake = self.downsample(input_fake)

            loss += ((grad_loss.norm(2, dim=1) - 1) ** 2).mean()

        return loss

    def calc_gen_loss(self, input_fake):
        # calculate the loss to train G
        outs0 = self.forward(input_fake)
        loss = 0
        for it, (out0) in enumerate(outs0):
            if self.gan_type == 'lsgan':
                loss += torch.mean((out0 - 1) ** 2)  # LSGAN
            elif self.gan_type == 'nsgan':
                all1 = Variable(torch.ones_like(out0.data).to(self.device), requires_grad=True)
                loss += torch.mean(F.binary_cross_entropy(F.sigmoid(out0), all1))
            else:
                assert 0, "Unsupported GAN type: {}".format(self.gan_type)
        return loss


##################################################################################
# Generator
##################################################################################

class DirectGenMS(nn.Module):
    def __init__(self, gen_opt):
        super(DirectGenMS, self).__init__()
        self.dim = gen_opt.gen.dim
        self.norm = gen_opt.gen.norm
        self.activ = gen_opt.gen.activ
        self.pad_type = gen_opt.gen.pad_type
        self.n_layers = gen_opt.gen.n_layer
        self.input_dim = gen_opt.input_dim
        self.pretrained = gen_opt.gen.vgg_pretrained
        self.output_dim = gen_opt.output_dim
        self.decoder_mode = gen_opt.gen.decoder_mode

        encoder_name = gen_opt.gen.encoder_name

        # Feature extractor as Encoder
        if encoder_name == 'vgg11':
            self.encoder = Vgg11EncoderMS(pretrained=self.pretrained)
        elif encoder_name == 'vgg19':
            self.encoder = Vgg19EncoderMS(pretrained=self.pretrained)
        else:
            raise ValueError('encoder name should in [vgg11/vgg19], but it is: {}'.format(encoder_name))

        self.decoder = DecoderMS(self.input_dim, dim=self.dim, output_dim=self.output_dim,
                                 n_layers=self.n_layers, pad_type=self.pad_type, activ=self.activ,
                                 norm=self.norm, decoder_mode=self.decoder_mode)

    def decode(self, x, feats=None):
        return self.decoder(x, feats)

    def encode(self, x):
        return self.encoder(x)

    def forward(self, x):
        feats = self.encoder(x)
        out = self.decode(x, feats)
        if self.output_dim == 6:
            out_a = out[:, :3, :, :]
            out_b = out[:, 3:, :, :]
            return out_a, out_b
        return out, feats


##################################################################################
# Encoder and Decoders
##################################################################################

class Vgg11EncoderMS(nn.Module):
    """Vgg encoder wiht multi-scales"""

    def __init__(self, pretrained):
        super(Vgg11EncoderMS, self).__init__()
        features = list(vgg11(pretrained=pretrained).features)
        self.features = nn.ModuleList(features)

    def forward(self, x):
        result_dict = {}
        layer_names = ['conv1_1',
                       'conv2_1',
                       'conv3_1', 'conv3_2',
                       'conv4_1', 'conv4_2',
                       'conv5_1', 'conv5_2']
        idx = 0
        for ii, model in enumerate(self.features):
            x = model(x)
            if ii in {0, 3, 6, 8, 11, 13, 16, 18}:
                result_dict[layer_names[idx]] = x
                idx += 1

        out_feature = {
            'input': x,
            'shallow': result_dict['conv1_1'],
            'low': result_dict['conv2_1'],
            'mid': result_dict['conv3_2'],
            'deep': result_dict['conv3_2'],
            'out': result_dict['conv5_2']
        }
        return out_feature


class Vgg19EncoderMS(nn.Module):
    def __init__(self, pretrained):
        super(Vgg19EncoderMS, self).__init__()
        features = list(vgg19(pretrained=pretrained).features)
        self.features = nn.ModuleList(features)

    def forward(self, x):
        result_dict = {}
        layer_names = ['conv1_1', 'conv1_2',
                       'conv2_1', 'conv2_2',
                       'conv3_1', 'conv3_2', 'conv3_3', 'conv3_4',
                       'conv4_1', 'conv4_2', 'conv4_3', 'conv4_4',
                       'conv5_1', 'conv5_2', 'conv5_3', 'conv5_4']
        idx = 0
        for ii, model in enumerate(self.features):
            x = model(x)
            if ii in {0, 2, 5, 7, 10, 12, 14, 16, 19, 21, 23, 25, 28, 30, 32, 34}:
                result_dict[layer_names[idx]] = x
                idx += 1

        out_feature = {
            'input': x,
            'shallow': result_dict['conv1_2'],
            'low': result_dict['conv2_2'],
            'mid': result_dict['conv3_2'],
            'deep': result_dict['conv4_2'],
            'out': result_dict['conv5_2']
        }
        return out_feature


class DecoderMS(nn.Module):
    def __init__(self, input_dim, dim, output_dim, n_layers, pad_type, activ, norm, decoder_mode='Basic'):
        """output_shape = [H, W, C]"""
        super(DecoderMS, self).__init__()

        # fusion block
        self.fuse_out = Conv2dBlock(512, 256, kernel_size=3, stride=1,
                                    pad_type=pad_type, activation=activ, norm=norm)
        self.fuse_deep = Conv2dBlock(768, 128, kernel_size=3, stride=1,
                                     pad_type=pad_type, activation=activ, norm=norm)        # in channel 512 for vgg11
        self.fuse_mid = Conv2dBlock(384, 64, kernel_size=3, stride=1,
                                    pad_type=pad_type, activation=activ, norm=norm)
        self.fuse_low = Conv2dBlock(192, 32, kernel_size=3, stride=1,
                                    pad_type=pad_type, activation=activ, norm=norm)
        self.fuse_shallow = Conv2dBlock(96, 16, kernel_size=3, stride=1,
                                        pad_type=pad_type, activation=activ, norm=norm)
        self.fuse_input = Conv2dBlock(16 + input_dim, dim, kernel_size=3, stride=1, padding=1,
                                      pad_type=pad_type, activation=activ, norm=norm)
        self.contextual_blocks = []

        rates = [1, 3, 5, 9, 13]
        if n_layers > 5:
            raise NotImplementedError('contextual layer should less or equal to 5')
        if decoder_mode == 'Basic':
            for i in range(n_layers):
                self.contextual_blocks += [Conv2dBlock(dim, dim, kernel_size=3, dilation=rates[i], padding=rates[i],
                                                       pad_type=pad_type, activation=activ, norm=norm)]
        elif decoder_mode == 'Residual':
            for i in range(n_layers):
                self.contextual_blocks += [ResDilateBlock(input_dim=dim, dim=dim, output_dim=dim, rate=rates[i],
                                                          pad_type=pad_type, activation=activ, norm=norm)]
        else:
            raise NotImplementedError

        # use reflection padding in the last conv layer
        self.contextual_blocks += [
            Conv2dBlock(dim, dim, kernel_size=3, padding=1, norm=norm, activation=activ, pad_type=pad_type)]
        self.contextual_blocks += [
            Conv2dBlock(dim, output_dim, kernel_size=1, norm='none', activation='none', pad_type=pad_type)]
        self.contextual_blocks = nn.Sequential(*self.contextual_blocks)

    @staticmethod
    def _fuse_feature(x, feature):
        _, _, h, w = feature.shape
        x = F.interpolate(x, size=(h, w), mode='bilinear', align_corners=False)
        x = torch.cat([x, feature], dim=1)
        return x

    def forward(self, input_x, feat_dict):
        x = feat_dict['out']
        x = self.fuse_out(x)
        x = self._fuse_feature(x, feat_dict['deep'])
        x = self.fuse_deep(x)
        x = self._fuse_feature(x, feat_dict['mid'])
        x = self.fuse_mid(x)
        x = self._fuse_feature(x, feat_dict['low'])
        x = self.fuse_low(x)
        x = self._fuse_feature(x, feat_dict['shallow'])
        x = self.fuse_shallow(x)
        x = self._fuse_feature(x, input_x)
        x = self.fuse_input(x)

        x = self.contextual_blocks(x)
        return x


##################################################################################
# Basic Blocks
##################################################################################

class Conv2dBlock(nn.Module):
    def __init__(self, input_dim, output_dim, kernel_size, stride=1,
                 padding=0, norm='none', activation='relu', pad_type='zero', dilation=1):
        super(Conv2dBlock, self).__init__()
        self.use_bias = True
        # initialize padding
        if pad_type == 'reflect':
            self.pad = nn.ReflectionPad2d(padding)
        elif pad_type == 'replicate':
            self.pad = nn.ReplicationPad2d(padding)
        elif pad_type == 'zero':
            self.pad = nn.ZeroPad2d(padding)
        else:
            assert 0, "Unsupported padding type: {}".format(pad_type)

        # initialize normalization
        norm_dim = output_dim
        if norm == 'bn':
            self.norm = nn.BatchNorm2d(norm_dim)
        elif norm == 'in':
            # self.norm = nn.InstanceNorm2d(norm_dim, track_running_stats=True)
            self.norm = nn.InstanceNorm2d(norm_dim)
        elif norm == 'ln':
            self.norm = LayerNorm(norm_dim)
        elif norm == 'none':
            self.norm = None
        else:
            assert 0, "Unsupported normalization: {}".format(norm)

        # initialize activation
        if activation == 'relu':
            self.activation = nn.ReLU(inplace=True)
        elif activation == 'lrelu':
            self.activation = nn.LeakyReLU(0.2, inplace=True)
        elif activation == 'prelu':
            self.activation = nn.PReLU()
        elif activation == 'selu':
            self.activation = nn.SELU(inplace=True)
        elif activation == 'tanh':
            self.activation = nn.Tanh()
        elif activation == 'sigmoid':
            self.activation = nn.Sigmoid()
        elif activation == 'none':
            self.activation = None
        else:
            assert 0, "Unsupported activation: {}".format(activation)

        # initialize convolution
        self.conv = nn.Conv2d(input_dim, output_dim, kernel_size, stride,
                              bias=self.use_bias, dilation=dilation)

    def forward(self, x):
        x = self.conv(self.pad(x))
        # if self.norm:
        #     x = self.norm(x)
        if self.activation:
            x = self.activation(x)
        return x


# define a ResNet(dilated) block
class ResDilateBlock(nn.Module):
    def __init__(self, input_dim, dim, output_dim, rate,
                 padding=0, norm='none', activation='relu', pad_type='zero', use_bias=False):
        super(ResDilateBlock, self).__init__()
        # initialize activation
        if activation == 'relu':
            self.activation = nn.ReLU(inplace=True)
        elif activation == 'lrelu':
            self.activation = nn.LeakyReLU(0.2, inplace=True)
        elif activation == 'prelu':
            self.activation = nn.PReLU()
        elif activation == 'selu':
            self.activation = nn.SELU(inplace=True)
        elif activation == 'tanh':
            self.activation = nn.Tanh()
        elif activation == 'sigmoid':
            self.activation = nn.Sigmoid()
        elif activation == 'none':
            self.activation = None
        else:
            assert 0, "Unsupported activation: {}".format(activation)

        if pad_type == 'reflect':
            self.pad = nn.ReflectionPad2d(padding)
        elif pad_type == 'replicate':
            self.pad = nn.ReplicationPad2d(padding)
        elif pad_type == 'zero':
            self.pad = nn.ZeroPad2d(padding)
        else:
            assert 0, "Unsupported padding type: {}".format(pad_type)

        feature_, conv_block = self.build_conv_block(input_dim, dim, output_dim, rate,
                                                     pad_type, norm, use_bias)
        self.feature_ = feature_
        self.conv_block = conv_block

    def build_conv_block(self, input_dim, dim, output_dim, rate,
                         padding_type, norm, use_bias=False):

        # branch feature_: in case the output_dim is different from input
        feature_ = [self.pad_layer(padding_type, padding=0),
                    nn.Conv2d(input_dim, output_dim, kernel_size=1, stride=1,
                              bias=False, dilation=1),
                    self.norm_layer(norm, output_dim),
                    ]
        feature_ = nn.Sequential(*feature_)

        # branch convolution:
        conv_block = []

        conv_block += [self.pad_layer(padding_type, padding=0),
                       nn.Conv2d(input_dim, dim, kernel_size=1, stride=1,
                                 bias=False, dilation=1),
                       self.norm_layer(norm, dim),
                       self.activation]
        # dilated conv, padding = dilation_rate, when k=3, s=1
        conv_block += [self.pad_layer(padding_type, padding=rate),
                       nn.Conv2d(dim, dim, kernel_size=3, stride=1,
                                 bias=False, dilation=rate),
                       self.norm_layer(norm, dim),
                       self.activation]
        conv_block += [self.pad_layer(padding_type, padding=0),
                       nn.Conv2d(dim, output_dim, kernel_size=1, stride=1,
                                 bias=False, dilation=1),
                       self.norm_layer(norm, output_dim),
                       ]
        conv_block = nn.Sequential(*conv_block)
        return feature_, conv_block

    @staticmethod
    def pad_layer(padding_type, padding):
        if padding_type == 'reflect':
            pad = nn.ReflectionPad2d(padding)
        elif padding_type == 'replicate':
            pad = nn.ReplicationPad2d(padding)
        elif padding_type == 'zero':
            pad = nn.ZeroPad2d(padding)
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)
        return pad

    @staticmethod
    def norm_layer(norm, norm_dim):
        if norm == 'bn':
            norm_layer_ = nn.BatchNorm2d(norm_dim)
        elif norm == 'in':
            # self.norm = nn.InstanceNorm2d(norm_dim, track_running_stats=True)
            norm_layer_ = nn.InstanceNorm2d(norm_dim)
        elif norm == 'ln':
            norm_layer_ = LayerNorm(norm_dim)
        elif norm == 'none':
            norm_layer_ = None
        else:
            assert 0, "Unsupported normalization: {}".format(norm)
        return norm_layer_

    def forward(self, x):
        feature_ = self.feature_(x)
        conv = self.conv_block(x)
        out = feature_ + conv
        out = self.activation(out)
        return out


##################################################################################
# Normalization layers
##################################################################################

class LayerNorm(nn.Module):
    def __init__(self, num_features, eps=1e-5, affine=True):
        super(LayerNorm, self).__init__()
        self.num_features = num_features
        self.affine = affine
        self.eps = eps

        if self.affine:
            self.gamma = nn.Parameter(torch.Tensor(num_features).uniform_())
            self.beta = nn.Parameter(torch.zeros(num_features))

    def forward(self, x):
        shape = [-1] + [1] * (x.dim() - 1)
        # print(x.size())
        if x.size(0) == 1:
            # These two lines run much faster in pytorch 0.4 than the two lines listed below.
            mean = x.view(-1).mean().view(*shape)
            std = x.view(-1).std().view(*shape)
        else:
            mean = x.view(x.size(0), -1).mean(1).view(*shape)
            std = x.view(x.size(0), -1).std(1).view(*shape)

        x = (x - mean) / (std + self.eps)

        if self.affine:
            shape = [1, -1] + [1] * (x.dim() - 2)
            x = x * self.gamma.view(*shape) + self.beta.view(*shape)
        return x


def _get_norm_layer(norm_type='in'):
    if norm_type == 'bn':
        norm_layer = functools.partial(nn.BatchNorm2d, affine=True)
    elif norm_type == 'in':
        norm_layer = functools.partial(nn.InstanceNorm2d, affine=False)
    elif norm_type == 'none':
        norm_layer = None
    else:
        raise NotImplementedError('normalization layer [%s] is not found' % norm_type)
    return norm_layer


def _get_active_function(act_type='relu'):
    if act_type == 'relu':
        act_func = nn.ReLU(True)
    elif act_type == 'lrelu':
        act_func = nn.LeakyReLU(0.2, True)
    elif act_type == 'prelu':
        act_func = nn.PReLU()
    elif act_type == 'selu':
        act_func = nn.SELU(inplace=True)
    elif act_type == 'sigmoid':
        act_func = nn.Sigmoid()
    elif act_type == 'tanh':
        act_func = nn.Tanh()
    elif act_type == 'none':
        act_func = None
    else:
        raise NotImplementedError('activation function [%s] is not found' % act_type)
    return act_func


##################################################################################
# Distribution distance measurements and losses blocks
##################################################################################


class KLDivergence(nn.Module):
    def __init__(self, size_average=None, reduce=True, reduction='mean'):
        super(KLDivergence, self).__init__()
        self.eps = 1e-12
        self.log_softmax = nn.LogSoftmax()
        self.kld = nn.KLDivLoss(size_average=size_average, reduce=reduce, reduction=reduction)
        pass

    def forward(self, x, y):
        # normalize
        x = self.log_softmax(x)
        y = self.log_softmax(y)
        return self.kld(x, y)


class JSDivergence(KLDivergence):
    def __init__(self, size_average=True, reduce=True, reduction='mean'):
        super(JSDivergence, self).__init__(size_average, reduce, reduction)

    def forward(self, x, y):
        # normalize
        x = self.log_softmax(x)
        y = self.log_softmax(y)
        m = 0.5 * (x + y)

        return 0.5 * (self.kld(x, m) + self.kld(y, m))


class SSIMLoss(nn.Module):
    def __init__(self, window_size=11, size_average=True):
        """
        SSIM Loss, return 1 - SSIM
        """
        super(SSIMLoss, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.channel = 1
        self.window = self._create_window(window_size, self.channel)

    @staticmethod
    def _gaussian(window_size, sigma):
        gauss = torch.Tensor(
            [math.exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
        return gauss / gauss.sum()

    def _create_window(self, window_size, channel):
        _1D_window = self._gaussian(window_size, 1.5).unsqueeze(1)
        _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
        window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
        return window

    @staticmethod
    def _ssim(img1, img2, window, window_size, channel, size_average=True):
        mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
        mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

        mu1_sq = mu1.pow(2)
        mu2_sq = mu2.pow(2)
        mu1_mu2 = mu1 * mu2

        sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
        sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
        sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2

        C1 = 0.01 ** 2
        C2 = 0.03 ** 2

        ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

        if size_average:
            return ssim_map.mean()
        else:
            return ssim_map.mean(1).mean(1).mean(1)

    def forward(self, img_a, img_b):
        (_, channel, _, _) = img_a.size()

        if channel == self.channel and self.window.data.type() == img_a.data.type():
            window = self.window
        else:
            window = self._create_window(self.window_size, channel)

            if img_a.is_cuda:
                window = window.cuda(img_a.get_device())
            window = window.type_as(img_a)

            self.window = window
            self.channel = channel

        ssim_v = self._ssim(img_a, img_b, window, self.window_size, channel, self.size_average)

        return 1 - ssim_v


class LossAdaptor(object):
    """
    An adaptor aim to balance loss via the std of the loss
    """

    def __init__(self, queue_size=100, param_only=True):
        self.size = queue_size
        self.history = Queue(maxsize=self.size)
        self.param_only = param_only

    def __call__(self, loss_var):
        if self.history.qsize() < self.size:
            param = 1.
            self.history.put(loss_var)
        else:
            self.history.put(loss_var)
            param = np.mean(self.history.queue)

        if self.param_only:
            return param
        else:
            return param * loss_var


class RetinaLoss(nn.Module):
    def __init__(self):

        super(RetinaLoss, self).__init__()
        self.l1_diff = nn.L1Loss()
        self.sigmoid = nn.Sigmoid()
        self.downsample = nn.AvgPool2d(2)
        self.level = 3
        self.eps = 1e-6
        pass

    @staticmethod
    def compute_gradient(img):
        grad_x = img[:, :, 1:, :] - img[:, :, :-1, :]
        grad_y = img[:, :, :, 1:] - img[:, :, :, :-1]
        return grad_x, grad_y

    def compute_exclusion_loss(self, img1, img2):
        """
        NOTE: To make image1 and image2 look different in retina way, TODO: need to be debug in detail, bad-ass
        :param img1:
        :param img2:
        :return:
        """
        gradx_loss = []
        grady_loss = []

        for l in range(self.level):
            gradx1, grady1 = self.compute_gradient(img1)
            gradx2, grady2 = self.compute_gradient(img2)

            if torch.mean(torch.abs(gradx2)) < self.eps or torch.mean(torch.abs(gradx2)) < self.eps:
                gradx_loss.append(0)
                grady_loss.append(0)
                continue

            alphax = 2.0 * torch.mean(torch.abs(gradx1)) / (torch.mean(torch.abs(gradx2)) + self.eps)
            alphay = 2.0 * torch.mean(torch.abs(grady1)) / (torch.mean(torch.abs(grady2)) + self.eps)

            if torch.isnan(alphax) or torch.isnan(alphay):
                gradx_loss.append(0)
                grady_loss.append(0)
                continue

            gradx1_s = (self.sigmoid(gradx1) * 2) - 1
            grady1_s = (self.sigmoid(grady1) * 2) - 1
            gradx2_s = (self.sigmoid(gradx2 * alphax) * 2) - 1
            grady2_s = (self.sigmoid(grady2 * alphay) * 2) - 1

            gradx_loss.append(torch.mean(torch.mul(torch.pow(gradx1_s, 2), torch.pow(gradx2_s, 2)) ** 0.25))
            grady_loss.append(torch.mean(torch.mul(torch.pow(grady1_s, 2), torch.pow(grady2_s, 2)) ** 0.25))

            img1 = self.downsample(img1)
            img2 = self.downsample(img2)

        loss = 0.5 * (sum(gradx_loss) / float(len(gradx_loss)) + sum(grady_loss) / float(len(grady_loss)))

        print(loss)

        return loss

    def compute_gradient_loss(self, img1, img2):
        """
        NOTE: To make image1 and image2 look same in retina way
        :param img1:
        :param img2:
        :return:
        """
        losses = []
        for l in range(self.level):
            gradx1, grady1 = self.compute_gradient(img1)
            gradx2, grady2 = self.compute_gradient(img2)

            loss = 0.5 * (self.l1_diff(gradx1, gradx2) + self.l1_diff(grady1, grady2))
            losses.append(loss)

        loss = 0 if len(losses) == 0 else sum(losses) / len(losses)
        return loss

    def forward(self, img_b, img_r, mode='exclusion'):
        """  Mode in [exclusion/gradient] """
        with torch.no_grad():
            if mode == 'exclusion':
                loss = self.compute_exclusion_loss(img_b, img_r)
            elif mode == 'gradient':
                loss = self.compute_gradient_loss(img_b, img_r)
            else:
                raise NotImplementedError("mode should in [exclusion/gradient]")
        return loss


# Defines the GAN loss which uses either LSGAN or the regular GAN.
# When LSGAN is used, it is basically same as MSELoss,
# but it abstracts away the need to create the target label tensor
# that has the same size as the input
class GANLoss(nn.Module):
    def __init__(self, use_lsgan=True, target_real_label=1.0, target_fake_label=0.0,
                 tensor=torch.FloatTensor):
        super(GANLoss, self).__init__()
        self.real_label = target_real_label
        self.fake_label = target_fake_label
        self.real_label_var = None
        self.fake_label_var = None
        self.Tensor = tensor
        if use_lsgan:
            self.loss = nn.MSELoss()
        else:
            self.loss = nn.BCELoss()

    def get_target_tensor(self, input, target_is_real):
        target_tensor = None
        if target_is_real:
            create_label = ((self.real_label_var is None) or
                            (self.real_label_var.numel() != input.numel()))
            if create_label:
                real_tensor = self.Tensor(input.size()).fill_(self.real_label)
                self.real_label_var = Variable(real_tensor, requires_grad=False)
            target_tensor = self.real_label_var
        else:
            create_label = ((self.fake_label_var is None) or
                            (self.fake_label_var.numel() != input.numel()))
            if create_label:
                fake_tensor = self.Tensor(input.size()).fill_(self.fake_label)
                self.fake_label_var = Variable(fake_tensor, requires_grad=False)
            target_tensor = self.fake_label_var
        return target_tensor

    def forward(self, input, target_is_real):
        target_tensor = self.get_target_tensor(input, target_is_real)
        return self.loss(input, target_tensor)


class VggLoss(nn.Module):
    def __init__(self, pretrained):
        super(VggLoss, self).__init__()
        assert type(pretrained) is Vgg11EncoderMS or type(pretrained) is Vgg19EncoderMS
        self.feature_extrator = pretrained
        self.l1_loss = nn.L1Loss()

    def forward(self, input, output):
        with torch.no_grad():
            vgg_real = self.feature_extrator(input)
            vgg_fake = self.feature_extrator(output)

            p0 = self.l1_loss(vgg_real['input'], vgg_fake['input'])
            p1 = self.l1_loss(vgg_real['shallow'], vgg_fake['shallow']) / 2.6
            p2 = self.l1_loss(vgg_real['low'], vgg_fake['low']) / 4.8
            p3 = self.l1_loss(vgg_real['mid'], vgg_fake['mid']) / 3.7
            p4 = self.l1_loss(vgg_real['deep'], vgg_fake['deep']) / 5.6
            p5 = self.l1_loss(vgg_real['out'], vgg_fake['out']) * 10 / 1.5

        return p0 + p1 + p2 + p3 + p4 + p5


##################################################################################
# Test codes
##################################################################################

def test_gen():
    pass


def test_retina_loss():
    import cv2
    img_1 = cv2.imread('/media/ros/Files/ws/Dataset/Reflection/Berkeley/synthetic/transmission_layer/53.jpg')
    img_2 = cv2.imread('/media/ros/Files/ws/Dataset/Reflection/Berkeley/synthetic/reflection_layer/21.jpg')

    img_1 = cv2.resize(img_1, (300, 300))
    img_2 = cv2.resize(img_2, (300, 300))

    img_1 = np.float32(img_1)
    img_2 = np.float32(img_2)

    img_1 = np.transpose(img_1, (2, 0, 1))
    img_2 = np.transpose(img_2, (2, 0, 1))

    v_1 = torch.from_numpy(img_1).unsqueeze(0)
    v_2 = torch.from_numpy(img_2).unsqueeze(0)

    retina = RetinaLoss()

    l1 = retina.compute_exclusion_loss(v_1, v_2)
    l1_1 = retina.compute_exclusion_loss(v_1, v_1)
    l1_2 = retina.compute_exclusion_loss(v_2, v_2)

    l2 = retina.compute_gradient_loss(v_1, v_2)
    l2_1 = retina.compute_gradient_loss(v_1, v_1)
    l2_2 = retina.compute_gradient_loss(v_2, v_2)

    print(l1.item())
    print(l1_1.item())
    print(l1_2.item())

    print(l2.item())
    print(l2_1.item())
    print(l2_2.item())


if __name__ == '__main__':
    test_retina_loss()
