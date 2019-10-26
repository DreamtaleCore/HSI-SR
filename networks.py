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

from utils.utils import initialize_weights

##################################################################################
# Interfaces
##################################################################################

def get_discriminator(dis_opt, train_mode=None):
    """Get a discriminator"""
    # multi-scale dis
    return Discriminator(dis_opt)

def get_teacher(tea_opt, train_mode=None):
    """Get a teacher"""
    return SRResNet(tea_opt)

def get_student(stu_opt, train_mode=None):
    """Get a student"""
    return SRResNet(stu_opt)


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

    def _make_layer(block, n_layers):
        layers = []
        for _ in range(n_layers):
            layers.append(block())
        return nn.Sequential(*layers)

    def forward(self, x):
        fea = self.lrelu(self.conv_first(x))
        out = self.recon_trunk(fea)

        if self.upscale == 4:
            out = self.lrelu(self.pixel_shuffle(self.upconv1(out)))
            out = self.lrelu(self.pixel_shuffle(self.upconv2(out)))
        elif self.upscale == 3 or self.upscale == 2:
            out = self.lrelu(self.pixel_shuffle(self.upconv1(out)))

        out = self.conv_last(self.lrelu(self.HRconv(out)))
        base = F.interpolate(x, scale_factor=self.upscale, mode='bilinear', align_corners=False)
        out += base
        out_feature = {
            'out': out
        }
        return out_feature


##################################################################################
# Discriminator
##################################################################################

class Discriminator(nn.Module):
    """Discriminator, Auxiliary Classifier."""
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
