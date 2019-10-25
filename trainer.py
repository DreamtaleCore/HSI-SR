from networks import get_generator, get_discriminator, RetinaLoss, VggLoss
from utils import weights_init, get_scheduler, get_model_list
from torch.autograd import Variable
import torch
import torch.nn as nn
import os


class Trainer(nn.Module):
    def __init__(self, param):
        super(Trainer, self).__init__()
        lr_g = param['lr_g']
        lr_d = param['lr_d']
        # Initiate the networks
        self.generator = get_generator(param)
        self.discriminator = get_discriminator(param)

        # Setup the optimizers
        beta1 = param['beta1']
        beta2 = param['beta2']
        dis_params = list(self.discriminator.parameters())
        gen_params = list(self.generator.parameters())
        self.dis_opt = torch.optim.Adam(dis_params,
                                        lr=lr_d, betas=(beta1, beta2), weight_decay=param['weight_decay'])
        self.gen_opt = torch.optim.Adam(gen_params,
                                        lr=lr_g, betas=(beta1, beta2), weight_decay=param['weight_decay'])
        self.dis_scheduler = get_scheduler(self.dis_opt, param)
        self.gen_scheduler = get_scheduler(self.gen_opt, param)
        # self.dis_scheduler = None
        # self.gen_scheduler = None

        # Network weight initialization
        # self.apply(weights_init(param['init']))
        self.discriminator.apply(weights_init('gaussian'))
        self.best_result = float('inf')

        # Load VGG model if needed
        if param['vgg_w'] != 0:
            self.vgg_loss = VggLoss(self.generator.encoder)
        self.retina_loss = RetinaLoss()

    def recon_criterion(self, input, target):
        return torch.mean(torch.abs(input - target))

    def forward(self, x_i):
        self.eval()
        y_i = self.generator(x_i)
        self.train()
        return y_i

    # noinspection PyAttributeOutsideInit
    def gen_update(self, x_in, y_out, param):
        self.gen_opt.zero_grad()

        y_pred = self.generator(x_in)[0]

        # loss constraints
        self.loss_vgg = self.vgg_loss(y_pred, y_out) if param['vgg_w'] != 0 else 0
        self.loss_pixel = self.recon_criterion(y_pred, y_out) if param['pixel_w'] != 0 else 0
        self.loss_retina = self.retina_loss(y_pred, y_out,
                                            'gradient') if param['retina_w'] != 0 else 0
        self.loss_gen = self.discriminator.calc_gen_loss(y_pred) if param['gan_w'] != 0 else 0

        # total loss
        self.loss_total = param['vgg_w'] * self.loss_vgg + \
                          param['pixel_w'] * self.loss_pixel + \
                          param['retina_w'] + self.loss_retina + \
                          param['gan_w'] + self.loss_gen

        self.loss_total.backward()
        self.gen_opt.step()

    def sample(self, x_in, y_out):
        self.eval()
        xs_pred = []
        for i in range(x_in.size(0)):
            _pred = self.generator(x_in[i].unsqueeze(0))[0]

            xs_pred.append(_pred)
        preds = torch.cat(xs_pred)
        self.train()
        return x_in, y_out, preds

    # noinspection PyAttributeOutsideInit
    def dis_update(self, x_in, y_pred, param):
        self.dis_opt.zero_grad()
        pred = self.generator(x_in)[0]
        # D loss
        if param['gan_w'] != 0:
            self.loss_dis = self.discriminator.calc_dis_loss(pred.detach(), y_pred)
            self.loss_dis.backward()
            self.dis_opt.step()
        else:
            self.loss_dis = 0

    def update_learning_rate(self):
        if self.dis_scheduler is not None:
            self.dis_scheduler.step()
        if self.gen_scheduler is not None:
            self.gen_scheduler.step()

    def resume(self, checkpoint_dir, param):
        # Load generators
        last_model_name = get_model_list(checkpoint_dir, "gen")
        state_dict = torch.load(last_model_name)
        self.generator.load_state_dict(state_dict['generator'])
        self.best_result = state_dict['best_result']
        epoch = int(last_model_name[-6: -3])
        # Load discriminators
        last_model_name = get_model_list(checkpoint_dir, "dis")
        state_dict = torch.load(last_model_name)
        self.discriminator.load_state_dict(state_dict['discriminator'])
        # Load optimizers
        state_dict = torch.load(os.path.join(checkpoint_dir, 'optimizer.pt'))
        self.dis_opt.load_state_dict(state_dict['dis'])
        self.gen_opt.load_state_dict(state_dict['gen'])
        # Re-initialize schedulers
        try:
            self.dis_scheduler = get_scheduler(self.dis_opt, param, epoch)
            self.gen_scheduler = get_scheduler(self.gen_opt, param, epoch)
        except Exception as e:
            print('Warning: {}'.format(e))
        print('Resume from epoch %d' % epoch)
        return epoch

    def save(self, snapshot_dir, iterations):
        # Save generators, discriminators, and optimizers
        gen_name = os.path.join(snapshot_dir, 'gen_%03d.pt' % (iterations + 1))
        dis_name = os.path.join(snapshot_dir, 'dis_%03d.pt' % (iterations + 1))
        opt_name = os.path.join(snapshot_dir, 'optimizer.pt')
        torch.save({'generator': self.generator.state_dict(), 'best_result': self.best_result}, gen_name)
        torch.save({'discriminator': self.discriminator.state_dict()}, dis_name)
        torch.save({'gen': self.gen_opt.state_dict(), 'dis': self.dis_opt.state_dict()}, opt_name)
