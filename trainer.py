"""
Trainer for Hyper Spectral Image Super-resolution, KD based
"""
from networks import get_student, get_teacher, RetinaLoss, VggLoss
from utils import weights_init, get_scheduler, get_model_list
from torch.autograd import Variable
import torch
import torch.nn as nn
import os


class Trainer(nn.Module):
    def __init__(self, param):
        super(Trainer, self).__init__()
        lr = param['lr']
        # Initiate the networks
        self.student = get_student(param)
        self.teacher = get_teacher(param)

        # Setup the optimizers
        beta1 = param['beta1']
        beta2 = param['beta2']
        teacher_params = list(self.discriminator.parameters())
        student_params = list(self.generator.parameters())

        self.stu_opt = torch.optim.Adam(student_params,
                                        lr=lr, betas=(beta1, beta2), weight_decay=param['weight_decay'])
        self.stu_scheduler = get_scheduler(self.stu_opt, param)
        # self.dis_scheduler = None
        # self.gen_scheduler = None

        # Network weight initialization
        # self.apply(weights_init(param['init']))
        self.best_result = float('inf')

        self.criteria = RetinaLoss()

    def forward(self, x_i):
        self.eval()
        y_stu = self.student(x_i)
        y_tea = self.teacher(x_i)
        self.train()
        return y_stu, y_tea

    # noinspection PyAttributeOutsideInit
    def gen_update(self, x_in, y_out, param):
        self.stu_opt.zero_grad()

        stu_pred = self.student(x_in)
        tea_pred = self.teacher(x_in)

        # loss constraints
        self.loss_total = self.criteria(stu_pred, tea_pred, y_out, param)

        self.loss_total.backward()
        self.stu_opt.step()

    def sample(self, x_in, y_out):
        self.eval()
        stu_pred = []
        tea_pred = []
        for i in range(x_in.size(0)):
            _pred_stu = self.student(x_in[i].unsqueeze(0))['out']
            _pred_tea = self.teacher(x_in[i].unsqueeze(0))['out']

            stu_pred.append(_pred_stu)
            tea_pred.append(_pred_tea)
        pred_stu = torch.cat(stu_pred)
        pred_tea = torch.cat(tea_pred)
        self.train()
        return x_in, y_out, pred_stu, pred_tea

    def update_learning_rate(self):
        if self.stu_scheduler is not None:
            self.stu_scheduler.step()

    def resume(self, checkpoint_dir, param):
        # Load generators
        last_model_name = get_model_list(checkpoint_dir, "student")
        state_dict = torch.load(last_model_name)
        self.generator.load_state_dict(state_dict['student'])
        self.best_result = state_dict['best_result']
        epoch = int(last_model_name[-6: -3])
        # Load discriminators
        last_model_name = get_model_list(checkpoint_dir, "teacher")
        state_dict = torch.load(last_model_name)
        self.discriminator.load_state_dict(state_dict['teacher'])
        # Load optimizers
        state_dict = torch.load(os.path.join(checkpoint_dir, 'optimizer.pt'))
        self.dis_opt.load_state_dict(state_dict['teacher'])
        self.stu_opt.load_state_dict(state_dict['student'])
        # Re-initialize schedulers
        try:
            self.stu_scheduler = get_scheduler(self.stu_opt, param, epoch)
        except Exception as e:
            print('Warning: {}'.format(e))
        print('Resume from epoch %d' % epoch)
        return epoch

    def save(self, snapshot_dir, iterations):
        # Save generators, discriminators, and optimizers
        gen_name = os.path.join(snapshot_dir, 'student_%03d.pt' % (iterations + 1))
        dis_name = os.path.join(snapshot_dir, 'teacher.pt')
        opt_name = os.path.join(snapshot_dir, 'optimizer.pt')
        torch.save({'student': self.generator.state_dict(), 'best_result': self.best_result}, gen_name)
        torch.save({'teacher': self.discriminator.state_dict()}, dis_name)
        torch.save({'student': self.stu_opt.state_dict(), 'teacher': self.dis_opt.state_dict()}, opt_name)
