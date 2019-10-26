import argparse
import os
import shutil
import cv2
import numpy as np
import tensorboardX
import torch
import torch.backends.cudnn as cudnn
import tqdm
from torchvision import transforms

from trainer import Trainer
from utils import get_local_time
from utils import get_reflection_data_loader, prepare_sub_folder, write_html, \
    write_loss, get_config, write_2images, to_number

parser = argparse.ArgumentParser()
parser.add_argument('--config', type=str, default='configs/teacher-student.yaml', help='Path to the configs file.')
parser.add_argument('--output_path', type=str, default='checkpoints-tmp', help="outputs path")
parser.add_argument("--resume", action="store_true", default=False)
parser.add_argument("--force_save", action="store_true", default=False)
parser.add_argument('--gpu_id', type=int, default=0, help="gpu id")
opts = parser.parse_args()

cudnn.benchmark = True

# Load experiment setting
config = get_config(opts.config)
display_size = config['display_size']

torch.cuda.set_device(opts.gpu_id)

# Setup model and data loader
trainer = Trainer(config)

trainer.cuda()
train_loader, eval_loader = get_reflection_data_loader(config)

# Setup logger and output folders
model_name = os.path.splitext(os.path.basename(opts.config))[0]
train_writer = tensorboardX.SummaryWriter(os.path.join(opts.output_path + "/logs", model_name))
output_directory = os.path.join(opts.output_path + "/outputs", model_name)
checkpoint_directory, image_directory = prepare_sub_folder(output_directory)
shutil.copy(opts.config, os.path.join(output_directory, 'configs.yaml'))  # copy configs file to output folder

# Start training
start_epoch = trainer.resume(checkpoint_directory, param=config) if opts.resume else 0

print('Start from epoch {}, total epochs {}. {} images for each epoch.'.format(start_epoch, config['n_epoch'],
                                                                               len(train_loader)))

to_pil = transforms.ToPILImage()

iterations = start_epoch * len(train_loader)

for epoch in range(start_epoch, config['n_epoch']):
    for it, (img_in, targets) in enumerate(train_loader):
        trainer.update_learning_rate()

        images_in, images_out = img_in.cuda().detach(), \
                                targets['output'].cuda().detach()

        # img_in = images_in[0].cpu()
        # img_out = images_out[0].cpu()
        #
        # img_in = to_pil(img_i)
        # img_out = to_pil(img_out)
        #
        # cv2.imshow('image in', np.asarray(img_in)[:, :, ::-1])
        # cv2.imshow('image out', np.asarray(img_out)[:, :, ::-1])
        # cv2.waitKey()
        #
        # continue

        trainer.dis_update(images_in, images_out, config)
        trainer.gen_update(images_in, images_out, config)

        # Dump training stats in log file
        if (iterations + 1) % config['log_iter'] == 0:
            print('<{}> [Epoch: {}] [Iter: {}/{}] | Loss: {}'.format(get_local_time(), epoch, it, len(train_loader),
                                                                     to_number(trainer.loss_total)))
            write_loss(iterations, trainer, train_writer)

        # Write images
        if (iterations + 1) % config['image_save_iter'] == 0:
            with torch.no_grad():
                outputs = trainer.sample(images_in, images_out)
            write_2images(outputs, display_size, image_directory, 'train_%08d' % (iterations + 1))
            # HTML
            write_html(output_directory + "/index.html", iterations + 1, config['image_save_iter'], 'images')

        iterations += 1

    if epoch % config['snapshot_save_epoch'] == 0:
        current_loss = trainer.loss_total
        print('=' * 40)
        print('<{}> The model, the current training loss is {}'.format(get_local_time(), current_loss))
        print('=' * 40)

        if epoch % config['eval_interval'] == 0:
            t_bar = tqdm.tqdm(eval_loader)
            t_bar.set_description('Eval')
            l1_losses = []
            for img_in, targets in t_bar:
                images_in, images_out = img_in.cuda().detach(), \
                                        targets['output'].cuda().detach()
                x_in, y_gt, y_pred = trainer.sample(images_in, images_out)

                loss = trainer.recon_criterion(y_pred, y_gt).item()

                t_bar.set_description('Eval - L1: {}'.format(loss))
                l1_losses.append(loss)

            avg_loss = np.mean(l1_losses)
            if avg_loss < trainer.best_result or opts.force_save:
                trainer.best_result = avg_loss
                print('=' * 40)
                print('<{}> Save the model, the best loss is {}'.format(get_local_time(), avg_loss))
                print('=' * 40)
                trainer.save(checkpoint_directory, epoch)
            else:
                print('\n<{}> The current loss is {}'.format(get_local_time(), avg_loss))
