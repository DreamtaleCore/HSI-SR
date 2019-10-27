from __future__ import print_function

from tqdm import tqdm

from utils import get_config, evaluation_matrix, save_mat
from trainer import Trainer
import argparse
from torch.autograd import Variable
import torchvision.utils as vutils
import numpy as np
from data import read_mat as image_loader
import torch
import os
from torchvision import transforms
from PIL import Image

parser = argparse.ArgumentParser()
parser.add_argument('--configs', type=str, default='configs/teacher-student.yaml', help="net configuration")
parser.add_argument('--input_dir', type=str, default='E:/MMSR/datasets/train_data/mat_bicLRx4',
                    help="input image path")
parser.add_argument('--gt_dir', type=str, default='E:/MMSR/datasets/train_data/mat',
                    help="ground truth image path, images need have the same name")
parser.add_argument('--output_folder', type=str, default='id_mit_train-inner-opt',
                    help="output image path")
parser.add_argument('--checkpoint', type=str, default='checkpoints/mit_inner-opt/gen_00440000.pt',
                    help="checkpoint of pre-trained model")
parser.add_argument('--save_top_k', type=int, default=3,
                    help="save the top [k] best results")
opts = parser.parse_args()

IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG', '.bmp', '.BMP',
    '.png', '.PNG', '.ppm', '.PPM', '.mat', '.MAT'
]


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)


if not os.path.exists(opts.output_folder):
    os.makedirs(opts.output_folder)

# Load experiment setting
config = get_config(opts.config)

# Setup model and data loader
# configs['vgg_model_path'] = opts.output_path
# Setup model and data loader
trainer = Trainer(config)

state_dict = torch.load(opts.checkpoint, map_location='cuda:0')
trainer.student.load_state_dict(state_dict['student'])
trainer.teacher.load_state_dict(state_dict['teacher'])

trainer.cuda()
trainer.eval()

if 'new_size' in config:
    new_size = config['new_size']
else:
    new_size = config['new_size_i']

SI_SR = trainer.forward

test_loader = []
if os.path.isdir(opts.input):
    image_list = os.listdir(opts.input)
    image_list = [x for x in image_list if is_image_file(x)]
else:
    with open(os.path.join(opts.input), 'r') as fid:
        lines = fid.readlines()
        image_list = [x.strip() for x in lines]
for img_name in image_list:
    name = os.path.basename(img_name)
    if os.path.isfile(opts.input):
        dir_input = os.path.dirname(opts.input)
    else:
        dir_input = opts.input
    tmp_pwd = os.path.join(dir_input, img_name)
    image_tf = tmp_pwd
    test_loader.append((image_tf, {'name': [name]}))

out_root = opts.output_folder
if not os.path.exists(out_root):
    os.makedirs(out_root)


def gather_results(result_dict):
    """
    Convert the result to text
    :param result_dict:
    :return:
    """
    lines = []
    lines.append(','.join([str(x) for x in result_dict.keys()]) + '\n')
    for items in zip(result_dict.values()):
        lines.append(','.join([str(x) for x in items]) + '\n')
    return lines


with torch.no_grad():
    stu_results = {}
    tea_results = {}
    t_bar = tqdm(test_loader)
    t_bar.set_description('Processing')
    for (image_in, targets) in t_bar:
        t_bar = tqdm(test_loader)
        img_name = targets['name'][0]
        if 'name' not in stu_results:
            stu_results['name'] = []
        if 'name' not in tea_results:
            tea_results['name'] = []
        stu_results['name'].append(img_name)
        tea_results['name'].append(img_name)

        images_in = image_loader(path=image_in, phase=None, key='im_LR')
        images_gt = image_loader(path=image_in.replace(opts.input_dir, opts.gt_dir), phase=None, key='im_LR')

        images_in = torch.from_numpy(np.ascontiguousarray(np.transpose(images_in, (2, 0, 1)))).float()
        images_in = images_in.unsqueeze(0).cuda().detach()

        stu_pred, tea_pred = SI_SR(images_in)
        stu_pred = np.transpose(stu_pred.detach().cpu().squeeze().numpy(), (1, 2, 0))
        tea_pred = np.transpose(tea_pred.detach().cpu().squeeze().numpy(), (1, 2, 0))

        eval_stu = evaluation_matrix(stu_pred, images_gt)
        eval_tea = evaluation_matrix(tea_pred, images_gt)

        for k, v in eval_stu:
            if k not in stu_results:
                stu_results[k] = []
            stu_results[k].append(v)
        for k, v in eval_tea:
            if k not in tea_results:
                tea_results[k] = []
            tea_results[k].append(v)

    print('\n Compute Done.')
    print('Saving the log information...')
    stu_info = gather_results(stu_results)
    tea_info = gather_results(tea_results)

    fid_stu = open(os.path.join(out_root, 'result_student.csv'), 'w')
    fid_tea = open(os.path.join(out_root, 'result_teacher.csv'), 'w')
    fid_stu.writelines(stu_info)
    fid_tea.writelines(tea_info)
    fid_stu.close()
    fid_tea.close()

    print('Saving the top-{} best results for student.'.format(opts.save_top_k))
    eval_judgement = stu_results[list(stu_results.keys())[0]]
    idxs = np.argsort(eval_judgement)
    # if find the top-k minimal values, change `idxs[:opts.save_top_k]` to `idxs[-opts.save_top_k:]`
    saving_names = stu_results['name'][idxs[:opts.save_top_k]]
    t_bar = tqdm(saving_names)
    t_bar.set_description('Saving')

    for img_name in t_bar:
        images_in = image_loader(path=image_in, phase=None, key='im_LR')
        images_gt = image_loader(path=image_in.replace(opts.input_dir, opts.gt_dir), phase=None, key='im_LR')

        images_in = torch.from_numpy(np.ascontiguousarray(np.transpose(images_in, (2, 0, 1)))).float()
        images_in = images_in.unsqueeze(0).cuda().detach()

        stu_pred, tea_pred = SI_SR(images_in)
        stu_pred = np.transpose(stu_pred.detach().cpu().squeeze().numpy(), (1, 2, 0))
        tea_pred = np.transpose(tea_pred.detach().cpu().squeeze().numpy(), (1, 2, 0))

        image_base_name, post_fix = os.path.basename(img_name).split('.')[-2:]
        stu_name = '{}-{}.{}'.format(image_base_name, 'student', post_fix)
        tea_name = '{}-{}.{}'.format(image_base_name, 'teacher', post_fix)
        gt_name  = '{}-{}.{}'.format(image_base_name, 'student', post_fix)

        save_mat(os.path.join(out_root, stu_name), stu_pred)
        save_mat(os.path.join(out_root, stu_name), tea_pred)
        save_mat(os.path.join(out_root, gt_name), images_gt)

    print('\nAll Done.')

