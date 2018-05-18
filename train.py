from __future__ import print_function

import argparse
import os

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torchvision
from torchvision import datasets, models, transforms
import cPickle as pkl

import nets

from training_helper import TorchTraining

parser = argparse.ArgumentParser()
parser.add_argument('job_name', type=str)
parser.add_argument('data_dir', type=str, help='Dataset root directory')
parser.add_argument('--num_classes', type=int, default=1000)
parser.add_argument('--logdir', type=str, default='./logs/logs', help='Dir for tensorboard event logs')
parser.add_argument('--arch', type=str, default='resnet18_multisource', help='Architecture to use during training')
parser.add_argument('--mask', type=str, default=None, help='name of the masking function to use')
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--num_epochs', type=int, default=30)
parser.add_argument('--lr', type=float, default=0.001)
parser.add_argument('--momentum', type=float, default=0.9)
parser.add_argument('--step_size', type=int, default=10)
parser.add_argument('--gamma', type=float, default=0.2, help='lr stepdown coefficient')
parser.add_argument('--num_workers', type=int, default=4)
args = parser.parse_args()

print(args)


#######################
# Image preprocessing #
#######################

data_transforms = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    'test': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])}


################
# Load dataset #
################

image_datasets = {x: datasets.ImageFolder(os.path.join(args.data_dir, x),
                                          data_transforms[t])
                  for x, t in zip(['train', 'test'],['train', 'test'])}
dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x],
                                              batch_size=args.batch_size,
                                              shuffle=x=='train',
                                              num_workers=args.num_workers)
               for x in (['train', 'test'])}
dataset_sizes = {x: len(image_datasets[x])for x in ['train', 'test']}
class_names = image_datasets['train'].classes


###########################
# Load pre-trained models #
###########################

p_nets = []
'''
TODO: Qui si potranno caricare i modelli finetunati precedentemente
TODO: parameter requires grad <- False
'''


################
# Create model #
################

print('Loading architecture')
masking_fn = nets.__dict__[args.mask] if args.mask is not None else lambda x: x
model = nets.__dict__[args.arch](p_nets, args.num_classes, masking_fn)


####################
# Prepare training #
####################

use_gpu = torch.cuda.is_available()
if use_gpu:
    model = model.cuda()

criterion = nn.CrossEntropyLoss()

params_to_optimize = [param for param in model.parameters() if param.requires_grad]
optimizer_conv = optim.SGD(params_to_optimize, lr=args.lr, momentum=args.momentum)

exp_lr_scheduler = lr_scheduler.StepLR(optimizer_conv, step_size=args.step_size, gamma=args.gamma)

tt = TorchTraining(idra_model, logger_path=args.logdir)


##################
# Start training #
##################

model = tt.train_model(dataloaders, criterion, optimizer_conv, exp_lr_scheduler, num_epochs=args.num_epochs)


##########
# Saving #
##########

'''
Todo
'''
