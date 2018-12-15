import argparse
import os

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler

from transforms import transforms_factory
from datasets import dataset_factory
from nets import nets_factory
from dataloaders import dataloader_factory
from trainings import training_factory


def parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('job_name', type=str)
    parser.add_argument('--logdir', type=str, default='/home/antonio/test_log')
    parser.add_argument('--dataset_name', type=str, default='cubs')
    parser.add_argument('--dataset_dir', type=str, default='/home/antonio/Data/Datasets/cubs_cropped/')
    parser.add_argument('--dataloader', type=str, default='standard_dataloader')
    parser.add_argument('--model_name', type=str, default='resnet18_multisource')
    parser.add_argument('--pretrained', type=bool, default=True, help='Initialize from a pretrained model')
    parser.add_argument('--side_branches', type=str,
                        default='/home/antonio/logs-mitl/finetune_resnet18_cubs_1_cubs_resnet18_multisource_30_0.010000/cubs_resnet18_multisource_60_0.010000.pth')
    parser.add_argument('--transforms_name', type=str, default='imagenet')
    parser.add_argument('--training_fn', type=str, default='standard_training')
    parser.add_argument('--im_size', type=int, default=224)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--learning_rate', type=float, default=0.01)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--weight_decay', type=float, default=0.0005)
    parser.add_argument('--num_epochs', type=int, default=30)
    parser.add_argument('--step_size', type=int, default=10)
    parser.add_argument('--gamma', type=float, default=0.1)
    parser.add_argument('--num_workers', type=int, default=4)
    args = parser.parse_args()
    return args


def main():

    args = parse()
    print(args)

    ####################
    # Image Transforms #
    ####################

    transforms_name = args.transforms_name or args.model_name
    get_transforms_fn = transforms_factory.get_transforms(transforms_name)

    data_transforms = get_transforms_fn(args.im_size)

    ##################
    # Select dataset #
    ##################

    dataset_name = args.dataset_name
    get_dataset_fn = dataset_factory.get_dataset(dataset_name)

    image_datasets = get_dataset_fn(args.dataset_dir, data_transforms=data_transforms)

    #######################
    # Prepare dataloaders #
    #######################

    dataloader = args.dataloader
    get_dataloader_fn = dataloader_factory.get_dataloader(dataloader)

    dataloaders = get_dataloader_fn(image_datasets, args.batch_size, args.num_workers)

    ##################
    # Select network #
    ##################

    model_name = args.model_name
    get_network_fn = nets_factory.get_network(model_name)

    num_classes = len(image_datasets['train'].classes) #UNSAFE
    nets = []
    for branch in args.side_branches.split(','):
        net = torch.load(branch)
        for param in net.parameters():
            param.requires_grad = False
        nets.append(net)
    model = get_network_fn(num_classes, pretrained=args.pretrained)

    ####################
    # Prepare training #
    ####################

    use_gpu = torch.cuda.is_available()
    if use_gpu:
        model = model.cuda()

    criterion = nn.CrossEntropyLoss()
    params_to_optimize = model.parameters()
    optimizer = optim.SGD(params_to_optimize, lr=args.learning_rate, momentum=args.momentum, weight_decay=args.weight_decay)

    exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)

    training_fn = args.training_fn
    get_training_fn = training_factory.get_training(training_fn)

    log_path = '%s/%s_%s_%s_%d_%f' % (args.logdir, args.job_name, args.dataset_name,
                                      args.model_name, args.num_epochs, args.learning_rate)
    training = get_training_fn(model, logger_path=log_path, use_gpu=use_gpu)

    #########
    # Train #
    #########

    training.train_model(dataloaders, criterion, optimizer, exp_lr_scheduler, args.num_epochs)

    training.test_model(dataloaders['test'])

    ########
    # Save #
    ########

    training.save_model('%s/%s_%s_%d_%f' % (log_path, args.dataset_name,
                                            args.model_name, args.num_epochs, args.learning_rate))


if __name__ == '__main__':
    main()
