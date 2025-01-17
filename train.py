# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from pathlib import Path
import argparse
import json
import math
import os
import random
import signal
import subprocess
import sys
import time

from PIL import Image, ImageOps, ImageFilter
from torch import nn, optim
import torch
import torchvision
import torchvision.transforms as transforms
# from axialnet import ResAxialAttentionUNet, AxialBlock
import wandb
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2
import matplotlib.pyplot as plt
import cv2

# from unet import UNet


parser = argparse.ArgumentParser(description='Barlow Twins Training')
parser.add_argument('--data', default='train_dataset/img', type=Path, metavar='DIR',
                    help='path to dataset')
parser.add_argument('--workers', default=8, type=int, metavar='N',
                    help='number of data loader workers')
parser.add_argument('--epochs', default=1000, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--batch-size', default=256, type=int, metavar='N',
                    help='mini-batch size')
parser.add_argument('--learning-rate-weights', default=0.002, type=float, metavar='LR',
                    help='base learning rate for weights')
parser.add_argument('--learning-rate-biases', default=0.000048, type=float, metavar='LR',
                    help='base learning rate for biases and batch norm parameters')
parser.add_argument('--weight-decay', default=1e-6, type=float, metavar='W',
                    help='weight decay')
parser.add_argument('--lambd', default=0.0051, type=float, metavar='L',
                    help='weight on off-diagonal terms')
parser.add_argument('--projector', default='8192-8192-8192', type=str,
                    metavar='MLP', help='projector MLP')
parser.add_argument('--print-freq', default=100, type=int, metavar='N',
                    help='print frequency')
parser.add_argument('--pretrained', default=False, type=bool, metavar='N',
                    help='pretrain with imagenet')
parser.add_argument('--checkpoint-path-load', default=None, type=Path,
                    metavar='DIR', help='path to checkpoint file to load from')
parser.add_argument('--checkpoint-path-save', default='./checkpoint/checkpoint.pth', type=Path,
                    metavar='DIR', help='path to checkpoint file to save to')

wandb.login(key='ed94033c9c3bebedd51d8c7e1daf4c6eafe44e09')
wandb.init(project='barlow-twins', entity='sborar')
config = wandb.config


def save_images(y1):
    org = y1[0].permute(1, 2, 0).numpy()
    norm = (org - np.min(org)) / (np.max(org) - np.min(org))
    plt.imsave('y1.png', norm)


def main():
    args = parser.parse_args()
    #
    # writer = SummaryWriter()
    args.rank = 0
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print(args)

    model = BarlowTwins(args)
    model.to(device)
    param_weights = []
    param_biases = []
    for param in model.parameters():
        if param.ndim == 1:
            param_biases.append(param)
        else:
            param_weights.append(param)
    parameters = [{'params': param_weights}, {'params': param_biases}]
    optimizer = LARS(parameters, lr=0, weight_decay=args.weight_decay,
                     weight_decay_filter=exclude_bias_and_norm,
                     lars_adaptation_filter=exclude_bias_and_norm)

    # automatically resume from checkpoint if it exists
    if args.checkpoint_path_load and (args.checkpoint_path_load).is_file():
        ckpt = torch.load(args.checkpoint_path_load,
                          map_location='cpu')
        start_epoch = ckpt['epoch']
        model.load_state_dict(ckpt['model'])
        optimizer.load_state_dict(ckpt['optimizer'])
    else:
        start_epoch = 0

    wandb.watch(model)

    dataset = torchvision.datasets.ImageFolder(args.data, Transform())
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=args.batch_size)

    start_time = time.time()
    # scaler = torch.cuda.amp.GradScaler()
    mean_x = []
    mean_y = []
    mean_z = []
    std_x = []
    std_y = []
    std_z = []
    for epoch in range(start_epoch, start_epoch + args.epochs):
        print('epoch', epoch)
        # sampler.set_epoch(epoch)
        for step, ((y1, y2), _) in enumerate(loader, start=epoch * len(loader)):
            y1 = y1.to(device)
            y2 = y2.to(device)
            adjust_learning_rate(args, optimizer, loader, step)
            optimizer.zero_grad()
            mean_x.append(y2[:, 0, :, :].mean().item())
            mean_y.append(y2[:, 1, :, :].mean().item())
            mean_z.append(y2[:, 2, :, :].mean().item())
            std_x.append(y2[:, 0, :, :].std().item())
            std_y.append(y2[:, 1, :, :].std().item())
            std_z.append(y2[:, 2, :, :].std().item())
            print('mean_x', np.mean(mean_x))
            print('mean_y', np.mean(mean_y))
            print('mean_z', np.mean(mean_z))
            print('std_x', np.std(std_x))
            print('std_y', np.std(std_y))
            print('std_z', np.std(std_z))

            # with torch.cuda.amp.autocast():
            # print('y',y1)
            # save_images(y1)
            loss = model.forward(y1, y2)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
            optimizer.step()
            #
            # for tag, parm in model.named_parameters():
            #     writer.add_histogram(tag, parm.grad.data.cpu().numpy(), epoch)

            if step % args.print_freq == 0:
                if args.rank == 0:
                    stats = dict(epoch=epoch, step=step,
                                 lr_weights=optimizer.param_groups[0]['lr'],
                                 lr_biases=optimizer.param_groups[1]['lr'],
                                 loss=loss.item(),
                                 time=int(time.time() - start_time))
                    print(json.dumps(stats))
                    # print(json.dumps(stats), file=stats_file)
                    wandb.log({"loss": loss.item()})
                    # writer.add_scalar('Loss/train', loss.item(), step)

        if args.rank == 0:
            # save checkpoint
            state = dict(epoch=epoch + 1, model=model.state_dict(),
                         optimizer=optimizer.state_dict())
            torch.save(state, args.checkpoint_path_save)
    if args.rank == 0:
        # save final model
        state = dict(epoch=epoch + 1, model=model.state_dict(),
                     optimizer=optimizer.state_dict())
        torch.save(state, args.checkpoint_path_save)
    # writer.close()


def adjust_learning_rate(args, optimizer, loader, step):
    max_steps = args.epochs * len(loader)
    warmup_steps = 10 * len(loader)
    base_lr = 8
    if step < warmup_steps:
        print('warmup')
        lr = base_lr * step / warmup_steps
    else:
        step -= warmup_steps
        max_steps -= warmup_steps
        q = 0.5 * (1 + math.cos(math.pi * step / max_steps))
        end_lr = base_lr * 0.001
        lr = base_lr * q + end_lr * (1 - q)
    print('lr:', lr)
    optimizer.param_groups[0]['lr'] = lr * args.learning_rate_weights
    optimizer.param_groups[1]['lr'] = lr * args.learning_rate_biases


def handle_sigusr1(signum, frame):
    os.system(f'scontrol requeue {os.getenv("SLURM_JOB_ID")}')
    exit()


def handle_sigterm(signum, frame):
    pass


def off_diagonal(x):
    # return a flattened view of the off-diagonal elements of a square matrix
    n, m = x.shape
    assert n == m
    return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()


class BarlowTwins(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        # self.backbone = ResAxialAttentionUNet(AxialBlock, [1, 2, 4, 1], s= 0.125)
        self.backbone = torchvision.models.resnet18(zero_init_residual=True, pretrained=args.pretrained)
        self.backbone.fc = nn.Identity()
        # self.backbone = UNet(3)

        # projector
        sizes = [512] + list(map(int, args.projector.split('-')))
        # sizes = [32768] + list(map(int, args.projector.split('-')))
        layers = []
        for i in range(len(sizes) - 2):
            layers.append(nn.Linear(sizes[i], sizes[i + 1], bias=False))
            layers.append(nn.BatchNorm1d(sizes[i + 1]))
            layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Linear(sizes[-2], sizes[-1], bias=False))
        self.projector = nn.Sequential(*layers)

        # normalization layer for the representations z1 and z2
        self.bn = nn.BatchNorm1d(sizes[-1], affine=False)

    def forward(self, y1, y2):
        batch_size = y1.shape[0]
        b1 = self.backbone(y1)
        z1 = self.projector(b1)
        b2 = self.backbone(y2)
        z2 = self.projector(b2)
        # print(z2)

        # empirical cross-correlation matrix
        c = self.bn(z1).T @ self.bn(z2)

        # sum the cross-correlation matrix between all gpus
        c.div_(self.args.batch_size)
        # torch.distributed.all_reduce(c)

        on_diag = torch.diagonal(c).add_(-1).pow_(2).sum()
        off_diag = off_diagonal(c).pow_(2).sum()
        loss = on_diag + self.args.lambd * off_diag
        return loss


class LARS(optim.Optimizer):
    def __init__(self, params, lr, weight_decay=0, momentum=0.9, eta=0.001,
                 weight_decay_filter=None, lars_adaptation_filter=None):
        defaults = dict(lr=lr, weight_decay=weight_decay, momentum=momentum,
                        eta=eta, weight_decay_filter=weight_decay_filter,
                        lars_adaptation_filter=lars_adaptation_filter)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self):
        for g in self.param_groups:
            for p in g['params']:
                dp = p.grad

                if dp is None:
                    continue

                if g['weight_decay_filter'] is None or not g['weight_decay_filter'](p):
                    dp = dp.add(p, alpha=g['weight_decay'])

                if g['lars_adaptation_filter'] is None or not g['lars_adaptation_filter'](p):
                    param_norm = torch.norm(p)
                    update_norm = torch.norm(dp)
                    one = torch.ones_like(param_norm)
                    q = torch.where(param_norm > 0.,
                                    torch.where(update_norm > 0,
                                                (g['eta'] * param_norm / update_norm), one), one)
                    dp = dp.mul(q)

                param_state = self.state[p]
                if 'mu' not in param_state:
                    param_state['mu'] = torch.zeros_like(p)
                mu = param_state['mu']
                mu.mul_(g['momentum']).add_(dp)

                p.add_(mu, alpha=-g['lr'])


def exclude_bias_and_norm(p):
    return p.ndim == 1


class GaussianBlur(object):
    def __init__(self, p):
        self.p = p

    def __call__(self, img):
        if random.random() < self.p:
            sigma = random.random() * 1.9 + 0.1
            return img.filter(ImageFilter.GaussianBlur(sigma))
        else:
            return img


class Solarization(object):
    def __init__(self, p):
        self.p = p

    def __call__(self, img):
        if random.random() < self.p:
            return ImageOps.solarize(img)
        else:
            return img


class Transform:
    def __init__(self):
        self.transform = A.Compose([
            A.Resize(128, 128, p=1),
            A.RandomScale(scale_limit=(0.75, 1.5), p=0.5),
            A.PadIfNeeded(min_height=128, min_width=128, p=1),
            A.CenterCrop(128, 128, p=1),
            A.RandomBrightnessContrast(brightness_limit=(-0.3, 0.3), contrast_limit=(-0.3, 0.3), p=0.5),
            A.Blur(p=0.5),
            A.Cutout(p=0.5),
            A.ElasticTransform(p=0.5, alpha_affine=10, border_mode=cv2.BORDER_CONSTANT),
            A.Normalize(mean=(0.28, 0.28, 0.28),
                        std=(0.031, 0.031, 0.031),
                        max_pixel_value=1.0, p=1),
            ToTensorV2(always_apply=True)
        ])
        self.transform_prime = A.Compose([
            A.Resize(128, 128, p=1),
            A.RandomScale(scale_limit=(0.75, 1.5), p=0.5),
            A.PadIfNeeded(min_height=128, min_width=128, p=1),
            A.CenterCrop(128, 128, p=1),
            A.RandomBrightnessContrast(brightness_limit=(-0.3, 0.3), contrast_limit=(-0.3, 0.3), p=0.5),
            A.Blur(p=0.5),
            A.Cutout(p=0.5),
            A.ElasticTransform(p=0.5, alpha_affine=10, border_mode=cv2.BORDER_CONSTANT),
            A.Normalize(mean=(0.28, 0.28, 0.28),
                        std=(0.031, 0.031, 0.031),
                        max_pixel_value=1.0, p=1),
            ToTensorV2(always_apply=True)
        ])

        # elastic deformation
        # local brightness and contrast changes

    def __call__(self, x):
        image = np.float32(np.array(x) / 255.0)
        y1 = self.transform(image=image)['image']
        y2 = self.transform_prime(image=image)['image']
        return y1, y2


if __name__ == '__main__':
    main()
