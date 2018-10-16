import torch
import torch.nn as nn
import numpy as np
import torchvision as tv
import torchvision.transforms as transforms
import torch.utils.data as data
import argparse
import logging
import torch.backends.cudnn as cudnn
import os
import time
from dataset.mnist import get_dataset
from torch import optim
from utils.criterioni import accuracy, joint_loss
from utils.AverageMeter import AverageMeter
from models.lenet import Lenet
import random

logger = logging.getLogger()
logger.setLevel(logging.INFO)

def parse_args():
    parser = argparse.ArgumentParser(description='command for train pseudo-labels model')
    parser.add_argument('--lr', type=float, default=1.5, help='learning rate')
    parser.add_argument('--size_labeled', type=int, default=32, help='#labeled images in each mini-batch')
    parser.add_argument('--size_unlabeled', type=int, default=256, help='#unlabeled images in each mini-batch')
    parser.add_argument('--batch_size', type=int, default=128, help='batch size for val/test')
    parser.add_argument('--epoch', type=int, default=1000, help='#training epoches')
    parser.add_argument('--gpu', type=int, default=None, help='GPU ID (negative value indicates CPU')
    parser.add_argument('--out', type=str, default='./data/model_data', help='root of the output')
    parser.add_argument('--train_root', type=str, default='./data/img_data/train', help='root of the train dataset')
    parser.add_argument('--test_root', type=str, default='./data/img_data/test', help='root of the test dataset')
    parser.add_argument('--download', type=float, default=True, help='download dataset')
    parser.add_argument('--seed', type=int, default=None, help='seed for initializing training')
    args = parser.parse_args()

    return args

def data_config(args):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    train_labeled, train_unlabeled, val = get_dataset(args, transform, transform)
    test = tv.datasets.MNIST(args.test_root, train=False, transform=transform, download=args.download)

    train_labeled_loader = data.DataLoader(train_labeled, batch_size=args.size_labeled, shuffle=True, num_workers=4)
    train_unlabeled_loader = data.DataLoader(train_unlabeled, batch_size=args.size_unlabeled, shuffle=True, num_workers=4)
    val_loader = data.DataLoader(val, batch_size=args.size_labeled, shuffle=False, num_workers=4)
    test_loader = data.DataLoader(test, batch_size=args.batch_size, shuffle=False, num_workers=4)
    print('-------> Data loading')
    return train_labeled_loader, train_unlabeled_loader, val_loader, test_loader


def network_config(args):
    # Random seed
    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
    network = Lenet()
    if args.gpu is not None:
        network = network.cuda(args.gpu)
    else:
        network = nn.DataParallel(network).cuda()
    print('Total params: %2.fM' % (sum(p.numel()) for p in network.parameters() / 1000000.0))

    # need to update
    optimizer = optim.SGD(network.parameters(), lr=args.lr, momentum= 0.5)
    cudnn.benchmark = True

    return network, optimizer

def record_params(args):
    dst_folder = args.out + '/lr-{}'.format(args.lr)
    if not os.path.exists(dst_folder):
        os.makedirs(dst_folder)

    rd = open(dst_folder + '/config.txt', 'w')
    rd.write('lr:%f' % args.lr + '\n')
    rd.close()

    handler = logging.FileHandler(dst_folder + '/train.log')
    handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    return dst_folder

def record_result(dst_folder, best_ac):
    dst = dst_folder + '/config.txt'
    rd = open(dst, 'a+')
    rd.write('best_ac:%3f' % best_ac + '\n')
    rd.close()

def save_checkpoint(state, dst_folder, epoch):
    dst = dst_folder + '/epoch-' + str(epoch) + '.pkl'
    torch.save(state, dst)

def train(train_labeled_loader, train_unlabeled_loader, network, optimizer, epoch, args):
    batch_time = AverageMeter()
    train_loss = AverageMeter()
    conditional_entropy = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    iter_labeled = iter(train_labeled_loader)
    iter_unlabeled = iter(train_unlabeled_loader)

    # switch to train mode
    network.train()

    # measure data loaading time
    end = time.time()

    for i in range(len(train_labeled_loader)):
        img_labeled, labels = next(iter_labeled)
        img_unlabeled, _ = next(iter_unlabeled)
        img_labeled = img_labeled.cuda(args.gpu, non_blocking=True)
        labels = labels.cuda(args.gpu, non_blocking=True)
        img_unlabeled = img_unlabeled.cuda(args.gpu, non_blocking=True)

        # compute output
        outputs_labeled = network(img_labeled)
        outputs_unlabeled = network(img_unlabeled)
        loss = joint_loss(outputs_labeled, outputs_unlabeled, labels, epoch)

        prec1, prec5 = accuracy(outputs_labeled, labels, top=[1,5])
        train_loss = joint_loss(outputs_labeled, outputs_unlabeled, labels, epoch)
        top1.update(prec1, img_labeled.size(0))
        top5.update(prec5, img_labeled.size(0))

        # compute gradient and  do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

    return train_loss.avg, top5.avg, top1.avg, batch_time.sum

def validate(val_loader, network, criterion):
    batch_time = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    val_loss = AverageMeter()

    # switch to evaluate mode
    network.eval()

    with torch.no_grad():
        end = time.time()
        for images, labels in val_loader:
            images = images.cuda(args.gpu, non_blocking=True)
            labels = labels.cuda(args.gpu, non_blocking=True)
            outputs = network(images)
            prec1, prec5 = accuracy(outputs, labels, top=[1,5])
            loss = criterion(outputs, labels) * images.size(1)

            top1.update(prec1, images.size(0))
            top5.update(prec5, images.size(0))
            val_loss.update(loss, images.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

    return top5.avg, top1.avg, batch_time.sum

def main(args, dst_folder):
    # only record the best top1_ac for validation set.
    best_ac = 0.0

    # data loader
    train_labeled_loader, train_unlabeled_loader, val_loader, test_loader = data_config(args)

    # criterion
    val_criterion = nn.BCEWithLogitsLoss()

    # network config
    network, optimizer = network_config()

    for epoch in range(args.epoch):
        train_loss, top5_train_ac, top1_train_ac, train_time = train(train_labeled_loader, train_unlabeled_loader, network, optimizer, epoch, args)
        # evaluate on validation set
        top5_val_ac, top1_val_ac, val_time = validate(val_loader, network, val_criterion)
        # remember best prec@1, save checkpoint and logging to the console
        if top1_val_ac >= best_ac:
            state = {'state_dict': network.state_dict(), 'epoch': epoch, 'ac': [top5_val_ac, top1_val_ac], 'best_ac': best_ac}
            best_ac = top1_val_ac
            # save model
            save_checkpoint(state, dst_folder, epoch)
        # logging
        logging.info('Epoch: [{}|{}], train_loss: {:.3f}, top1_train_ac: {:.3f}, top5_val_ac: {:.3f}, top1_val_ac: {:.3f}, val_time: {:.3f}, train_time: {:.3f}'.format(epoch, args.epoch, train_loss, top1_train_ac, top5_val_ac, top1_val_ac, val_time, train_time))

    print('Best ac: %f' % best_ac)
    record_result(dst_folder, best_ac)

if __name__ == '__main__':
    args = parse_args()
    logging.info(args)
    # record params
    dst_folder = record_params(args)
    # train
    main(args, dst_folder)