import argparse
import datetime
import os
import time

import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.init as init
import torch.optim as optim
import torch.utils.data as data

from data import (
    MEANS,
    POLLEN_ROOT,
    VOC_ROOT,
    POLLENDetection,
    VOCDetection,
    detection_collate,
    pollen,
    voc,
)
from layers.modules import MultiBoxLoss
from nvidia_ssd import build_ssd
from utils.augmentations import SSDAugmentation


def str2bool(v):
    return v.lower() in ('yes', 'true', 't', '1')


parser = argparse.ArgumentParser(
    description='Single Shot MultiBox Detector Training With Pytorch'
)
parser.add_argument(
    '--dataset',
    default='VOC',
    choices=['VOC', 'POLLEN'],
    type=str,
    help='VOC or POLLEN',
)
parser.add_argument(
    '--dataset_root', default=VOC_ROOT, help='Dataset root directory path'
)
parser.add_argument('--basenet', default='', help='Pretrained base model')
parser.add_argument(
    '--batch_size', default=32, type=int, help='Batch size for training'
)
parser.add_argument(
    '--resume',
    default=None,
    type=str,
    help='Checkpoint state_dict file to resume training from',
)
parser.add_argument(
    '--start_iter', default=0, type=int, help='Resume training at this iter'
)
parser.add_argument(
    '--num_workers', default=4, type=int, help='Number of workers used in dataloading'
)
parser.add_argument(
    '--cuda', default=True, type=str2bool, help='Use CUDA to train model'
)
parser.add_argument('--lr', default=1e-3, type=float, help='initial learning rate')
parser.add_argument(
    '--momentum', default=0.9, type=float, help='Momentum value for optim'
)
parser.add_argument(
    '--weight_decay', default=5e-4, type=float, help='Weight decay for SGD'
)
parser.add_argument('--gamma', default=0.1, type=float, help='Gamma update for SGD')
parser.add_argument(
    '--visdom', default=False, type=str2bool, help='Use visdom for loss visualization'
)
parser.add_argument(
    '--save_folder', default='weights/', help='Directory for saving checkpoint models'
)


if torch.cuda.is_available():
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
else:
    torch.set_default_tensor_type('torch.FloatTensor')


def train(args):
    run_id = datetime.datetime.now().strftime('%Y-%m-%dT%H-%M-%S')
    if args.dataset == 'VOC':
        cfg = voc
        dataset = VOCDetection(
            root=args.dataset_root, transform=SSDAugmentation(cfg['min_dim'], MEANS)
        )
    elif args.dataset == 'POLLEN':
        cfg = pollen
        dataset = POLLENDetection(
            root=args.dataset_root, transform=SSDAugmentation(cfg['min_dim'], MEANS)
        )

    if args.visdom:
        import visdom

        global viz
        viz = visdom.Visdom()

    ssd_net = build_ssd('train', cfg, args.basenet)
    net = ssd_net

    if args.cuda:
        net = torch.nn.DataParallel(ssd_net)
        cudnn.benchmark = True

    if args.resume:
        print('Resuming training, loading {}...'.format(args.resume))
        ssd_net.load_weights(args.resume)
    else:
        print('Initializing weights...')
        ssd_net._init_weights()

    if args.cuda:
        net = net.cuda()

    optimizer = optim.SGD(
        net.parameters(),
        lr=args.lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
    )
    criterion = MultiBoxLoss(
        cfg['num_classes'], 0.5, True, 0, True, 3, 0.5, False, args.cuda
    )

    net.train()
    # loss counters
    loc_loss = 0
    conf_loss = 0
    epoch = 0
    print('Loading the dataset...')

    epoch_size = len(dataset) // args.batch_size
    print('Training SSD on:', dataset.name)
    print('Using the specified args:')
    print(args)

    step_index = 0

    if args.visdom:
        vis_title = 'SSD.PyTorch on ' + dataset.name
        vis_legend = ['Loc Loss', 'Conf Loss', 'Total Loss']
        iter_plot = create_vis_plot('Iteration', 'Loss', vis_title, vis_legend)
        epoch_plot = create_vis_plot('Epoch', 'Loss', vis_title, vis_legend)

    data_loader = data.DataLoader(
        dataset,
        args.batch_size,
        num_workers=args.num_workers,
        shuffle=True,
        collate_fn=detection_collate,
        pin_memory=True,
    )
    # create batch iterator
    batch_iterator = iter(data_loader)
    for iteration in range(args.start_iter, cfg['max_iter']):
        if args.visdom and iteration != 0 and (iteration % epoch_size == 0):
            update_vis_plot(
                epoch, loc_loss, conf_loss, epoch_plot, 'append', epoch_size
            )
            # reset epoch loss counters
            loc_loss = 0
            conf_loss = 0
            epoch += 1

        if iteration in cfg['lr_steps']:
            step_index += 1
            adjust_learning_rate(optimizer, args.lr, args.gamma, step_index)

        # load train data
        try:
            images, targets = next(batch_iterator)
        except StopIteration:
            batch_iterator = iter(data_loader)
            images, targets = next(batch_iterator)

        if args.cuda:
            images = images.cuda()
            targets = [ann.cuda() for ann in targets]
        # forward
        t0 = time.time()
        out = net(images)
        # [print(x.size()) for x in out]
        # backprop
        optimizer.zero_grad()
        loss_l, loss_c = criterion(out, targets)
        loss = loss_l + loss_c
        loss.backward()
        optimizer.step()
        t1 = time.time()
        loc_loss += loss_l.item()
        conf_loss += loss_c.item()

        if iteration % 10 == 0:
            print('timer: %.4f sec.' % (t1 - t0))
            print(
                'iter ' + repr(iteration) + ' || Loss: %.4f ||' % (loss.item()),
                end=' ',
            )

        if args.visdom:
            update_vis_plot(
                iteration,
                loss_l.item(),
                loss_c.item(),
                iter_plot,
                'append',
            )

        if iteration != 0 and iteration % 5000 == 0:
            print('Saving state, iter:', iteration)
            torch.save(
                ssd_net.state_dict(),
                f'{args.save_folder}{run_id}_{args.dataset}_{iteration}.pth',
            )
    torch.save(
        ssd_net.state_dict(),
        f'{args.save_folder}{run_id}_{args.dataset}.pth',
    )


def adjust_learning_rate(optimizer, lr_start, gamma, step):
    """Sets the learning rate to the initial LR decayed by 10 at every
        specified step
    # Adapted from PyTorch Imagenet example:
    # https://github.com/pytorch/examples/blob/master/imagenet/main.py
    """
    lr = lr_start * (gamma ** (step))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def create_vis_plot(_xlabel, _ylabel, _title, _legend):
    return viz.line(
        X=torch.zeros((1,)).cpu(),
        Y=torch.zeros((1, 3)).cpu(),
        opts=dict(xlabel=_xlabel, ylabel=_ylabel, title=_title, legend=_legend),
    )


def update_vis_plot(iteration, loc, conf, window, update_type, epoch_size=1):
    viz.line(
        X=torch.ones((1, 3)).cpu() * iteration,
        Y=torch.Tensor([loc, conf, loc + conf]).unsqueeze(0).cpu() / epoch_size,
        win=window,
        update=update_type,
    )


if __name__ == '__main__':
    args = parser.parse_args()
    if not os.path.exists(args.save_folder):
        os.mkdir(args.save_folder)

    train(args)
