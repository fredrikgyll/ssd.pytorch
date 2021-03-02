"""Adapted from:
    @longcw faster_rcnn_pytorch: https://github.com/longcw/faster_rcnn_pytorch
    @rbgirshick py-faster-rcnn https://github.com/rbgirshick/py-faster-rcnn
    Licensed under The MIT License [see LICENSE for details]
"""

import argparse
import os
import pickle
import time
from pathlib import Path

import numpy as np
import torch
import torch.backends.cudnn as cudnn
from torch.autograd import Variable

from data import (
    POLLEN_CLASSES as labelmap,
    POLLEN_ROOT,
    POLLENDetection,
    VOCAnnotationTransform,
    pollen,
)
from nvidia_ssd import build_ssd
from utils import BaseTransform


def str2bool(v):
    return v.lower() in ("yes", "true", "t", "1")


parser = argparse.ArgumentParser(description='Single Shot MultiBox Detector Evaluation')
parser.add_argument(
    '--trained_model',
    default='weights/ssd300_mAP_77.43_v2.pth',
    type=str,
    help='Trained state_dict file path to open',
)
parser.add_argument(
    '--save_folder', default='eval/', type=str, help='File path to save results'
)
parser.add_argument(
    '--confidence_threshold',
    default=0.01,
    type=float,
    help='Detection confidence threshold',
)
parser.add_argument(
    '--top_k',
    default=5,
    type=int,
    help='Further restrict the number of predictions to parse',
)
parser.add_argument(
    '--cuda', default=True, type=str2bool, help='Use cuda to train model'
)
parser.add_argument(
    '--dataset_root', default=POLLEN_ROOT, help='Location of VOC root directory'
)
parser.add_argument(
    '--cleanup',
    default=True,
    type=str2bool,
    help='Cleanup and remove results files following eval',
)

args = parser.parse_args()

if not os.path.exists(args.save_folder):
    os.mkdir(args.save_folder)

if torch.cuda.is_available():
    if args.cuda:
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
    if not args.cuda:
        print(
            "WARNING: It looks like you have a CUDA device, but aren't using \
              CUDA.  Run with --cuda for optimal eval speed."
        )
        torch.set_default_tensor_type('torch.FloatTensor')
else:
    torch.set_default_tensor_type('torch.FloatTensor')

annopath = Path(
    os.path.join(args.dataset_root, 'POLLEN2019', 'Annotations', 'test.pkl')
)

YEAR = '2019'
devkit_path = args.dataset_root + 'POLLEN' + YEAR
dataset_mean = (104, 117, 123)
set_type = 'test'


class Timer(object):
    """A simple timer."""

    def __init__(self):
        self.total_time = 0.0
        self.calls = 0
        self.start_time = 0.0
        self.diff = 0.0
        self.average_time = 0.0

    def tic(self):
        # using time.time instead of time.clock because time time.clock
        # does not normalize for multithreading
        self.start_time = time.time()

    def toc(self, average=True):
        self.diff = time.time() - self.start_time
        self.total_time += self.diff
        self.calls += 1
        self.average_time = self.total_time / self.calls
        if average:
            return self.average_time
        else:
            return self.diff


def get_output_dir(name, phase):
    """Return the directory where experimental artifacts are placed.
    If the directory does not exist, it is created.
    A canonical path is built using the name from an imdb and a network
    (if not None).
    """
    filedir = os.path.join(name, phase)
    if not os.path.exists(filedir):
        os.makedirs(filedir)
    return filedir


def get_voc_results_file_template(image_set, cls):
    # VOCdevkit/VOC2007/results/det_test_aeroplane.txt
    filename = 'det_' + image_set + '_%s.txt' % (cls)
    filedir = os.path.join(devkit_path, 'results')
    if not os.path.exists(filedir):
        os.makedirs(filedir)
    path = os.path.join(filedir, filename)
    return path


def write_voc_results_file(all_boxes, dataset):
    for cls_ind, cls in enumerate(labelmap):
        print(f'Writing {cls:s} VOC results file')
        filename = get_voc_results_file_template(set_type, cls)
        with open(filename, 'wt') as f:
            for im_ind, index in enumerate(dataset.ids):
                dets = all_boxes[cls_ind + 1][im_ind]
                if dets.size == 0:
                    continue
                # the VOCdevkit expects 1-based indices
                for k in range(dets.shape[0]):
                    f.write(
                        f'{index[1]:s} {dets[k, -1]:.3f} '
                        f'{dets[k, 0] + 1:.1f} {dets[k, 1] + 1:.1f} '
                        f'{dets[k, 2] + 1:.1f} {dets[k, 3] + 1:.1f}\n'
                    )


def do_python_eval(output_dir='output', use_07=True):
    aps = []
    # The PASCAL VOC metric changed in 2010

    print('VOC07 metric? ' + ('Yes' if use_07 else 'No'))
    if not os.path.isdir(output_dir):
        os.mkdir(output_dir)
    for i, cls in enumerate(labelmap):
        filename = get_voc_results_file_template(set_type, cls)
        rec, prec, ap = voc_eval(
            filename,
            annopath,
            i,
            ovthresh=0.5,
            use_07_metric=use_07,
        )
        aps += [ap]
        print(f'AP for {cls} = {ap:.4f}')
        with open(os.path.join(output_dir, f'{cls}_pr.pkl'), 'wb') as f:
            pickle.dump({'rec': rec, 'prec': prec, 'ap': ap}, f)
    print(f'Mean AP = {np.mean(aps):.4f}')
    print('~~~~~~~~')
    print('Results:')
    for ap in aps:
        print(f'{ap:.3f}')
    print(f'{np.mean(aps):.3f}')
    print('~~~~~~~~')
    print('')
    print('--------------------------------------------------------------')
    print('Results computed with the **unofficial** Python eval code.')
    print('Results should be very close to the official MATLAB eval code.')
    print('--------------------------------------------------------------')


def voc_ap(rec, prec, use_07_metric=True):
    """ap = voc_ap(rec, prec, [use_07_metric])
    Compute VOC AP given precision and recall.
    If use_07_metric is true, uses the
    VOC 07 11 point method (default:True).
    """
    if use_07_metric:
        # 11 point metric
        ap = 0.0
        for t in np.arange(0.0, 1.1, 0.1):
            if np.sum(rec >= t) == 0:
                p = 0
            else:
                p = np.max(prec[rec >= t])
            ap = ap + p / 11.0
    else:
        # correct AP calculation
        # first append sentinel values at the end
        mrec = np.concatenate(([0.0], rec, [1.0]))
        mpre = np.concatenate(([0.0], prec, [0.0]))

        # compute the precision envelope
        for i in range(mpre.size - 1, 0, -1):
            mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

        # to calculate area under PR curve, look for points
        # where X axis (recall) changes value
        i = np.where(mrec[1:] != mrec[:-1])[0]

        # and sum (\Delta recall) * prec
        ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap


def voc_eval(
    detpath,
    annopath,
    classname,
    ovthresh=0.5,
    use_07_metric=True,
):
    """rec, prec, ap = voc_eval(detpath,
                               annopath,
                               classname,
                               [ovthresh],
                               [use_07_metric])
    Top level function that does the PASCAL VOC evaluation.
    detpath: Path to detections
       detpath.format(classname) should produce the detection results file.
    annopath: Path to annotations pickle file.
    classname: Category name (duh)
    [ovthresh]: Overlap threshold (default = 0.5)
    [use_07_metric]: Whether to use VOC07's 11 point AP computation
       (default True)"""
    # assumes detections are in detpath.format(classname)
    # first load gt
    recs = pickle.load(annopath.open('rb'))

    imagenames = list(recs.keys())

    # extract gt objects for this class
    class_recs = {}
    npos = 0
    for imagename in imagenames:
        bbox = np.array(
            [obj[:4] for obj in recs[imagename] if int(obj[4]) == classname]
        )
        difficult = np.zeros(bbox.shape[0]).astype(bool)
        det = [False] * len(bbox)
        npos = npos + sum(~difficult)
        class_recs[imagename] = {'bbox': bbox, 'difficult': difficult, 'det': det}

    # read dets
    detfile = detpath.format(classname)
    with open(detfile, 'r') as f:
        lines = f.readlines()
    if any(lines) == 1:

        splitlines = [x.strip().split(' ') for x in lines]
        image_ids = [x[0] for x in splitlines]
        confidence = np.array([float(x[1]) for x in splitlines])
        BB = np.array([[float(z) for z in x[2:]] for x in splitlines])

        # sort by confidence
        sorted_ind = np.argsort(-confidence)

        BB = BB[sorted_ind, :]
        image_ids = [image_ids[x] for x in sorted_ind]

        # go down dets and mark TPs and FPs
        nd = len(image_ids)
        tp = np.zeros(nd)
        fp = np.zeros(nd)
        for d in range(nd):
            R = class_recs[image_ids[d]]
            bb = BB[d, :].astype(float)
            ovmax = -np.inf
            BBGT = R['bbox'].astype(float)
            if BBGT.size > 0:
                # compute overlaps
                # intersection
                ixmin = np.maximum(BBGT[:, 0], bb[0])
                iymin = np.maximum(BBGT[:, 1], bb[1])
                ixmax = np.minimum(BBGT[:, 2], bb[2])
                iymax = np.minimum(BBGT[:, 3], bb[3])
                iw = np.maximum(ixmax - ixmin, 0.0)
                ih = np.maximum(iymax - iymin, 0.0)
                inters = iw * ih
                uni = (
                    (bb[2] - bb[0]) * (bb[3] - bb[1])
                    + (BBGT[:, 2] - BBGT[:, 0]) * (BBGT[:, 3] - BBGT[:, 1])
                    - inters
                )
                overlaps = inters / uni
                ovmax = np.max(overlaps)
                jmax = np.argmax(overlaps)

            if ovmax > ovthresh:
                if not R['difficult'][jmax]:
                    if not R['det'][jmax]:
                        tp[d] = 1.0
                        R['det'][jmax] = 1
                    else:
                        fp[d] = 1.0
            else:
                fp[d] = 1.0

        # compute precision recall
        fp = np.cumsum(fp)
        tp = np.cumsum(tp)
        rec = tp / float(npos)
        # avoid divide by zero in case the first detection matches a difficult
        # ground truth
        prec = tp / np.maximum(tp + fp, np.finfo(np.float64).eps)
        ap = voc_ap(rec, prec, use_07_metric)
    else:
        rec = -1.0
        prec = -1.0
        ap = -1.0

    return rec, prec, ap


def test_net(save_folder, net, cuda, dataset, top_k, im_size=300, thresh=0.05):
    num_images = len(dataset)
    # all detections are collected into:
    #    all_boxes[cls][image] = N x 5 array of detections in
    #    (x1, y1, x2, y2, score)
    all_boxes = [[[] for _ in range(num_images)] for _ in range(len(labelmap) + 1)]

    # timers
    _t = {'im_detect': Timer(), 'misc': Timer()}
    output_dir = get_output_dir('ssd300_pollen', set_type)
    det_file = os.path.join(output_dir, 'detections.pkl')

    for i in range(num_images):
        im, gt, h, w = dataset.pull_item(i)

        x = Variable(im.unsqueeze(0))
        if args.cuda:
            x = x.cuda()
        _t['im_detect'].tic()
        with torch.no_grad():
            detections = net(x)
        detect_time = _t['im_detect'].toc(average=False)

        # skip j = 0, because it's the background class
        for j in range(1, detections.size(1)):
            dets = detections[0, j, :]
            mask = dets[:, 0].gt(0.0).expand(5, dets.size(0)).t()
            dets = torch.masked_select(dets, mask).view(-1, 5)
            if dets.size(0) == 0:
                continue
            boxes = dets[:, 1:]
            boxes[:, 0] *= w
            boxes[:, 2] *= w
            boxes[:, 1] *= h
            boxes[:, 3] *= h
            scores = dets[:, 0].cpu().numpy()
            cls_dets = np.hstack((boxes.cpu().numpy(), scores[:, np.newaxis])).astype(
                np.float32, copy=False
            )
            all_boxes[j][i] = cls_dets

        print(f'im_detect: {i+1:d}/{num_images:d} {detect_time:.3f}s')

    with open(det_file, 'wb') as f:
        pickle.dump(all_boxes, f, pickle.HIGHEST_PROTOCOL)

    print('Evaluating detections')
    evaluate_detections(all_boxes, output_dir, dataset)


def evaluate_detections(box_list, output_dir, dataset):
    write_voc_results_file(box_list, dataset)
    do_python_eval(output_dir)


if __name__ == '__main__':
    # load net
    cfg = pollen
    num_classes = len(labelmap) + 1  # +1 for background
    net = build_ssd('test', pollen)  # initialize SSD
    net.load_state_dict(torch.load(args.trained_model))
    net.eval()
    print('Finished loading model!')
    # load data
    dataset = POLLENDetection(
        args.dataset_root,
        [('2019', set_type)],
        BaseTransform(net.size),
    )
    if args.cuda:
        net = net.cuda()
        cudnn.benchmark = True
    # evaluation
    test_net(
        args.save_folder,
        net,
        args.cuda,
        dataset,
        args.top_k,
        300,
        thresh=args.confidence_threshold,
    )
