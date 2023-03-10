#!/usr/bin/env python
# coding: utf-8


import argparse
import os
import shutil
import sys
import time
import warnings
from random import sample

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn import metrics
from torch.autograd import Variable
from torch.optim.lr_scheduler import MultiStepLR

from PTR_model import Net
from PTR_data import PeriodicTable, get_train_val_test_loader, collate_pool
from sklearn.utils import class_weight
from torch.utils.data import Dataset, DataLoader, TensorDataset

parser = argparse.ArgumentParser(description='PTR classification model')
parser.add_argument('modelpath', help='path to the trained model.')
#parser.add_argument('cifpath', help='path to the directory of CIF files.')
parser.add_argument('data_options', metavar='OPTIONS', nargs='+',
                    help='dataset options')
parser.add_argument('--task', choices=['regression', 'classification'],
                    default='regression', help='complete a regression or '
                                                   'classification task (default: regression)')
parser.add_argument('-b', '--batch-size', default=256, type=int,
                    metavar='N', help='mini-batch size (default: 256)')
parser.add_argument('-j', '--workers', default=0, type=int, metavar='N',
                    help='number of data loading workers (default: 0)')
parser.add_argument('--disable-cuda', action='store_true',
                    help='Disable CUDA')
parser.add_argument('--print-freq', '-p', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')

parser.add_argument( '--class_1_weight','--class1w', default=1, type=int, metavar='N',
                    help='the weight of class 1 for accuracy parameter calculation, '
                    'only usefull for classification')
parser.add_argument('--class_eval_average','--average', default='macro', type=str, metavar='macro',
                    help='the average method for auc_score calculation,' 
                   'only usefull for classification')
parser.add_argument('--label_1_weight', '--label1w',default=1, type=int, metavar='N',
                    help='the weight of class 1 during data batch,'
                   'It defines the probability of being choised into this batch for class 1.'
                   'only usefull for classification')
args = parser.parse_args(sys.argv[1:])


if os.path.isfile(args.modelpath):
    print("=> loading model params '{}'".format(args.modelpath))
    model_checkpoint = torch.load(args.modelpath,
                                  map_location=lambda storage, loc: storage)
    model_args = argparse.Namespace(**model_checkpoint['args'])
    print("=> loaded model params '{}'".format(args.modelpath))
else:
    print("=> no model params found at '{}'".format(args.modelpath))

args.cuda = not args.disable_cuda and torch.cuda.is_available()


if args.task == 'regression':
    best_mae_error = 1e10
else:
    best_mae_error = 0.


def main():
    global args, best_mae_error
    
    #dataset= PeriodicTable(*args.data_options)
    dataset=PeriodicTable('A2B2O7_flatstate.csv')
    collate_fn = collate_pool
    
    test_loader = DataLoader(dataset, batch_size=32, shuffle=True,
                        num_workers=0, collate_fn=collate_fn,
                        pin_memory=False)

    model=Net(classification=True) 

    if model_args.task == 'classification':
        criterion = nn.NLLLoss()
    else:
        criterion = nn.MSELoss()


    normalizer = Normalizer(torch.zeros(2))
    checkpoint = torch.load(args.modelpath, map_location=lambda storage, loc: storage)
    model.load_state_dict(checkpoint['state_dict'])
    normalizer.load_state_dict(checkpoint['normalizer'])

    validate(test_loader, model, criterion, normalizer, test=True)

            
def validate(val_loader, model, criterion, normalizer, class_1_weight=1, class_eval_average='macro', test=False):
    batch_time = AverageMeter()
    losses = AverageMeter()
    
    if args.task == 'regression':
        mae_errors = AverageMeter()
    else:

        accuracies = AverageMeter()
        precisions = AverageMeter()
        recalls = AverageMeter()
        fscores = AverageMeter()
        auc_scores = AverageMeter()
    if test:
        test_targets = []
        test_preds = []
        test_crystal = []

    # switch to evaluate mode
    model.eval()

    end = time.time()
    for i, (inputs, target, crystal) in enumerate(val_loader,0):

        if args.task == 'regression':
            target_normed = normalizer.norm(target)
        else:
            target_normed = target.view(-1).long()
        if args.cuda:
            with torch.no_grad():
                target_var = Variable(target_normed.cuda(non_blocking=True))
        else:
            with torch.no_grad():
                target_var = Variable(target_normed)
#         # compute output
        output = model(inputs)
        loss = criterion(output, target_var)
        losses.update(loss.data.cpu().item(), target.size(0))
        
        if args.task == 'regression':
            mae_error = mae(normalizer.denorm(output.data.cpu()), target)
            losses.update(loss.data.cpu().item(), target.size(0))
            mae_errors.update(mae_error, target.size(0))
            if test:
                test_pred = normalizer.denorm(output.data.cpu())
                test_target = target
                test_preds += test_pred.view(-1).tolist()
                test_targets += test_target.view(-1).tolist()
                test_crystal += crystal
        else:
            try:
                accuracy, precision, recall, fscore, auc_score = \
                    class_eval(output.data.cpu(), target, class_1_weight=class_1_weight, class_eval_average=class_eval_average)
            except ValueError:
                print('[Warning] only one class type in this batch.')
                pass

            accuracies.update(accuracy, target.size(0))
            precisions.update(precision, target.size(0))
            recalls.update(recall, target.size(0))
            fscores.update(fscore, target.size(0))
            auc_scores.update(auc_score, target.size(0))
            if test:
                test_pred = torch.exp(output.data.cpu())
                test_target = target
                assert test_pred.shape[1] == 2
                test_preds += test_pred[:, 1].tolist()
                test_targets += test_target.view(-1).tolist()
                test_crystal += crystal

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            if args.task == 'regression':
                print('Test: [{0}/{1}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'MAE {mae_errors.val:.3f} ({mae_errors.avg:.3f})'.format(
                    i, len(val_loader), batch_time=batch_time, loss=losses,
                    mae_errors=mae_errors))
            else:
                print('Test: [{0}/{1}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Precision {prec.val[0]:.4f} {prec.val[1]:.4f} ({prec.avg[0]:.4f} {prec.avg[1]:.4f})\t'
                      'Recall {recall.val[0]:.4f} {recall.val[1]:.4f} ({recall.avg[0]:.4f} {recall.avg[1]:.4f})\t'
                      'F1 {f1.val[0]:.4f} {f1.val[1]:.4f} ({f1.avg[0]:.4f} {f1.avg[1]:.4f})\t'
                       'Accu {accu.val:.3f} ({accu.avg:.3f})\t'
                      'AUC {auc.val:.3f} ({auc.avg:.3f})'.format(
                    i, len(val_loader), batch_time=batch_time, loss=losses,
                    prec=precisions, recall=recalls, f1=fscores, 
                    accu=accuracies, auc=auc_scores))

    if test:
        star_label = '**'
        import csv
        with open('test_results.csv', 'w') as f:
            writer = csv.writer(f)
            for crystal, target, pred in zip(test_crystal, test_targets, test_preds):
                writer.writerow((crystal,target, pred))
    else:
        star_label = '*'
    if args.task == 'regression':
        print(' {star} MAE {mae_errors.avg:.3f}'.format(star=star_label,
                                                        mae_errors=mae_errors))
        return mae_errors.avg
    else:
        print(' {star} Accu {accu.avg:.3f}'.format(star=star_label,
                                                 accu=accuracies))
        return accuracies.avg

class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def class_eval(prediction, target, class_1_weight=1, class_eval_average= 'macro'):
    try:
        sample_weight=class_weight.compute_sample_weight({0:1, 1:class_1_weight}, target)
    except ValueError:
        print('[Warning] only one class type in this batch.')
        sample_weight = None
        
    prediction = np.exp(prediction.numpy())
    target = target.numpy()
    pred_label = np.argmax(prediction, axis=1)
    target_label = np.squeeze(target)
    if not target_label.shape:
        target_label = np.asarray([target_label])
    if prediction.shape[1] == 2:
        precision, recall, fscore, _ = metrics.precision_recall_fscore_support(
            target_label, pred_label, average=None, sample_weight=sample_weight)
        auc_score = metrics.roc_auc_score(target_label, prediction[:, 1], sample_weight=sample_weight, average=class_eval_average)
        accuracy = metrics.accuracy_score(target_label, pred_label, sample_weight=sample_weight)
    else:
        raise NotImplementedError
    return accuracy, precision, recall, fscore, auc_score
     
def mae(prediction, target):
    """
    Computes the mean absolute error between prediction and target

    Parameters
    ----------

    prediction: torch.Tensor (N, 1)
    target: torch.Tensor (N, 1)
    """
    return torch.mean(torch.abs(target - prediction))
    
    
def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth.tar')
        
def adjust_learning_rate(optimizer, epoch, k):
    """Sets the learning rate to the initial LR decayed by 10 every k epochs"""
    assert type(k) is int
    lr = args.lr * (0.1 ** (epoch // k))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

class Normalizer(object):
    """Normalize a Tensor and restore it later. """

    def __init__(self, tensor):
        """tensor is taken as a sample to calculate the mean and std"""
        self.mean = torch.mean(tensor)
        self.std = torch.std(tensor)

    def norm(self, tensor):
        return (tensor - self.mean) / self.std

    def denorm(self, normed_tensor):
        return normed_tensor * self.std + self.mean

    def state_dict(self):
        return {'mean': self.mean,
                'std': self.std}

    def load_state_dict(self, state_dict):
        self.mean = state_dict['mean']
        self.std = state_dict['std']
        
        
if __name__ == '__main__':
    main()
