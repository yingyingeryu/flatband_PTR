#!/usr/bin/env python
# coding: utf-8

import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, TensorDataset

from sklearn.model_selection import train_test_split
from sklearn import metrics, preprocessing
from sklearn.utils import class_weight
import os
import json
import csv
from PTR_data import perodict_data, get_train_val_test_loader
from PTR_model import Net
import time
from torch.optim.lr_scheduler import MultiStepLR
from torch.autograd import Variable


def train(train_loader, model, criterion, optimizer, epoch, class_1_weight=1, class_eval_average='macro'):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    
    accuracies = AverageMeter()
    precisions = AverageMeter()
    recalls = AverageMeter()
    fscores = AverageMeter()
    auc_scores = AverageMeter()

    # switch to train mode
    model.train()

    end = time.time()
    for i, (inputs, target) in enumerate(train_loader, 0):
        # measure data loading time
        data_time.update(time.time() - end)

        target_normed = target.view(-1).long()
        
        target_var = Variable(target_normed)

        # compute output
        output = model(inputs)

        loss = criterion(output, target_var)
        try:
            accuracy, precision, recall, fscore, auc_score = \
                class_eval(output.data.cpu(), target, class_1_weight=class_1_weight, class_eval_average=class_eval_average)
        except ValueError:
            print('[Warning] only one class type in this batch.')
            pass

        losses.update(loss.data.cpu().item(), target.size(0))
        accuracies.update(accuracy, target.size(0))
        precisions.update(precision, target.size(0))
        recalls.update(recall, target.size(0))
        fscores.update(fscore, target.size(0))
        auc_scores.update(auc_score, target.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % 100 == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.3f} ({loss.avg:.3f})\t'
                  'Precision {prec.val[0]:.3f} {prec.val[1]:.3f} ({prec.avg[0]:.3f} {prec.avg[1]:.3f})\t'
                  'Recall {recall.val[0]:.3f} {recall.val[1]:.3f} ({recall.avg[0]:.3f} {recall.avg[1]:.3f})\t'
                  'F1 {f1.val[0]:.3f} {f1.val[1]:.3f} ({f1.avg[0]:.3f} {f1.avg[1]:.3f})\t'
                  'Accu {accu.val:.3f} ({accu.avg:.3f})\t'
                  'AUC {auc.val:.3f} ({auc.avg:.3f})'.format(
                epoch, i, len(train_loader), batch_time=batch_time,
                data_time=data_time, loss=losses, prec=precisions, 
                recall=recalls, f1=fscores,accu=accuracies,
                auc=auc_scores)
            )
            
def validate(val_loader, model, criterion, class_1_weight=1, class_eval_average='macro', test=False):
    batch_time = AverageMeter()
    losses = AverageMeter()

    accuracies = AverageMeter()
    precisions = AverageMeter()
    recalls = AverageMeter()
    fscores = AverageMeter()
    auc_scores = AverageMeter()
    if test:
        test_targets = []
        test_preds = []

    # switch to evaluate mode
    model.eval()

    end = time.time()
    for i, (inputs, target) in enumerate(val_loader,0):

        target_normed = target.view(-1).long()

        with torch.no_grad():
            target_var = Variable(target_normed)

#         # compute output
        output = model(inputs)
#         print('--------------------output-----------------------------')
#         print(output)
#         print('----------------target_var-----------------------------')
#         print(target_var)
#         print('---------------------target-------------------------')
#         print(target)
#         print('---------------------------------------------------------------------------')
        loss = criterion(output, target_var)
        losses.update(loss.data.cpu().item(), target.size(0))
        
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

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % 50 == 0:
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
            for target, pred in zip(test_targets, test_preds):
                writer.writerow((target, pred))
    else:
        star_label = '*'
    print(' {star} AUC {auc.avg:.3f}'.format(star=star_label,
                                             auc=auc_scores))
    return auc_scores.avg

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
            target_label, pred_label, average=None, sample_weight=sample_weight, warn_for=tuple())
        auc_score = metrics.roc_auc_score(target_label, prediction[:, 1], 
                                          sample_weight=sample_weight, average=class_eval_average)
        accuracy = metrics.accuracy_score(target_label, pred_label, sample_weight=sample_weight)
    else:
        raise NotImplementedError
    return accuracy, precision, recall, fscore, auc_score

        
        

if __name__ == '__main__':
    main()

