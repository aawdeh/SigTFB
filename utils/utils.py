# ------------------------------------
# python modules
# ------------------------------------
from __future__ import division
import numpy as np
import sklearn.metrics
from sklearn.model_selection import train_test_split
import sys
from sys import path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
import pandas as pd
import math
import random
from sklearn.decomposition import PCA
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

# ------------------------------------
# own modules
# ------------------------------------
path.append("/project/6006657/aaseel/Aim2/Scripts/")
path.append('/project/6006657/aaseel/Aim2/SW/ChromDragoNN/pytorch_classification/utils')
from DNN.misc import AverageMeter
from DNN.evaluationMetrics import mse_eval, accuracy_eval, roc_auc_score_modified, multilabel_eval

####################################################################################################################################
# Set SEED
####################################################################################################################################
# torch.backends.cudnn.deterministic = True
# torch.backends.cudnn.benchmark = False
# np.random.seed(0)
# random.seed(0)
####################################################################################################################################
def findNumberofExonsPerTF(file):
    """
        Need to find number of exons per TF to find range of concatenated data.
    """

    h5_file = h5py.File(file, 'r')
    TF_list = list(h5.file.keys())

    exonList = []
    for TF in TF_list:
        exonList.append(len(h5[TF][list(h5[TF].keys())[0]][list(h5[TF][list(h5[TF].keys())[0]].keys())[0]]['table'][:]))

    print("Smallest number of exons " + str(min(exonList)) + " , Largest number of exons " + str(max(exonList))) #1, 166

def load_model_checkpoint(filepath, device):
    """
        https://discuss.pytorch.org/t/saving-customized-model-architecture/21512/2
        state_dict --  a dict that maps each layer to its parameter tensor -- only layers with learnable parameters
    """
    print("Load best model from stage 1.")
    print(filepath)
    checkpoint = torch.load(filepath, map_location=torch.device(device))
    model_stage = checkpoint['model']
    args = checkpoint['args']
    # load pretrained weights
    model_stage.load_state_dict(checkpoint['state_dict'])

    return checkpoint, model_stage, args

def basset_loss(outputs, targets, args, stage=1):
    '''
        Taken from ChromDragoNN. According to Basset.
    '''
    #BCE : (targets*log(sigmoid(outputs)) + (1-targets)*(log(1-sigmoid(outputs) ) ) )
    # outputs are softmax values for each cell type, shape batch_size x NUM_CELL_TYPES
    if args.customSampler_OneLoss and stage==1:
        #here we need value for each output node
        return targets*F.logsigmoid(outputs) + (1-targets)*(-outputs + F.logsigmoid(outputs))
    else:
        if stage==1:
            return -torch.sum(targets*F.logsigmoid(outputs) + (1-targets)*(-outputs + F.logsigmoid(outputs)))/targets.size()[0] #batchsize
        else:
            return -torch.sum(targets*F.logsigmoid(outputs.flatten()) + (1-targets)*(-outputs.flatten() + F.logsigmoid(outputs.flatten())))/targets.size()[0] #batchsize

def getBaseline(ratio_P_N):
    '''
        Input: is ratio of number of positive instances over number of instances for each class.

        Output: Returns the baseline value of 'y' that will give us the least loss for each class [c1, c2 ... ]

        Here we derive the basset function and equate it to 0.
    '''
    return np.log(ratio_P_N) - np.log(1 - ratio_P_N)

def evaluateBaseline(valid_dl, device, model, baselineBool, maxClass=None):
    '''
    '''
    print('Evaluate model')
    LossOverall = AverageMeter()
    MSEOverall = AverageMeter()
    AccuracyOverall = AverageMeter()
    PrecisionOverall = AverageMeter()
    RecallOverall = AverageMeter()
    HammingLossOverall = AverageMeter()

    all_preds=[]
    all_targets=[]
    stop=0

    model.eval()
    with torch.no_grad():
        for batch_index, (inputs, labels) in enumerate(valid_dl):
            inputs, labels = inputs.to(device), labels.to(device) #transfer the data to gpu
            batch_size = labels.size(0)

            if baselineBool:
                predicted_outputs = torch.zeros(batch_size, len(maxClass))
                for c in range(len(maxClass)):
                    predicted_outputs[:,c] = maxClass[c]
            else:
                predicted_outputs, _, _ = model(inputs)

            predicted_outputs = predicted_outputs.to(device)
            loss = basset_loss(predicted_outputs, labels)

            # Stats -- method according to chromdragonn
            mse = mse_eval(predicted_outputs.data, labels.data) #per batch
            accuracy, precision, recall = accuracy_eval(predicted_outputs, labels)
            EMR, hamming_loss = multilabel_eval(predicted_outputs, labels)

            #Update metrics -- find better way?
            LossOverall.update(loss.item(), batch_size)
            MSEOverall.update(mse.item(), batch_size)
            AccuracyOverall.update(accuracy, batch_size)
            PrecisionOverall.update(precision, batch_size)
            RecallOverall.update(recall, batch_size)
            HammingLossOverall.update(hamming_loss, batch_size)

            all_preds.append(predicted_outputs.cpu().data.numpy())
            all_targets.append(labels.cpu().data.numpy())

            print("Batch: " + str(batch_index+1)
                   + " ,Loss: " + str(loss.item())
                   + " ,MSE: " + str(MSEOverall.avg)
                   + " ,Accuracy: " + str(AccuracyOverall.avg))

            # stop+=1
            # if stop==2:
            #     break

        #Taken from ChromDragoNN -- finds measurement across instances in each CL then takes the mean -- macroaveraging? Find precision per class and then take the avg
        all_targets = np.concatenate(all_targets, axis=0)
        all_preds = np.concatenate(all_preds, axis=0)
        auprc = np.mean([sklearn.metrics.average_precision_score(all_targets[:,i], all_preds[:,i]) for i in range(all_preds.shape[1])])
        auc = np.mean([roc_auc_score_modified(all_targets[:,i], all_preds[:,i]) for i in range(all_preds.shape[1])])

    metrics = {"loss": LossOverall.avg, "MSE":MSEOverall.avg, "Accuracy":AccuracyOverall.avg, "Precision":PrecisionOverall.avg, "Recall":RecallOverall.avg,
                "HammingLoss": HammingLossOverall.avg, "AUPRC":auprc, "AUC":auc}

    return metrics

def evaluate_rangeofbaselines(valid_dl, device, model, baselineBool, maxClass):
    '''
        The idea here is to find the best baseline for the basset loss function to compare results to.
        To do so we will look at a range of numbers from 0->1. [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
    '''

    dimen = len(maxClass)
    baselineRange = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]

    basset_loss_range = {}
    for constant in baselineRange:
        metrics = evaluateBaseline(valid_dl, device, model, baselineBool=True, maxClass=[constant]*dimen)
        basset_loss_range[constant] = metrics['loss']

    return basset_loss_range

def plotMetric(avg_train, avg_valid, ylabelTitle, modelname, PATH):
    '''
        Plot learning curves
    '''
    fig = plt.figure()
    plt.plot(avg_train, '-r', label="Training")
    if len(avg_valid) == len(avg_train):
        plt.plot(avg_valid, '-b', label="Validation")
    plt.ylabel(ylabelTitle)
    plt.xlabel('Iterations (per tens)')
    plt.legend(loc='upper right')
    plt.show()
    plt.savefig(os.path.join(PATH, ylabelTitle + ".pdf"), bbox_inches="tight")

def saveData(avg_train, avg_valid, metricname, modelname, PATH):
    '''
        Save metric per epoch for validation and training.
    '''
    epochs = np.arange(1,len(avg_train)+1)

    if len(avg_train) == len(avg_valid):
        df = pd.DataFrame({"Epochs":epochs, "Train":avg_train, "Valid":avg_valid})
        df.to_csv(os.path.join(PATH, modelname + "_" + metricname + ".csv"),index=False)
    else:
        df = pd.DataFrame({"Epochs":epochs, "Train":avg_train})
        df.to_csv(os.path.join(PATH, modelname + "_" + metricname + ".csv"),index=False)

def save_model(model, args):
    """

    """
    print("Ax Train :: Stage 2 :: Save model for run " + str(args.trial_index) + ".")
    print(args)
    state = {'state_dict': model.state_dict(),
             'args': args,
             'model': model}

    Path(os.path.join(args.saveModelPath, args.TF, args.AB)).mkdir(parents=True, exist_ok=True)
    torch.save(state, os.path.join(args.saveModelPath, args.TF, args.AB, args.TF + "." + args.AB + "." + str(int(args.trial_index)) +'.pth.tar'))

def saveIndex(train_idx, validate_idx, args):
    """
    """
    print("Ax Train :: Stage 2 :: Save indexes for run " + str(args.trial_index) + ".")
    Path(os.path.join(args.idxPath, args.TF, args.AB)).mkdir(parents=True, exist_ok=True)

    with open(os.path.join(args.idxPath, args.TF, args.AB, 'train' + str(int(args.trial_index)) + '.txt'), 'w') as f:
        f.write('\n'.join(str(num) for num in train_idx))

    with open(os.path.join(args.idxPath, args.TF, args.AB, 'valid' + str(int(args.trial_index)) +'.txt'), 'w') as f:
        f.write('\n'.join(str(num) for num in validate_idx))

def getIndex(idxPath,run):
    """
        Get train and validation indexes obtained when running the Ax optimzier when tuning the hyperparameters.
    """
    train_index = pd.read_csv(os.path.join(idxPath, "train" + str(run)+ ".txt"),header=None)[0].values.tolist()
    valid_index = pd.read_csv(os.path.join(idxPath, "valid" + str(run)+ ".txt"),header=None)[0].values.tolist()
    return train_index, valid_index

def torch_seed(seed):
    """
        https://sajjjadayobi.github.io/blog/tips/2021/02/24/reproducibility.html
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.cuda.manual_seed_all(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
