# ------------------------------------
# python modules
# ------------------------------------
from __future__ import division
import numpy as np
import torch
import torch.nn.functional as F
import sklearn.metrics
from sklearn.metrics import hamming_loss, roc_auc_score, accuracy_score
import sys

####################################################################################################################################
# Evaluation Metrics -- multilabel classification
####################################################################################################################################

def accuracy_eval(predicted_outputs, labels, args, stage, thresh=0.5):
    '''
        STAGE 1 -----------------------------------------------------------------------------------------------------------------
        For stage 1 multilabel classification
        Calculate accuracy. https://github.com/lidan1/multidensenet/blob/master/train.py
        -- microaveraging -- https://towardsdatascience.com/journey-to-the-center-of-multi-label-classification-384c40229bff#ee3b

        STAGE 2 -----------------------------------------------------------------------------------------------------------------
        Evaluation metrics for stage 2 of training.
        The final layer of the neural network is a log_softmax layer.

        y = log( e^xi / (sumj e^xj) )

        predicted_outputs has the form num_instances (eg 5) * num_outputs (2)
        labels eg [0, 0, 1, ... ]
        --------------------------------------------------------------------------------------------------------------------------

        https://github.com/kundajelab/ChromDragoNN/blob/master/utils/model_pipeline.py eval function

    '''
    if stage == 1:
        pred = torch.sigmoid(predicted_outputs).data.gt(thresh).float().cpu() #if were using basset loss use this to calculate accuracy
    else:
        if args.sigmoid_bool:
            pred = torch.sigmoid(predicted_outputs).data.gt(thresh).float().flatten().cpu()
        else:
            predicted_outputs = torch.index_select(predicted_outputs.cpu(), 1, torch.tensor([1])).view(-1)
            predicted_outputs = torch.exp(predicted_outputs)
            pred = torch.ge(predicted_outputs, thresh).float()

    labels = labels.float().cpu()
    true_positive = (pred + labels).eq(2).sum() #1+1
    true_negative = (pred + labels).eq(0).sum() #0+0
    false_positive = (pred - labels).eq(1).sum() #1-0
    false_negative = (pred - labels).eq(-1).sum() #0-1

    num = true_negative + true_positive
    denom = true_negative + true_positive + false_negative + false_positive

    try:
        accuracy = num.item()/denom.item()
    except (FloatingPointError, ZeroDivisionError):
        accuracy = 0.0

    try:
        precision = true_positive.item() / (true_positive + false_positive).item()
    except (FloatingPointError, ZeroDivisionError):
        precision = 0.0

    try:
        recall = true_positive.item() / (true_positive + false_negative).item()
    except (FloatingPointError, ZeroDivisionError):
        recall = 0.0

    try:
        F1_Score = 2 * ((precision * recall) / (precision + recall))
    except (FloatingPointError, ZeroDivisionError):
        F1_Score = 0.0

    return accuracy, F1_Score, precision, recall

def mse_eval(outputs, labels):
    '''
        For stage 1 multilabel classification
        Taken from ChromDragoNN. According to Basset. MSE? not rmse
    '''
    return torch.mean((outputs-labels)**2)

def roc_auc_score_modified(y_true, y_pred):
    '''
        https://stackoverflow.com/questions/45139163/roc-auc-score-only-one-class-present-in-y-true
    '''
    if len(np.unique(y_true)) == 1: # bug in roc_auc_score
        return accuracy_score(y_true, np.rint(y_pred))

    return roc_auc_score(y_true, y_pred)

def multilabel_eval(predicted_outputs, labels, thresh=0.5):
    '''

        For stage 1 multilabel classification

        Hamming Loss: Calculates loss generated using the XOR operation between the original binary strings of class labels
                    and predicted class labels for a data instance and calculates the average across the data set

        Jaccard Score: Size of the intersection divided by the size of the union of two label sets,
                    Compares set of predicted labels for a sample to the corresponding set of true labels in 'labels'

    '''
    pred = torch.sigmoid(predicted_outputs).data.gt(thresh).float()
    labels = labels.cpu()
    pred = pred.cpu()
    Exact_Match_Ratio = accuracy_score(labels, pred)
    hamloss = hamming_loss(labels, pred)
    return Exact_Match_Ratio, hamloss

def hamming_score(y_true, y_pred, normalize=True, sample_weight=None):
    '''
    Compute the Hamming score (a.k.a. label-based accuracy) for the multi-label case
    https://stackoverflow.com/q/32239577/395857
    '''
    acc_list = []
    for i in range(y_true.shape[0]):
        set_true = set( np.where(y_true[i])[0] )
        set_pred = set( np.where(y_pred[i])[0] )
        tmp_a = None
        if len(set_true) == 0 and len(set_pred) == 0:
            tmp_a = 1
        else:
            tmp_a = len(set_true.intersection(set_pred))/\
                    float( len(set_true.union(set_pred)) )
        acc_list.append(tmp_a)
    return np.mean(acc_list)

def multilabel_accuracy(predicted_outputs, labels, thresh=0.5):
    """
        Exact_Match_Ratio =  the set of labels predicted for a sample must exactly match the corresponding set of labels in y_true.
        Hamming score =  label-based accuracy for the multi-label case

    """
    pred = torch.sigmoid(predicted_outputs).data.gt(thresh).float().cpu()
    labels = labels.cpu()
    Exact_Match_Ratio = accuracy_score(labels, pred)
    Hamming_Score_ = hamming_score(labels, pred)
    return Hamming_Score_, Exact_Match_Ratio

def calculate_accuracy(predicted_outputs, labels, thresh=0.5):
    pred = torch.sigmoid(predicted_outputs).data.gt(thresh).float().cpu()
    labels = labels.cpu()
    accuracy = accuracy_score(labels, pred)
    return accuracy
