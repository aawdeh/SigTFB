import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset, Subset

import sys
from sys import path
from typing import Dict, List, Optional, Tuple
from sklearn.model_selection import train_test_split
import sklearn.metrics
from sklearn.metrics import average_precision_score
import numpy as np

# ------------------------------------
# own modules
# ------------------------------------
from utils.misc import AverageMeter
from utils.evaluationMetrics import accuracy_eval, roc_auc_score_modified
from utils.loadData import LoadData
from utils.args import getArgs_Stage2 as Args

#####################################################################################
# Inspired by https://github.com/facebook/Ax/blob/master/tutorials/tune_cnn.ipynb
# train -- function to optimize using Ax
# evaluate -- function to optimize using Ax
# train_epoch_merged -- to get metrics per epoch
# evaluate_epoch_merged -- to get metrics per epoch
# getRNAandChIP
#####################################################################################
def train(
    net: torch.nn.Module,
    dl : LoadData,
    train_loader: DataLoader,
    parameters: Dict[str, float],
    args : Args,
    dtype: torch.dtype,
    device: torch.device,
) -> nn.Module:
    """
        Train CNN on data -- used for tuning

        Args:
            net: initialized neural network for stage 2
            train_loader: DataLoader containing training set
            parameters: dictionary containing parameters to be passed to the optimizer.
                - lr: default (0.001)
                - momentum: default (0.0)
            args: dictionary of other parameters
            dtype: torch dtype
            device: torch device

        Returns:
            nn.Module: trained CNN.
    """
    print(args)
    # Initialize network
    net.to(dtype=dtype, device=device)
    net.train()

    # Define loss and optimizer
    criterion = nn.NLLLoss()
    optimizer = optim.SGD(
            [param for param in net.parameters() if param.requires_grad],
            lr = parameters.get("learning_rate", 0.001),
            momentum = parameters.get("momentum_rate", 0.0),
            weight_decay = parameters.get("weight_decay", 0.0),
            nesterov = True)

    num_epochs = parameters.get("num_epochs", 1)
    for e in range(num_epochs):
        for i, batch_unit in enumerate(train_loader):
            ChIPseq_batch, rnaCount_batch, atac_labels, labels, _ = getData(dl, batch_unit, args, type="train")
            optimizer.zero_grad()
            _, _, predicted_outputs = net(ChIPseq_batch, rnaCount_batch, atac_labels)
            loss = criterion(predicted_outputs, labels.long())
            loss.backward()
            optimizer.step()
            print("Epoch " + str(e) + ", Batch " + str(i) + ", loss " + str(loss.mean().item()))

    return net

#####################################################################################

def evaluate(
    net: nn.Module,
    dl: LoadData,
    data_loader: DataLoader,
    args: Args,
    dtype: torch.dtype,
    device: torch.device
) -> float:
    """
    Compute classification accuracy on provided dataset using for tuning

    Args:
        net: trained model
        data_loader: DataLoader containing the evaluation set
        dtype: torch dtype
        device: torch device

    Returns:
        dict: evalution metrics metrics
    """
    LossOverall = AverageMeter()
    AccuracyOverall = AverageMeter()

    all_preds=[]
    all_targets=[]

    net.eval()
    criterion = nn.NLLLoss()
    print("Evaluate")
    with torch.no_grad():
        for i, batch_unit in enumerate(data_loader):
            ChIPseq_batch, rnaCount_batch, atac_labels, labels, _ = getData(dl, batch_unit, args, type="valid")
            _, _, predicted_outputs = net(ChIPseq_batch, rnaCount_batch, atac_labels)

            loss = criterion(predicted_outputs, labels.long())
            LossOverall.update(loss.item(), args.batch_size)

            all_targets = np.concatenate((all_targets, labels.cpu().data.numpy()))
            all_preds = np.concatenate((all_preds, torch.index_select(predicted_outputs.cpu(), 1, torch.tensor([1])).view(-1).data.numpy()))

        auc = roc_auc_score_modified(all_targets, all_preds)
        auprc = sklearn.metrics.average_precision_score(all_targets, all_preds)
        print("Batch " + str(i) + ", auprc " + str(auprc))

    #second value of 0.0 is Standard error of the metric's mean, 0.0 for noiseless measurements.
    return auprc#{"auc": (auc, 0.0)}#, "auc": (auc, 0.0)}

#####################################################################################

def evaluate_epoch_merged_both(
    net: nn.Module,
    dl: LoadData,
    data_loader: DataLoader,
    args: Args,
    dtype: torch.dtype,
    device: torch.device,
    criterion: torch.nn
) -> [dict]:
    """
    Compute classification accuracy on provided dataset.
    Changed to evaluate model seperately on same vector and one hot encoding.

    Args:
        Again here RNA_Random == "Siamese" indicating that we are evaluating on 2 different dataset. This does NOT indicate that we are trainng with a siamese network
        net: trained model
        data_loader: DataLoader containing the evaluation set
        args:
        dtype: torch dtype
        device: torch device
        criterion:

    Returns:
        metrics
    """

    print("Evaluate epoch")
    # Evaluation metrics
    if args.withATAC:
        differentTypes = {0:'CL.noCA', 1:'noCL.noCA', 2:'CL.CA', 3:'noCL.CA'}
    else:
        differentTypes = {0:'CL.noCA', 1:'noCL.noCA'}

    type_eval_dict = {}
    all_pred_dict = {}
    auc_dict = {}
    auprc_dict = {}
    metrics_dict = {}
    for key in differentTypes:
        all_pred_dict[key] = []
        auc_dict[key] = []
        metrics_dict[key] = {}
        type_eval_dict[key] = {'LossOverall':AverageMeter(), 'F1ScoreOverall':AverageMeter(),
                                'PrecisionOverall':AverageMeter(),'RecallOverall':AverageMeter(),
                                'AccuracyOverall':AverageMeter()}

    all_targets=[]
    net.eval()
    with torch.no_grad():
        for i, batch_unit in enumerate(data_loader):
            ChIPseq_batch, rnaCount_batch_same, rnaCount_batch_onehot, atac_labels, labels = getData(dl, batch_unit, args, type="valid")
            all_targets = np.concatenate((all_targets, labels.cpu().data.numpy()))

            #We need to try different combinations of with and without each of cell specificty and chromatin accessbility
            for key in differentTypes:
                #CL
                if key == 0 or key == 2: #with CL
                    rnaCount_batch = rnaCount_batch_onehot
                else: #without CL key == 1 or key == 3
                    rnaCount_batch = rnaCount_batch_same
                #Chromatin accessbility
                if args.withATAC and key < 2:
                    atac_labels_mod = torch.zeros((atac_labels.shape[0],2)).to(args.device)
                elif args.withATAC:
                    atac_labels_mod = atac_labels.to(args.device)
                else:
                    atac_labels_mod = atac_labels

                _, _, predicted_outputs = net(ChIPseq_batch, rnaCount_batch, atac_labels_mod)
                loss = criterion(predicted_outputs, labels.long())
                accuracy, F1_score, precision, recall = accuracy_eval(predicted_outputs, labels, args, stage=2)
                type_eval_dict[key]['LossOverall'].update(loss, args.batch_size)
                type_eval_dict[key]['F1ScoreOverall'].update(F1_score, args.batch_size)
                type_eval_dict[key]['AccuracyOverall'].update(accuracy, args.batch_size)
                type_eval_dict[key]['PrecisionOverall'].update(precision, args.batch_size)
                type_eval_dict[key]['RecallOverall'].update(recall, args.batch_size)
                all_pred_dict[key] = np.concatenate((all_pred_dict[key], torch.index_select(predicted_outputs.cpu(), 1, torch.tensor([1])).view(-1).data.numpy()))
                auc_dict[key] = roc_auc_score_modified(all_targets, all_pred_dict[key])
                auprc_dict[key] = sklearn.metrics.average_precision_score(all_targets, all_pred_dict[key])

    for key in metrics_dict:
        metrics_dict[key] = {"loss":type_eval_dict[key]["LossOverall"].avg, "F1Score": type_eval_dict[key]["F1ScoreOverall"].avg,
                            "Accuracy":type_eval_dict[key]["AccuracyOverall"].avg, "Precision":type_eval_dict[key]["PrecisionOverall"].avg,
                            "Recall":type_eval_dict[key]["RecallOverall"].avg, "AUC":auc_dict[key], "AUPRC":auprc_dict[key]}

    return all_targets, all_pred_dict, metrics_dict

#####################################################################################

def getRNAandChIP(args, dl, ChIPseq_batch, labels, atac_labels=None, type=None):
    """
    returns: <'torch.Tensor'>,<'torch.Tensor'>,<'torch.Tensor'>,<'torch.Tensor'>
    Note: Included the atac seq label option in dl.sample_batch_stage2. Will need to modify for others.

    Args:
        args: dictionary containing parameters to be passed to the optimizer.
            - RNA_Random
                    [A] : Generate random vector per CL
                    [B] : Same random vector for all CLs
                    [C] : One hot encoding
                    [D] : Random vector per instance
                    [E] : Idea here is to include to model ideas into one. Include cell specific information
                        in this case thats one hot encoding, as well as general information non specifc cell type information
                    [test_both] : [Returns C and B]
                    [None] : Select rnaseq data
        ChIPseq_batch:
        labels:
        type: training, validating, or testing

    Returns:
        ChIPseq_batch
        rnaCount_batch
        labels
    """
    if args.RNA_Random == "test_both":
        CL_RNA_options, atac_labels, labels = dl.sample_batch_stage2_Siamese(ChIPseq_batch, labels, type, atac_labels)
        rnaCount_batch, rnaCount_batch_g1, rnaCount_batch_g2 = dl.getSumRNA()

        if type == "train":
            rnaCount_batch= rnaCount_batch[CL_RNA_options].T
            try:
                ChIPseq_batch, rnaCount_batch, atac_labels, labels = ChIPseq_batch.to(args.device),torch.from_numpy(rnaCount_batch.values).float().to(args.device), atac_labels.to(args.device), labels.to(args.device)
            except:
                ChIPseq_batch, rnaCount_batch, labels = ChIPseq_batch.to(args.device), torch.from_numpy(rnaCount_batch.values).float().to(args.device), labels.to(args.device)
            return ChIPseq_batch, rnaCount_batch, atac_labels, labels
        elif type == "valid":
            rnaCount_batch_g1, rnaCount_batch_g2=rnaCount_batch_g1[CL_RNA_options].T, rnaCount_batch_g2[CL_RNA_options].T
            try:
                ChIPseq_batch, rnaCount_batch_g1, rnaCount_batch_g2, atac_labels, labels = ChIPseq_batch.to(args.device), torch.from_numpy(rnaCount_batch_g1.values).float().to(args.device), torch.from_numpy(rnaCount_batch_g2.values).float().to(args.device), atac_labels.to(args.device), labels.to(args.device)
            except:
                ChIPseq_batch, rnaCount_batch_g1, rnaCount_batch_g2, labels = ChIPseq_batch.to(args.device), torch.from_numpy(rnaCount_batch_g1.values).float().to(args.device), torch.from_numpy(rnaCount_batch_g2.values).float().to(args.device), labels.to(args.device)
            return ChIPseq_batch, rnaCount_batch_g1, rnaCount_batch_g2, atac_labels, labels
        else:
            print("Type (training or testing) is not determined.")
            sys.exit(1)

    else:  #If RNA_Random = ["A", "B", "C", "E", None] or weighted valid or train
        CL_options, atac_labels, labels, CL_RNA_options_index = dl.sample_batch_stage2(ChIPseq_batch, labels, atac_labels)
        rnaCount_batch = dl.getSumRNA()[CL_options].T

        if args.withATAC:
            ChIPseq_batch, rnaCount_batch, atac_labels, labels = ChIPseq_batch.float().to(args.device), torch.from_numpy(rnaCount_batch.values).float().to(args.device), atac_labels.float().to(args.device), labels.float().to(args.device)
        else:
            ChIPseq_batch, rnaCount_batch, labels = ChIPseq_batch.float().to(args.device), torch.from_numpy(rnaCount_batch.values).float().to(args.device), labels.float().to(args.device)

        return ChIPseq_batch, rnaCount_batch, atac_labels, labels, CL_RNA_options_index

def getData(dl, batch_unit, args, type):
    """
    """
    atac_labels = None
    cl_id = None

    if args.customSampler:
        if args.withATAC:
            (ChIPseq_batch, labels, atac_labels, cl_id) = batch_unit
        else:
            (ChIPseq_batch, labels, cl_id) = batch_unit

        ChIPseq_batch = ChIPseq_batch.view((ChIPseq_batch.shape[0]*ChIPseq_batch.shape[1]),
                                            ChIPseq_batch.shape[2], ChIPseq_batch.shape[3], ChIPseq_batch.shape[4])

        labels = labels.view(-1, args.num_cell_types)
        cl_id = cl_id.cpu().detach().numpy()
        cl_id = np.repeat(cl_id, 2)

        if cl_id.shape[0] != labels.shape[0]:
            print("ERROR:: CL_ID LENGTH DOES NOT EQUAL THE LABEL LENGTH.")
            sys.exit(1)

        if args.withATAC:
            atac_labels = atac_labels.view(-1, args.num_cell_types)

    else:
        if args.withATAC:
            (ChIPseq_batch, labels, atac_labels) = batch_unit
        else:
            (ChIPseq_batch, labels) = batch_unit

    #ChIPseq_batch, rnaCount_batch, atac_labels, labels, CL_RNA_options_index
    return getRNAandChIP(args, dl, ChIPseq_batch, labels, atac_labels, type=type)
