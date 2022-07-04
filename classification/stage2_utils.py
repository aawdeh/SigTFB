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
path.append("/project/6006657/aaseel/Aim2/Scripts_GithubEdition")
from DNN.misc import AverageMeter
from DNN.evaluationMetrics import accuracy_eval, roc_auc_score_modified
from DNN.loadData import LoadData
from DNN.args import getArgs_Stage2 as Args

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
            ChIPseq_batch, rnaCount_batch, labels, _ = getData(dl, batch_unit, args, type="train")
            optimizer.zero_grad()
            _, _, predicted_outputs = net(ChIPseq_batch, rnaCount_batch)
            loss = criterion(predicted_outputs, labels.long())
            loss.backward()
            optimizer.step()
            print("Epoch " + str(e) + ", Batch " + str(i) + ", loss " + str(loss.mean().item()))
            break
        break
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
            ChIPseq_batch, rnaCount_batch, labels, _ = getData(dl, batch_unit, args, type="valid")
            _, _, predicted_outputs = net(ChIPseq_batch, rnaCount_batch)

            loss = criterion(predicted_outputs, labels.long())
            LossOverall.update(loss.item(), args.batch_size)

            all_targets = np.concatenate((all_targets, labels.cpu().data.numpy()))
            all_preds = np.concatenate((all_preds, torch.index_select(predicted_outputs.cpu(), 1, torch.tensor([1])).view(-1).data.numpy()))
            break
        auc = roc_auc_score_modified(all_targets, all_preds)
        auprc = sklearn.metrics.average_precision_score(all_targets, all_preds)
        print("Batch " + str(i) + ", auprc " + str(auprc))

    return auprc

#####################################################################################
def getRNAandChIP(args, dl, ChIPseq_batch, labels, type=None):
    """
    returns: <'torch.Tensor'>,<'torch.Tensor'>,<'torch.Tensor'>,<'torch.Tensor'>

    Args:
        args: dictionary containing parameters to be passed to the optimizer.
            - Encoding_Random
                    [E] : Idea here is to include to model ideas into one. Include cell specific information
                        in this case thats one hot encoding, as well as general information non specifc cell type information
                    [test_both] : [Returns C and B]
        ChIPseq_batch:
        labels:
        type: training, validating, or testing

    Returns:
        ChIPseq_batch
        rnaCount_batch
        labels
    """
    if args.Encoding_Random == "test_both":
        CL_RNA_options, labels = dl.sample_batch_stage2_Siamese(ChIPseq_batch, labels, type)
        rnaCount_batch, rnaCount_batch_g1, rnaCount_batch_g2 = dl.getSumRNA()

        if type == "train":
            rnaCount_batch= rnaCount_batch[CL_RNA_options].T
            ChIPseq_batch, rnaCount_batch, labels = ChIPseq_batch.to(args.device), torch.from_numpy(rnaCount_batch.values).float().to(args.device), labels.to(args.device)
            return ChIPseq_batch, rnaCount_batch, labels
        elif type == "valid":
            rnaCount_batch_g1, rnaCount_batch_g2 = rnaCount_batch_g1[CL_RNA_options].T, rnaCount_batch_g2[CL_RNA_options].T
            ChIPseq_batch, rnaCount_batch_g1, rnaCount_batch_g2, labels = ChIPseq_batch.to(args.device), torch.from_numpy(rnaCount_batch_g1.values).float().to(args.device), torch.from_numpy(rnaCount_batch_g2.values).float().to(args.device), labels.to(args.device)
            return ChIPseq_batch, rnaCount_batch_g1, rnaCount_batch_g2, labels
        else:
            print("Type (training or testing) is not determined.")
            sys.exit(1)

    else:
        CL_options, labels, CL_RNA_options_index = dl.sample_batch_stage2(ChIPseq_batch, labels)
        rnaCount_batch = dl.getSumRNA()[CL_options].T
        ChIPseq_batch, rnaCount_batch, labels = ChIPseq_batch.float().to(args.device), torch.from_numpy(rnaCount_batch.values).float().to(args.device), labels.float().to(args.device)
        return ChIPseq_batch, rnaCount_batch, labels, CL_RNA_options_index

def getData(dl, batch_unit, args, type):
    """
    """
    cl_id = None

    if args.customSampler:
        (ChIPseq_batch, labels, cl_id) = batch_unit
        ChIPseq_batch = ChIPseq_batch.view((ChIPseq_batch.shape[0]*ChIPseq_batch.shape[1]),
                                            ChIPseq_batch.shape[2], ChIPseq_batch.shape[3], ChIPseq_batch.shape[4])

        labels = labels.view(-1, args.num_cell_types)
        cl_id = cl_id.cpu().detach().numpy()
        cl_id = np.repeat(cl_id, 2)

        if cl_id.shape[0] != labels.shape[0]:
            print("ERROR:: CL_ID LENGTH DOES NOT EQUAL THE LABEL LENGTH.")
            sys.exit(1)

    else:
        (ChIPseq_batch, labels) = batch_unit

    return getRNAandChIP(args, dl, ChIPseq_batch, labels, type=type)
