import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset, Subset
from torch.utils.data import WeightedRandomSampler

import sys
from sys import path
import os
from typing import Dict, List, Optional, Tuple
from sklearn.model_selection import train_test_split
import sklearn.metrics
import numpy as np
import random
import pandas as pd
from sklearn.metrics import average_precision_score

# ------------------------------------
# own modules
# ------------------------------------
from utils.misc import AverageMeter
from utils.evaluationMetrics import mse_eval, accuracy_eval, roc_auc_score_modified, multilabel_accuracy, calculate_accuracy
from utils.loadData import LoadData, H5Dataset
from utils.utils import load_model_checkpoint, basset_loss as criterion_basset
from utils.args import getArgs_Stage1 as Args
from utils.fasta_dinucleotide_shuffle import dinuclShuffle
from utils.sampler_multilablel_modified import MultilabelBalancedRandomSampler as MultilabelBalancedRandomSamplerCustom

#####################################################################################
# load_data
# set_dataLoaders
# set_dataLoaders_cv
# train
# evaluate
#####################################################################################
def load_data(
        args: Args,
) -> Tuple[Args, LoadData, H5Dataset, H5Dataset]:
    """
        Load chip and rna data and split data into training, validation, and test sets.

        Args:
            args: parameters

        Returns:
            args: modified parameters
            dl: LoadData object with all information
            DataLoader: training data
            DataLoader: validation data
            DataLoader: test data

            Assume typeChIP == "two" or "three"
    """
    print("LOAD DATA")
    dl = LoadData(chippath_file=args.chipData, rnaseq_path=args.rnaData, TF=args.TF, typeChIP=args.typeChIP,
                typeRNA=args.typeRNA, RNA_Random=args.RNA_Random, withATAC=args.withATAC, withRNA=args.withRNA,
                unique=args.unique, upsampleRows=args.upsampleRows,customSampler=args.customSampler)

    args.num_cell_types = dl.getNumCellTypes()
    if args.RNA_Random == "test_both" or args.RNA_Random == "E":
        args.num_units = dl.getNumCellTypes()
    else:
        args.num_units = dl.getNumExons()

    dataset, test_set = dl.getChIP()

    return args, dl, dataset, test_set

def set_dataLoaders_cv_sampler(
    dataset : H5Dataset,
    args : Args,
    train_idx : list,
    validate_idx : list
) -> Tuple[DataLoader, DataLoader]:
    """
        Splits the training set into valid and train.
        Args:
            data_set :
            args : dictionary of parameters
            train_index: indices that can be used to generate the train dataset according to 10 fold cv splits
            test_index: indices that can be used to generate the test dataset according to 10 fold cv splits
        Returns:
            DataLoader: training data
            DataLoader: validation data
    """
    instances = dataset.getData()
    targets = dataset.getTarget()

    if args.withATAC:
        atacseq = dataset.getATACseq()
        atacseq = np.where(atacseq>1, 1, atacseq)
        dataset.setATACseq(atacseq)

    if args.customSampler:
        train_sampler = MultilabelBalancedRandomSamplerCustom(targets, train_idx)
        validate_sampler = MultilabelBalancedRandomSamplerCustom(targets, validate_idx)
    else:
        if args.withATAC:
            X_train, X_val, y_train, y_val, atac_train, atac_val = instances[train_idx], instances[validate_idx], targets[train_idx], targets[validate_idx], atacseq[train_idx], atacseq[validate_idx]
            train_set = torch.utils.data.TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train), torch.from_numpy(atac_train))
            valid_set = torch.utils.data.TensorDataset(torch.from_numpy(X_val), torch.from_numpy(y_val), torch.from_numpy(atac_val))
        else:
            X_train, X_val, y_train, y_val = instances[train_idx], instances[validate_idx], targets[train_idx], targets[validate_idx]
            train_set = torch.utils.data.TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train))
            valid_set = torch.utils.data.TensorDataset(torch.from_numpy(X_val), torch.from_numpy(y_val))

    if args.customSampler:
        train_loader = DataLoader(dataset=dataset, batch_size=args.batch_size, sampler=train_sampler)
        valid_loader = DataLoader(dataset=dataset, batch_size=args.batch_size, sampler=validate_sampler)
    else:
        train_loader = DataLoader(dataset=train_set, batch_size=args.batch_size, shuffle=True, drop_last=True)
        valid_loader = DataLoader(dataset=valid_set, batch_size=args.batch_size, shuffle=True, drop_last=True)

    return train_loader, valid_loader

def set_dataLoaders_sampler(
    dataset : H5Dataset,
    test_set : H5Dataset,
    args : Args,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
        Here we are using the sampler_multilablel sampler
        https://github.com/issamemari/pytorch-multilabel-balanced-sampler/blob/b11b7fa0d0334795bdcc600812d1edbe2ce7ea7b/sampler.py
    """
    instances = dataset.getData()
    targets = dataset.getTarget()

    if args.withATAC:
        atacseq = dataset.getATACseq()[:]
        atacseq = np.where(atacseq>1, 1, atacseq)
        dataset.setATACseq(atacseq)

    train_sampler, validate_sampler, test_sampler = None, None, None
    train_idx, validate_idx = None, None

    if args.typeChIP == "two":
        print(args.typeChIP)

        val_size=0.2
        indices = list(range(len(dataset)))
        np.random.shuffle(indices)
        split = int(np.floor(val_size * len(dataset)))
        train_idx, validate_idx = indices[split:], indices[:split]

        if args.customSampler:
            train_sampler = MultilabelBalancedRandomSamplerCustom(targets, train_idx)
            validate_sampler = MultilabelBalancedRandomSamplerCustom(targets, validate_idx)
        else:
            if args.withATAC:
                X_train, X_val, y_train, y_val, atac_train, atac_val = instances[train_idx], instances[validate_idx], targets[train_idx], targets[validate_idx], atacseq[train_idx], atacseq[validate_idx]
                train_set = torch.utils.data.TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train), torch.from_numpy(atac_train))
                valid_set = torch.utils.data.TensorDataset(torch.from_numpy(X_val), torch.from_numpy(y_val), torch.from_numpy(atac_val))
            else:
                X_train, X_val, y_train, y_val = instances[train_idx], instances[validate_idx], targets[train_idx], targets[validate_idx]
                train_set = torch.utils.data.TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train))
                valid_set = torch.utils.data.TensorDataset(torch.from_numpy(X_val), torch.from_numpy(y_val))

        if args.customSampler or args.balancedSampler:
            train_loader = DataLoader(dataset=dataset, batch_size=args.batch_size, sampler=train_sampler)
            valid_loader = DataLoader(dataset=dataset, batch_size=args.batch_size, sampler=validate_sampler)
        else:
            train_loader = DataLoader(dataset=train_set, batch_size=args.batch_size, shuffle=True, drop_last=True)
            valid_loader = DataLoader(dataset=valid_set, batch_size=args.batch_size, shuffle=True, drop_last=True)

    elif args.typeChIP == "three":
        if args.customSampler:
            train_sampler = MultilabelBalancedRandomSamplerCustom(targets)
        train_loader = DataLoader(dataset=dataset, batch_size=args.batch_size, sampler=train_sampler)
        valid_loader = None

    if args.customSampler:
        test_sampler = MultilabelBalancedRandomSamplerCustom(test_set.getTarget())
    test_loader = DataLoader(dataset=test_set, batch_size=args.batch_size, sampler=test_sampler)

    return train_idx, validate_idx, train_loader, valid_loader, test_loader

def getFasta(ChIPseq_batch):
    """
    """
    dict_fasta = {0:"A", 1:"C", 2:"G", 3:"T"}
    instance_max = np.argmax(ChIPseq_batch, axis=1).squeeze(1)
    batch_fasta = []
    for instance in instance_max:
        fasta_instance = ""
        for pos in instance:
            fasta_instance += dict_fasta.get(pos.item())
        batch_fasta.append(fasta_instance)
    return batch_fasta

def convert2onehot(batch):
    """
        convert to one hot
    """
    onehot_dict = {"A": [1.0, 0.0, 0.0, 0.0], "C": [0.0, 1.0, 0.0, 0.0],
                   "G": [0.0, 0.0, 1.0, 0.0], "T": [0.0, 0.0, 0.0, 1.0]}
    batch_onehot = []
    for instance in batch:
        onehot_instance = []
        for pos in instance:
            onehot_instance.append(onehot_dict.get(pos))
        batch_onehot.append(np.array(onehot_instance).transpose())
    batch_onehot =  torch.tensor(np.expand_dims(np.array(batch_onehot), axis=2)).float()
    return batch_onehot

def reverseBatch(fasta_batch):
    """
        needs fixing -- testing out code
    """
    reverse_dict = {"A": "T", "T": "A", "C":"G", "G":"C"}
    batch_reverse = []
    for fasta_instance in fasta_batch:
        batch_reverse.append("".join([reverse_dict.get(nuc) for nuc in fasta_instance])[::-1])
    batch_reverse_onehot = convert2onehot(batch_reverse)
    return batch_reverse_onehot

def dinucleotideShuffle(batch_fasta):
    """
        return dinucleotide shuffle
    """
    dinuc_list = []
    for instance in batch_fasta:
        dinuc_list.append(dinuclShuffle(instance))
    dinuc_batch = convert2onehot(np.array(dinuc_list))
    return dinuc_batch

def generatebatchpercl(ChIPseq_batch, labels, batch_size, cl_id):
    """
        Randomly select instances per cell line per batch
    """
    pos_ids = np.where(labels[:,cl_id] == 1)[0]
    neg_ids = np.where(labels[:,cl_id] == 0)[0]
    random_pos = np.random.choice(pos_ids, batch_size//2)
    random_neg = np.random.choice(neg_ids, batch_size - batch_size//2)
    ChIP_pos = ChIPseq_batch[random_pos]
    ChIP_neg = ChIPseq_batch[random_neg]
    labels_pos = labels[random_pos, cl_id]
    labels_neg = labels[random_neg, cl_id]
    ChIPseq_batch_cl = torch.cat((ChIP_pos, ChIP_neg), 0)
    labels_cl = torch.cat((labels_pos, labels_neg), 0)
    labels_cl = torch.unsqueeze(labels_cl, 1)
    return ChIPseq_batch_cl, labels_cl

def train(
    net: torch.nn.Module,
    train_loader: DataLoader,
    parameters: Dict[str, float],
    args : Args,
    dtype: torch.dtype,
    device: torch.device,
) -> nn.Module:
    """
        Train CNN on data

        Args:
            net: initialized neural network for stage 2
            train_loader: DataLoader containing training set
            parameters: dictionary containing parameters to be passed to the optimizer.
                - lr: default (0.001)
                - weight decay: default (0.0)
                - motif_std: default (1.0)
                - linear_std: default (1.0)
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

    # Define loss and optimizer -- those used for stage 2 are different?
    optimizer = optim.Adam(
           net.parameters(),
           lr = parameters.get("learning_rate", 0.001),
           weight_decay = parameters.get("weight_decay", 0.0))

    num_epochs = parameters.get("num_epochs", 1)

    for e in range(num_epochs):
        for i, batch_unit in enumerate(train_loader):
            if args.customSampler:
                if args.withATAC:
                    #what were trying to do here is predict accessbility instead of binding
                    (ChIPseq_batch, labels, atac_labels, cl_id) = batch_unit
                    labels = atac_labels
                else:
                    (ChIPseq_batch, labels, cl_id) = batch_unit

                ChIPseq_batch = ChIPseq_batch.view((ChIPseq_batch.shape[0]*ChIPseq_batch.shape[1]),
                                                    ChIPseq_batch.shape[2], ChIPseq_batch.shape[3], ChIPseq_batch.shape[4])
                labels = labels.view(-1, args.num_cell_types)
                cl_id = cl_id.cpu().detach().numpy()
                cl_id = np.repeat(cl_id, 2)
                if cl_id.shape[0] != labels.shape[0]:
                    print("ERROR :: CL_ID LENGTH DOES NOT EQUAL THE LABEL LENGTH.")
                    sys.exit(1)
            else:
                (ChIPseq_batch, labels) = batch_unit
                cl_id = None

            ChIPseq_batch, labels = ChIPseq_batch.to(device), labels.to(device)
            ChIPseq_batch, labels = ChIPseq_batch.float(), labels.float()
            optimizer.zero_grad()

            if args.rc or args.shuffle:
                fasta_instances = getFasta(ChIPseq_batch)

            if args.shuffle:
                dinuclShuffle_batch = dinucleotideShuffle(fasta_instances)
                dinuclShuffle_labels = torch.zeros(labels.shape[0], labels.shape[1])
                labels = torch.cat((labels, dinuclShuffle_labels), 0)
                ChIPseq_batch = torch.cat((ChIPseq_batch, dinuclShuffle_batch),0)

            if args.rc:
                ChIPseq_batch_rc = reverseBatch(fasta_instances)
                predicted_outputs, conv_out1, conv_out2, new_size = net(ChIPseq_batch, ChIPseq_batch_rc)
            else:
                predicted_outputs, conv_out, new_size = net(ChIPseq_batch)

            if args.lossFunction == "BASSET":
                loss = args.criterion(outputs=predicted_outputs, targets=labels, args=args, stage=1)
            else:
                loss = args.criterion(predicted_outputs, labels)

            #using a customer sampler weight loss depending on which class were looking at
            if args.customSampler_OneLoss:
                loss = loss[range(len(cl_id)), cl_id]
                loss = -loss.sum() / labels.size()[0]

            #check backward -- if working correctly or not -- loss
            loss.backward()
            optimizer.step()
            print("Epoch " + str(e) + ", Batch " + str(i) + ", loss " + str(loss.item()))

    return net, new_size

def evaluate(
    net: nn.Module,
    data_loader: DataLoader,
    args: Args,
    dtype: torch.dtype,
    device: torch.device
) -> float:
    """
    Compute classification accuracy on provided dataset.

    Args:
        net: trained model
        data_loader: DataLoader containing the evaluation set
        dtype: torch dtype
        device: torch device

    Returns:
        dict: evalution metrics metrics
    """
    all_preds = []
    all_targets = []

    net.eval()
    with torch.no_grad():
        for i, batch_unit in enumerate(data_loader):
            if args.customSampler:
                if args.withATAC:
                    #what were trying to do here is predict accessbility instead of binding
                    (ChIPseq_batch, labels, atac_labels, cl_id) = batch_unit
                    labels = atac_labels
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
            else:
                (ChIPseq_batch, labels) = batch_unit
                cl_id = None

            ChIPseq_batch, labels = ChIPseq_batch.to(device), labels.to(device)
            ChIPseq_batch, labels = ChIPseq_batch.float(), labels.float()

            if args.rc or args.shuffle:
                fasta_instances = getFasta(ChIPseq_batch)

            if args.shuffle:
                dinuclShuffle_batch = dinucleotideShuffle(fasta_instances)
                dinuclShuffle_labels = torch.zeros(labels.shape[0], labels.shape[1])
                labels = torch.cat((labels, dinuclShuffle_labels), 0)
                ChIPseq_batch = torch.cat((ChIPseq_batch, dinuclShuffle_batch),0)

            if args.rc:
                ChIPseq_batch_rc = reverseBatch(fasta_instances)
                predicted_outputs, _, _, _ = net(ChIPseq_batch, ChIPseq_batch_rc)
            else:
                predicted_outputs, _, _ = net(ChIPseq_batch)

            if args.lossFunction == "BASSET":
                loss = args.criterion(outputs=predicted_outputs, targets=labels, args=args, stage=1)
            else:
                loss = args.criterion(predicted_outputs, labels)

            #using a customer sampler weight loss depending on which class were looking at
            if args.customSampler_OneLoss:
                loss = loss[range(len(cl_id)), cl_id]
                loss = -loss.sum() / labels.size()[0]
                predicted_outputs = predicted_outputs[range(len(cl_id)), cl_id]
                labels = labels[range(len(cl_id)), cl_id]

            all_preds.append(predicted_outputs.cpu().data.numpy())
            all_targets.append(labels.cpu().data.numpy())

        print("Batch " + str(i) + ", loss " + str(loss.item()))

    all_targets = np.concatenate(all_targets, axis=0)
    all_preds = np.concatenate(all_preds, axis=0)

    if args.customSampler_OneLoss:
        accuracy = calculate_accuracy(torch.from_numpy(all_preds), torch.from_numpy(all_targets))
        return accuracy

    Hamming_Score_, Exact_Match_Ratio = multilabel_accuracy(torch.from_numpy(all_preds), torch.from_numpy(all_targets))
    auc = np.mean([roc_auc_score_modified(all_targets[:,i], all_preds[:,i]) for i in range(all_preds.shape[1])])
    auprc = np.mean([average_precision_score(all_targets[:,i], all_preds[:,i]) for i in range(all_preds.shape[1])])
    return Hamming_Score_
