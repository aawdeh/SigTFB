import torch
import numpy as np
from sys import path
import os
import sys
import pandas as pd
import inspect
from torch.utils.data import DataLoader, Dataset
from pathlib import Path

# ------------------------------------
# own modules
# ------------------------------------
path.append("/project/6006657/aaseel/Aim2/Scripts")
from AxPytorch.Stage2.ax_utils_stage2 import train
from AxPytorch.Stage1.ax_utils_stage1 import load_data, set_dataLoaders_cv_sampler, set_dataLoaders_sampler
from DNN.utils import load_model_checkpoint, getIndex, torch_seed
from DNN.Models.Stage2.stage2Model import Net_Stage2

#####################################################################################
# set parameters
#####################################################################################
dtype = torch.float

#####################################################################################
# 3. Train on full training set
#####################################################################################
def train_full(args, parameterization, dl, train_loader):
    args.linear_std = parameterization.get('linear_std',1.0)
    args.conv_std = parameterization.get('conv_std',1.0)
    args.num_channels = parameterization.get('num_channels', 16)
    args.batch_size = parameterization.get("batch_size", 64)
    args.num_hidden_layers = parameterization.get("num_hidden_layers", 1)
    args.neurons_per_layer = parameterization.get("neurons_per_layer", 1)
    args.neurons_per_layer_pre_concat = parameterization.get("neurons_per_layer_pre_concat", 1)
    args.dropout_prob = parameterization.get("dropout_prob", 0)
    args.freeze_pretrained_model = parameterization.get("freeze_pretrained_model", False)

    print("Ax Train :: Stage 2 :: Get pretrained stage 1 model.")
    checkpoint_stage1, model_stage1, _ = load_model_checkpoint(args.model_stage1_path, 'cpu')

    net = Net_Stage2(model_stage1, args, checkpoint_stage1['size_conv_out']).to(args.device)
    net.model_stage1.load_state_dict(checkpoint_stage1['state_dict'])
    model_stage1 = model_stage1.to(args.device)
    print("Ax Train :: Stage 2 :: Train on data of length " + str(len(train_loader)) + ".")
    model = train(net=net, dl=dl, train_loader=train_loader, parameters=parameterization, args=args, dtype=dtype,
                    device=args.device)

    return model, args

#####################################################################################
# 3. Save model
#####################################################################################
def save_model(model, args, type):
    """
        type could be using all training set or using part of training set
        "AllTrain" or "PartTrain"
    """
    print("Ax Train :: Stage 2 :: Save model for run " + str(args.trial_index) + ".")
    print(args)
    state = {'state_dict': model.state_dict(),
             'args': args,
             'model': model}

    Path(os.path.join(args.saveModelPath, type, args.TF, args.AB)).mkdir(parents=True, exist_ok=True)
    torch.save(state, os.path.join(args.saveModelPath, type, args.TF, args.AB, args.TF + "." + args.AB + "." + str(int(args.trial_index)) +'.pth.tar'))

#####################################################################################
# 4. Train and save
#####################################################################################
def saveModels(args, parameterization):
    """
    """
    print("Ax Train :: Stage 2 :: Save model")
    torch_seed(12345)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.device = device

    print("Ax Train :: Stage 2 :: Set DataLoaders best hyperparameters.")
    args, dl, dataset, test_set = load_data(args)
    train_idx, validate_idx, train_loader, valid_loader, test_loader = set_dataLoaders_sampler(dataset, test_set, args)
    model_part, args = train_full(args, parameterization, dl, train_loader)
    save_model(model_part, args, "PartTrain")
