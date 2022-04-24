import random
import datetime
import time
import sys

import random
import numpy as np
import os
import math

import torch
import torch.nn.functional as F
import torchvision
from torch import nn
from torch.utils.data import DataLoader

import torchvision.datasets as datasets
import torchvision.transforms as transforms

"""
    Simple CNN model to test ChIP-seq data with. -- DeepBind model
"""

class CNN_Multilabel(nn.Module):
    def __init__(self, args, init_weights=True):
        super(CNN_Multilabel, self).__init__()

        output_shape_layer1 = int((args.seq_length - 24)/1 + 1) #int((101 - 24)/1 + 1)
        self.args = args

        self.conv1 = nn.Conv2d(in_channels=4, out_channels=args.num_channels, kernel_size=(1,24), stride=1)
        self.relu1 = nn.ReLU()
        self.relu2 = nn.ReLU()
        self.maxpool1 = nn.MaxPool2d(kernel_size=(1,output_shape_layer1), stride=1)
        self.fc1 = nn.Linear(args.num_channels, 32)
        self.fc2 = nn.Linear(32, args.num_cell_types)
        self.sigmoid = nn.Sigmoid()

        if init_weights:
           self._initialize_weights()

    def forward(self, x):
        #torch.Size([batch_size, 4, 1, seq_length])
        out = self.maxpool1(self.relu1(self.conv1(x)))
        #considering a tensor with 4 dimensions
        new_size = out.size(1) * out.size(2) * out.size(3)
        out = out.view(-1, new_size)
        conv_out = out
        out = self.relu2(self.fc1(out))
        out = self.fc2(out)

        if self.args.lossFunction == "BCE":
            out = self.sigmoid(out)

        return out, conv_out, new_size

    def _initialize_weights(self):
        """
            Intializing weights.
            - https://github.com/pytorch/vision/blob/master/torchvision/models/vgg.py#L46-L59
        """
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, mean=0, std=self.args.conv_std)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0, std=self.args.linear_std)
                nn.init.constant_(m.bias, 0)

#Inspired by https://github.com/smaslova/AI-TAC/blob/2b6b4b0e3a0a2676c5b93f12311912f85b29c95d/code/aitac.py#L256
class motif_CNN_Multilabel(nn.Module):
    def __init__(self, original_model, args):
        super(motif_CNN_Multilabel, self).__init__()

        self.args = args
        self.conv1 = list(original_model.children())[0]
        self.relu1 = list(original_model.children())[1]
        self.relu2 = list(original_model.children())[2]
        self.maxpool1 = list(original_model.children())[3]
        self.fc1 = list(original_model.children())[4]
        self.fc2 = list(original_model.children())[5]
        self.sigmoid = list(original_model.children())[6]

    def forward(self, x):
        print("Forward")
        out = self.relu1(self.conv1(x))
        layer1_activations = torch.squeeze(out)
        layer1_out = self.maxpool1(out)

        #calculate average activation by filter for the whole batch
        filter_means_batch = layer1_activations.mean(0).mean(1)

        #run all other layers with 1 filter left out at a time
        batch_size = layer1_out.shape[0]
        predictions = torch.zeros(batch_size, self.args.num_channels, self.args.num_cell_types)

        print("Loop through filters")
        for i in range(self.args.num_channels):
            #modify filter i of first layer output
            filter_input = layer1_out.clone()
            filter_input[:,i,:,:] = filter_input.new_full((batch_size, 1, 1), fill_value=filter_means_batch[i].item())

            new_size = filter_input.size(1) * filter_input.size(2) * filter_input.size(3)
            out = filter_input.view(-1, new_size)
            #conv_out = out
            out = self.relu2(self.fc1(out))
            out = self.fc2(out)

            predictions[:,i,:] = out
            activations, act_index = torch.max(layer1_activations, dim=2)

        return predictions, layer1_activations, act_index

    def _initialize_weights(self):
        """
            Intializing weights.
            - https://github.com/pytorch/vision/blob/master/torchvision/models/vgg.py#L46-L59
        """
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, mean=0, std=self.args.conv_std)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0, std=self.args.linear_std)
                nn.init.constant_(m.bias, 0)
