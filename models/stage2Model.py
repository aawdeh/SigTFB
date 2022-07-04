import sys
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable

class Net_Stage2(nn.Module):
    def __init__(self, model_stage1, args, size_conv_out, init_weights=True):
        super(Net_Stage2, self).__init__()
        self.model_stage1 = model_stage1
        self.num_units = args.num_units
        self.neurons_per_layer_pre_concat = args.neurons_per_layer_pre_concat
        self.neurons_per_layer = args.neurons_per_layer
        self.attribution = args.attribution

        # freezing basset_model parameters and setting to eval mode
        if args.freeze_pretrained_model:
            for param in self.model_stage1.parameters():
                param.requires_grad = False
            self.model_stage1.eval()

        self.fc1 = nn.Linear(self.num_units, self.neurons_per_layer_pre_concat)
        self.bn1 = nn.BatchNorm1d(self.neurons_per_layer_pre_concat)
        self.dropout1 = torch.nn.Dropout(p=args.dropout_prob)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(self.neurons_per_layer_pre_concat, 1)
        self.fc3 = nn.Linear(1 + size_conv_out, args.neurons_per_layer)

        self.bn3 = nn.BatchNorm1d(args.neurons_per_layer)
        self.relu3 = nn.ReLU()
        self.dropout3 = torch.nn.Dropout(p=args.dropout_prob)
        self.fc4 = nn.Linear(args.neurons_per_layer, 2)
        self.log_softmax = nn.LogSoftmax(dim=1)
        self.args = args

        if init_weights:
           self._initialize_weights()

    def forward(self, s, g, a=None):
        '''
            s: chipseq sequence one hot encoded
            g: direct encoding of cell type, such that if there is 3 cell types for that specific TF. Then the possbilities include:
                - 1 0 0 CL1
                - 0 1 0 CL2
                - 0 0 1 CL3
                - 0 0 0 Not telling which cell type

        '''
        if self.args.freeze_pretrained_model: self.model_stage1.eval()
        _, conv_out, _ = self.model_stage1(s)
        g = self.dropout1(self.relu1(self.bn1(self.fc1(g))))
        g = self.fc2(g)
        g = torch.cat([conv_out, g], dim = -1)
        fc3_g = self.dropout3(self.relu3(self.bn3(self.fc3(g))))
        fc4_g = self.fc4(fc3_g)
        g = self.log_softmax(fc4_g)
        return fc3_g, fc4_g, g

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
