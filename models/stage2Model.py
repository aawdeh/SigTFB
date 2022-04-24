import sys
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable

class NetModified_OneLayer_OneUnit(nn.Module):
    def __init__(self, model_stage1, args, size_conv_out, init_weights=True):
        super(NetModified_OneLayer_OneUnit, self).__init__()
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

        if args.withATAC:
            self.fc3 = nn.Linear(1 + 2 + size_conv_out, args.neurons_per_layer)
        else:
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
            a: bits corresponding to chromatin accessbility. [1,0] -- no peak, [0,1] -- yes peak, [0,0] not telling

        '''
        if self.args.freeze_pretrained_model: self.model_stage1.eval()
        _, conv_out, _ = self.model_stage1(s)
        g = self.dropout1(self.relu1(self.bn1(self.fc1(g))))
        g = self.fc2(g)

        #account for chromatin accountibility + direct encoding of cell line type
        if self.args.withATAC:
            g = torch.cat([conv_out, a, g], dim = -1)
        else:
            g = torch.cat([conv_out, g], dim = -1)

        fc3_g = self.dropout3(self.relu3(self.bn3(self.fc3(g))))
        fc4_g = self.fc4(fc3_g)
        g = self.log_softmax(fc4_g)

        # if not self.attribution:
        #     g = self.log_softmax(g)

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

#Inspired by https://github.com/smaslova/AI-TAC/blob/2b6b4b0e3a0a2676c5b93f12311912f85b29c95d/code/aitac.py#L256
# The idea here is to get predictions per convolutional filter per instance when its dropped.
# We get args.num_channels predictions per instance.

class motif_NetModified_OneLayer_OneUnit(nn.Module):
    def __init__(self, original_model, args):
        super(motif_NetModified_OneLayer_OneUnit, self).__init__()

        self.args = args
        self.model_stage1 = list(original_model.children())[0]

        #set s1 parameters
        self.model_stage1.conv1 = list(list(original_model.children())[0].children())[0]
        self.model_stage1.relu1 = list(list(original_model.children())[0].children())[1]
        self.model_stage1.relu2 = list(list(original_model.children())[0].children())[2]
        self.model_stage1.maxpool1 = list(list(original_model.children())[0].children())[3]
        self.model_stage1.fc1 = list(list(original_model.children())[0].children())[4]
        self.model_stage1.fc2 = list(list(original_model.children())[0].children())[5]
        self.model_stage1.sigmoid = list(list(original_model.children())[0].children())[6]

        #set s2 parameters
        self.fc1 = list(original_model.children())[1]
        self.bn1 = list(original_model.children())[2]
        self.dropout1 = list(original_model.children())[3]
        self.relu1 = list(original_model.children())[4]
        self.fc2 = list(original_model.children())[5]
        self.fc3 = list(original_model.children())[6]
        self.bn3 = list(original_model.children())[7]
        self.relu3 = list(original_model.children())[8]
        self.dropout3 = list(original_model.children())[9]
        self.fc4 = list(original_model.children())[10]
        self.log_softmax = list(original_model.children())[11]

    def forward(self, s, g, a=None):
        #run for stage 1 to get filters from stage 1
        out = self.model_stage1.relu1(self.model_stage1.conv1(s))
        layer1_activations = torch.squeeze(out,dim=2)
        layer1_out = self.model_stage1.maxpool1(out)

        #calculate average activation by filter for the whole batch
        filter_means_batch = layer1_activations.mean(0).mean(1)

        #run all other layers with 1 filter left out at a time
        batch_size = layer1_out.shape[0]
        predictions = torch.zeros(batch_size, self.args.num_channels_s1, 2)

        #set gene expression -- from stage 2
        g = self.dropout1(self.relu1(self.bn1(self.fc1(g))))
        g = self.fc2(g)

        for i in range(self.args.num_channels_s1):
            #modify filter i of first layer output
            filter_input = layer1_out.clone()
            filter_input[:,i,:,:] = filter_input.new_full((batch_size, 1, 1), fill_value=filter_means_batch[i].item())

            #run stage 1
            new_size = filter_input.size(1) * filter_input.size(2) * filter_input.size(3)
            out = filter_input.view(-1, new_size)
            conv_out = out

            #run stage 2
            #account for chromatin accountibility and/or direct encoding of cell line type
            if self.args.withATAC:
                out_s2 = torch.cat([conv_out, a, g], dim = -1)
            else:
                out_s2 = torch.cat([conv_out, g], dim = -1)

            out_s2 = self.dropout3(self.relu3(self.bn3(self.fc3(out_s2))))
            out_s2 = self.fc4(out_s2)
            #out_s2 = self.log_softmax(out_s2)

            predictions[:,i,:] = out_s2
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


class NetModified_DynLayer(nn.Module):
    def __init__(self, model_stage1, args, size_conv_out, init_weights=True):
        super(NetModified_DynLayer, self).__init__()
        self.args = args
        self.model_stage1 = model_stage1
        self.num_units = args.num_units

        # freezing parameters and setting to eval mode
        if args.freeze_pretrained_model:
            for param in self.model_stage1.parameters():
                param.requires_grad = False
            self.model_stage1.eval()

        self.bn = nn.BatchNorm1d(args.neurons_per_layer)
        self.dropout = torch.nn.Dropout(p=args.dropout_prob)
        self.relu1 = nn.ReLU()

        self.hidden = nn.ModuleList()
        for i  in range(args.num_hidden_layers):
            if i == 0:
                if args.mergeRNA == "F": #one_hot
                    self.hidden.append(nn.Linear(args.num_cell_types + size_conv_out, args.neurons_per_layer))
                else:
                    self.hidden.append(nn.Linear(args.num_units + size_conv_out, args.neurons_per_layer))
            else:
                self.hidden.append(nn.Linear(args.neurons_per_layer, args.neurons_per_layer))

        self.fc_output = nn.Linear(args.neurons_per_layer, 2)
        self.log_softmax = nn.LogSoftmax(dim=1)

        if init_weights:
           self._initialize_weights()

    def forward(self, s, g):
        if self.args.freeze_pretrained_model: self.model_stage1.eval()
        _, conv_out, _ = self.model_stage1(s)
        g = torch.cat([conv_out, g], dim = -1)

        for layer in self.hidden:
            g = self.dropout(F.relu(self.bn(layer(g)))) #g = self.dropout1(self.relu1(self.bn1(self.fc1(g))))

        g = self.fc_output(g)
        g = self.log_softmax(g)
        return g

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
