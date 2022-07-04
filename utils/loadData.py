#!/usr/bin/env python
# coding: utf-8

# In[ ]:

# ------------------------------------
# python modules
# ------------------------------------
import os
import re
import random
import sys
import numpy as np
import h5py
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder
from sklearn.decomposition import PCA
from sklearn.utils import resample
from collections import Counter

import torch
import torch.nn.functional as F
from torch.utils import data
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

####################################################################################################################################
# Load ChIPseq and RNAseq data.
# ChIP files are saved as h5 files -- used to the preprocess_Basset.py method to preprocess the data from bed files to h5 files
####################################################################################################################################
class H5Dataset(data.Dataset):

    def __init__(self, h5_file, Data, Label, customSampler=False):
        super(H5Dataset, self).__init__()
        self.customSampler = customSampler
        self.data = h5_file.get(Data)
        self.target = h5_file.get(Label)

    def __getitem__(self, index):
        if self.customSampler:
            index, cl_id = index
            return (torch.from_numpy(self.data[index,:,:,:]).float(),
                    torch.from_numpy(self.target[index,:]).float(),
                    cl_id)
        else:
            return (torch.from_numpy(self.data[index,:,:,:]).float(),
                    torch.from_numpy(self.target[index,:]).float())

    def __len__(self):
        return self.data.shape[0]

    def getData(self):
        return self.data[:]

    def getTarget(self):
        return self.target[:]

    def setInstances(self, new_data):
        self.data = new_data

    def setTarget(self, new_target):
        self.target =  new_target

    def getStats(self):
        PosperCL = []
        for i in range(self.target.shape[1]):
            PosperCL.append(np.count_nonzero(self.target[:][:,i]))
        return PosperCL

    def getNumPosPerClass(self):
        return np.count_nonzero(self.target[:], axis=0)

class LoadData:
    def __init__(self, chippath_file=None, TF=None, customSampler=None,
                Encoding_Random=None, unique=None, upsampleRows=None):
        '''
        Inputs:
        - chippath_file: path to file that has chipseq data for a specific transcription factor (TF) and antibody (AB) combination (eg. TF.AB.h5)
        - divided: (bool) if yes, means instances are already divided into train, test and valid
        - Encoding_Random:
                - (E): One hot encoding + same random vector ("non specific")
                - (test_both): Return two vectors: One hot encoding & random vector

        - unique: (bool) if true, select unique peaks per CL

        '''
        self.customSampler = customSampler
        self.TF = TF
        self.Encoding_Random = Encoding_Random
        self.unique = unique
        self.upsampleRows = upsampleRows

        ############################################################################
        # load chip
        ############################################################################
        print('LOAD DATA : ChIP DATA CORRESPONDING TO ' + TF)
        self.loadData_ChIP(chippath_file)

        if self.Encoding_Random is not None:
            if self.Encoding_Random == "E" or self.Encoding_Random == "C":
                print("LOAD DATA : DIRECT ENCODING OF CELL TYPE.")
            else:
                print('LOAD DATA : CONCAT RNA USING METHOD ' + self.Encoding_Random)
            self.get_alternative_modified()

        #Here to select unique peaks per CL -- we look at targets
        if self.unique:
            self.selectUniquePeaks()
            #self.upsampleUniquePeaks()

        if self.upsampleRows:
            self.upsamplePeaks()

    def loadData_ChIP(self, chippath_file):
        """
            Data format: basset merged peaks and divided into training/valid and testing
            Load ChIP-seq peaks across different cell lines for a specifc TF and antibody combination. (num_examples, 4, 1, 101)
            ChIPfile is h5 format with keys: ['target_labels', 'test_headers', 'test_in', 'test_out', 'train_in', 'train_out']
            The "training data" can then be further divided into training and validation set.
        """
        h5_file = h5py.File(chippath_file, 'r') #open it once, not every single time

        self.train_set = H5Dataset(h5_file=h5_file, customSampler=self.customSampler,
                                    Data='train_in', Label='train_out')
        self.test_set = H5Dataset(h5_file=h5_file, customSampler=self.customSampler,
                                    Data='test_in', Label='test_out')

        self.target_labels = h5_file['target_labels'] #cell names
        self.test_headers = h5_file['test_headers'] #instance names for the test sequences eg chr1:1-100

        #Stats
        self.numSeqs_train = self.train_set.getData().shape[0] #(109661, 4, 1, 101),
        self.numSeqs_test = self.test_set.getData().shape[0]
        self.seq_length = self.test_set.getData().shape[3]
        self.cellLines = self.target_labels[:].astype('U13')
        self.num_targets = len(self.cellLines)

        print('LOAD DATA :: NUM TRAINING EXAMPLES : '  + str(self.numSeqs_train) + ' x ' + str(self.num_targets))
        print('LOAD DATA :: NUM TESTING EXAMPLES : '  + str(self.numSeqs_test) + ' x ' + str(self.num_targets))

    ###############################################################################
    def mergeCLs(self):
        """
            For a ChIP experiment, there might be two different experiment with the same cell line.
            So here we merge them.
        """

        targets_train = self.train_set.getTarget()
        targets_test = self.test_set.getTarget()
        cellLines_ChIP = self.cellLines
        cls_duplicates = [item for item, count in Counter(cellLines_ChIP).items() if count > 1]

        if len(cls_duplicates) > 0:
            for cl_dup in cls_duplicates:
                dup_idx_list = list(np.where(cellLines_ChIP == cl_dup)[0])
                targets_dup_train = np.logical_or.reduce(targets_train[:,dup_idx_list], axis=1) + 0
                targets_dup_test = np.logical_or.reduce(targets_test[:,dup_idx_list], axis=1) + 0

                cellLines_ChIP =  np.delete(cellLines_ChIP, dup_idx_list)
                targets_train = np.delete(targets_train, dup_idx_list, axis=1)
                targets_test = np.delete(targets_test, dup_idx_list, axis=1)

                #add merged cell line column
                targets_train = np.c_[targets_train,targets_dup_train]
                targets_test = np.c_[targets_test,targets_dup_test]
                cellLines_ChIP = list(cellLines_ChIP) + [cl_dup]

                self.train_set.setTarget(targets_train)
                self.test_set.setTarget(targets_test)
                self.cellLines = cellLines_ChIP
                self.num_targets = len(self.cellLines)

    def selectUniquePeaks(self):
        """
            To train on unique peaks per CL
            Remove rows where duplicates are found

        """
        print("LOAD DATA :: SELECT UNIQUE PEAKS.")
        targets = self.train_set.getTarget()
        instances = self.train_set.getData()
        unique_row, counts = np.unique(targets, axis=0, return_counts=True)

        ids_list = []
        for cl_id in range(len(self.one_hot_dict.keys()) - 1):
            ids_list.extend(list(np.where((targets == self.one_hot_dict[cl_id]).all(axis=1))[0]))
        ids_list.sort()

        self.train_set.setTarget(targets[ids_list])
        self.train_set.setInstances(instances[ids_list])

        targets = self.train_set.getTarget()
        instances = self.train_set.getData()
        unique_row, counts = np.unique(targets, axis=0, return_counts=True)

        print("LOAD DATA :: TOTAL NUM OF UNIQUE TRAINING INSTANCES : " + str(self.train_set.getData().shape[0]) )

    def upsampleUniquePeaks(self):
        """
        """
        print("LOAD DATA :: UPSAMPLE UNIQUE PEAKS.")
        targets = self.train_set.getTarget()
        instances = self.train_set.getData()

        unique_row, counts = np.unique(targets, axis=0, return_counts=True)
        max_count = max(counts)
        max_id = list(counts).index(max_count)
        max_row = unique_row[max_id]

        ids_list = []
        idx_cl = np.where((targets == max_row).all(axis=1))[0]
        ids_list.extend(list(idx_cl))

        for cl_id in range(len(self.one_hot_dict.keys()) - 1):
            if (self.one_hot_dict[cl_id] == max_row).all():
                continue

            idx_cl = np.where((targets == self.one_hot_dict[cl_id]).all(axis=1))[0]
            idx_cl_upsampled = np.random.choice(idx_cl, size=max_count, replace=True)
            ids_list.extend(list(idx_cl_upsampled))

        self.train_set.setTarget(targets[ids_list])
        self.train_set.setInstances(instances[ids_list])
        targets = self.train_set.getTarget()
        instances = self.train_set.getData()

        print("LOAD DATA :: TOTAL NUM OF UNIQUE TRAINING INSTANCES AFTER UPSAMPLING : " + str(self.train_set.getData().shape[0]) )

    def upsamplePeaks(self):
        """
        """
        print("LOAD DATA :: UPSAMPLE PEAKS.")
        targets = self.train_set.getTarget()
        instances = self.train_set.getData()[:]

        unique_row, counts = np.unique(targets, axis=0, return_counts=True)
        max_count = max(counts)
        max_id = list(counts).index(max_count)
        max_row = unique_row[max_id]

        ids_list = []
        idx_cl = np.where((targets == max_row).all(axis=1))[0]
        ids_list.extend(list(idx_cl))

        for cl_id in range(len(unique_row)):
            if (unique_row[cl_id] == max_row).all():
                continue
            idx_cl = np.where((targets == unique_row[cl_id]).all(axis=1))[0]
            idx_cl_upsampled = np.random.choice(idx_cl, size=max_count, replace=True)
            ids_list.extend(list(idx_cl_upsampled))
        ids_list.sort()

        self.train_set.setTarget(targets[ids_list])
        self.train_set.setInstances(instances[ids_list])
        targets = self.train_set.getTarget()
        instances = self.train_set.getData()

        print("LOAD DATA :: TOTAL NUM OF TRAINING INSTANCES AFTER UPSAMPLING : " + str(targets.shape[0]) )
        print("LOAD DATA :: NUM OF TRAINING INSTANCES PER CL AFTER UPSAMPLING : " + str(targets.sum(axis=0)) )

    def _map_type(self, type):
        """
            https://github.com/kundajelab/ChromDragoNN/blob/master/utils/data_iterator.py
        """
        if type == "train":
            return self.rna_training_list
        elif type == "valid":
            return self.rna_validation_list
        elif type == "test":
            return self.rna_test_list

    def onehotCLs(self):
        """
            returns a dict of the one hot encoding of the cell lines
            changed such that were doing only chipseq cls
        """
        print('CONCAT RNA DATA :: ONE HOT ENCODING OF CLS ')
        oneHot_cellLines = torch.eye(len(self.cellLines)).numpy()
        self.one_hot_dict = {}
        self.cl_dict = {}
        for i in range(len(self.cellLines)):
            self.one_hot_dict[i] = oneHot_cellLines[i]
            self.cl_dict[i] = self.cellLines[i]

    def getOneHotDictCL(self):
        return self.one_hot_dict, self.cl_dict

    def setOneHotDictCL(self,cl_onehot_dict, cl_dict, clnames, df_RNA_Summary, df_RNA_Summary_g1, df_RNA_Summary_g2):
        self.df_RNA_Summary = df_RNA_Summary
        self.df_RNA_Summary_g1 = df_RNA_Summary_g1
        self.df_RNA_Summary_g2 = df_RNA_Summary_g2
        self.one_hot_dict = cl_onehot_dict
        self.cl_dict = cl_dict
        self.cellLines = clnames
        self.num_cell_types = len(clnames)

    def sameVectorConcat(self):
        """
            returns a matrix of the same random/all zeros vector of length x (#transcripts, #exons, etc) selected for each CL
        """
        print('CONCAT RNA DATA :: ZERO VECTOR FOR ALL CLS ')
        new_data = [np.zeros(len(self.cellLines))]*len(self.cellLines)
        df_new_matrix = pd.DataFrame(new_data).T
        return df_new_matrix

    def get_alternative_modified(self):
        """
            Call this at the beginning after loading ChIP-seq data.

            We propose alternative methods of concatenation.
            - (E): One hot encoding + same random vector ("non specific")
            - (test_both): Return two vectors: One hot encoding & random vector

        """
        print("LOAD DATA :: GET RNA ALTERNATIVE METHOD : " + str(self.Encoding_Random))
        if self.Encoding_Random == "E":
            #One hot encoding
            self.onehotCLs()
            print("LOAD DATA : COMBINE CELL TYPE SPECIFIC AND CELL TYPE GENERAL IN ONE MODEL.")
            self.one_hot_dict[len(self.cellLines)] = [0]*len(self.cellLines) #Not telling it which CL it is
            self.cl_dict[len(self.cellLines)] = 'NoCL'
            self.df_RNA_Summary = pd.DataFrame(self.one_hot_dict).astype(int)

        elif self.Encoding_Random == "test_both":
            print("test both")
            #for training
            #Normal one hot encoded matrix with extra all zero cell line. Used for training.
            self.onehotCLs()
            print("LOAD DATA : COMBINE CELL TYPE SPECIFIC AND CELL TYPE GENERAL IN ONE MODEL.")
            self.one_hot_dict[len(self.cellLines)] = [0]*len(self.cellLines) #Not telling it which CL it is
            self.cl_dict[len(self.cellLines)] = 'NoCL'
            self.df_RNA_Summary = pd.DataFrame(self.one_hot_dict).astype(int)
            print(self.one_hot_dict)
            print(self.cl_dict)
            print(self.df_RNA_Summary)

            #for evaluation -- Here we will define two df_RNA_Summary matrics
            #returns a one hot encoding + same random vector
            self.df_RNA_Summary_g1 = self.sameVectorConcat()
            self.df_RNA_Summary_g2 = self.df_RNA_Summary.drop([len(self.cellLines)], axis=1)
            print(self.df_RNA_Summary_g1)
            print(self.df_RNA_Summary_g2)

    def sample_batch_stage2(self, ChIPseq_batch, labels):
        """
            ChIPseq_batch: chipseq sequences
            labels: label for each instance per CL [labels are in matrix of size: #INSTANCES * #CL]
            type: training, validation, testing

            Returns cell lines randomly selected according to batchsize -- this represents the cell type encoding part
                and a chipseq dataset with randomly selected sequences according to batchsize

            This method is inspired by ChromDragoNN.
            For a batch size of BATCHSIZE, randomly select sample sequences from original training set and validation set
            for each of the ChIPseq.
        """

        #For each randomly selected instance in training set
        #Number equal to number of peaks in training/validation set, randomly select a chipseq instance from the train/valid
        #Randomly select a cell line for each instance
        CL_RNA_options_index_org = np.random.choice(len(self.cellLines), len(ChIPseq_batch)) #each batch
        labels = labels[np.arange(len(ChIPseq_batch)), CL_RNA_options_index_org]

        #Such that non cell specific vectors are concatenated
        if self.Encoding_Random == "E":
            halfBatch = int(len(ChIPseq_batch)/2)
            CL_RNA_options_index = list(CL_RNA_options_index_org)
            CL_RNA_options_index = CL_RNA_options_index[:halfBatch] + [len(self.cellLines)]*(len(ChIPseq_batch) - halfBatch)

        return CL_RNA_options_index, labels,  CL_RNA_options_index_org

    def sample_batch_stage2_Siamese(self, ChIPseq_batch, labels, type, atac_labels=None):
        """
        """
        CL_RNA_options_index = np.random.choice(len(self.cellLines), len(ChIPseq_batch)) #each batch
        labels = labels[np.arange(len(ChIPseq_batch)), CL_RNA_options_index]

        atac_labels = None

        if type == "train":
            halfBatch = int(len(ChIPseq_batch)/2)
            CL_RNA_options_index = list(CL_RNA_options_index)
            CL_RNA_options_index = CL_RNA_options_index[:halfBatch] + [len(self.cellLines)]*(len(ChIPseq_batch) - halfBatch)

        return CL_RNA_options_index, atac_labels, labels

    def get_train_instances(self):
        return self.instances_train, self.labels_train

###############################################################################################################################################
    def getChIP(self):
        return self.train_set, self.test_set

    def getSumRNA(self):
        #return all, same, one hot
        if self.Encoding_Random == "test_both":
            return self.df_RNA_Summary, self.df_RNA_Summary_g1, self.df_RNA_Summary_g2
        else:
            return self.df_RNA_Summary

    def getCLs_ChIP(self):
        return self.target_labels[:].astype('U13')

    def getNumCellTypes(self):
        return len(self.cellLines)

    def getcellLines(self):
        return self.cellLines

    def getNumExons(self):
        return len(self.df_RNA_Summary)
