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

    def __init__(self, h5_file, withATAC, Data, Label, customSampler=False, ATACLabel=None):
        super(H5Dataset, self).__init__()
        self.withATAC = withATAC
        self.customSampler = customSampler
        self.data = h5_file.get(Data)
        self.target = h5_file.get(Label)
        if self.withATAC:
            self.atac_seq = h5_file.get(ATACLabel)

    def __getitem__(self, index):
        if self.customSampler:
            index, cl_id = index
            if self.withATAC:
                return (torch.from_numpy(self.data[index,:,:,:]).float(),
                        torch.from_numpy(self.target[index,:]).float(),
                        torch.from_numpy(self.atac_seq[index,:]).float(),
                        cl_id)
            else:
                return (torch.from_numpy(self.data[index,:,:,:]).float(),
                        torch.from_numpy(self.target[index,:]).float(),
                        cl_id)
        else:
            if self.withATAC:
                return (torch.from_numpy(self.data[index,:,:,:]).float(),
                        torch.from_numpy(self.target[index,:]).float(),
                        torch.from_numpy(self.atac_seq[index,:]).float())
            else:
                return (torch.from_numpy(self.data[index,:,:,:]).float(),
                        torch.from_numpy(self.target[index,:]).float())

    def __len__(self):
        return self.data.shape[0]

    def getData(self):
        return self.data[:]

    def getTarget(self):
        return self.target[:]

    def getATACseq(self):
        return self.atac_seq

    def setInstances(self, new_data):
        self.data = new_data

    def setATACseq(self, new_atac):
        self.atac_seq = new_atac

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
    def __init__(self, chippath_file=None, rnaseq_path=None, TF=None, typeChIP="one", typeRNA="gene_id", customSampler=None,
                RNA_Random=None, withATAC=None, withRNA=None, unique=None, upsampleRows=None):
        '''
        Inputs: chippath_file, rnapath_file
        - chippath_file: path to file that has chipseq data for a specific transcription factor (TF) and antibody (AB) combination (eg. TF.AB.h5)
        - rnapath_file: path to file that has (num_cell_types x num_genes) gene expression data -- fix: chromdragonn
        - divided: (bool) if yes, means instances are already divided into train, test and valid
        - typeChIP: (str) indicates how the ChIP was divided: train/valid/test or train/test or full
        - typeRNA: (str) "gene_id", "transcript_id" or "exon_id" level -- generated using RSEM
        - mergeRNA: (str) for each cell line, there can be multiple RNAseq experiments and for each of these experiemnts there are biological replicates
                - "A": Average across datasets for that cell line.
                - "B": Randomly select an rna experiment from above list -- use the average
                - "C": Randomly select an rna seq dataset from the above list of files.
                - "D": PCA
                - "E": Take all experiments into account
                - "F": One hot encoding of the cell lines added to final convolutional layer

        - RNArandom: (str) Always set with mergeRNA == "G"
                - (A): Generate random vector per CL.
                - (B): Same random vector for all CLs.
                - (C): One hot encoding.
                - (D): Random vector per instance.
                - (E): One hot encoding + same random vector ("non specific")
                - (test_both): Return two vectors: One hot encoding & random vector

        - withATAC: (bool) if true, atacseq used. Chromatin accessbility per ChIPseq used as a feature
        - withRNA: (bool) if true, atacseq used. RNA per CL per ChIPseq used as a feature
        - unique: (bool) if true, select unique peaks per CL

        '''

        self.withATAC = withATAC
        self.customSampler = customSampler
        self.typeChIP = typeChIP
        self.typeRNA = typeRNA
        self.TF = TF
        self.RNA_Random = RNA_Random
        self.withRNA = withRNA
        self.unique = unique
        self.upsampleRows = upsampleRows

        ############################################################################
        # load chip
        ############################################################################
        print('LOAD DATA : ChIP DATA CORRESPONDING TO ' + TF)
        self.loadData_ChIP(chippath_file, typeChIP)
        ############################################################################

        ############################################################################
        # if we add rna
        ############################################################################
        if self.withRNA:
            print('LOAD DATA : RNA DATA CORRESPONDING TO ' + TF + ' ' + typeRNA)
            self.loadData_RNA(rnaseq_path, TF, typeRNA)

            print('LOAD DATA : COMMON CLS BETWEEN RNA AND CHIP')
            self.cellLines = self.getCLs_Common()
            print('LOAD DATA : THERE ARE ' + str(len(self.cellLines)) + ' COMMON CLS.')

            print('LOAD DATA : REMOVE DATA CORRESPONDING TO CLS NOT FOUND IN BOTH RNA AND CHIP FOR TF.')
            self.removeTargets_ChIP() #remove targets corresponding to uncommon CLs

            #We test different methods of concatenation
            print('LOAD DATA : CONCAT RNA :: TAKE AVERAGE OF DATASETS PER CL.')
            self.merge_multiple_RNA_() #default to take average
        ############################################################################

        if not(self.RNA_Random == None or self.RNA_Random == "D"):
            if self.RNA_Random == "E" or self.RNA_Random == "C":
                print("LOAD DATA : DIRECT ENCODING OF CELL TYPE INSTEAD OF GENE EXPRESSION.")
            else:
                print('LOAD DATA : CONCAT RNA USING METHOD ' + self.RNA_Random)
            self.get_RNA_alternative_modified()

        #Here to select unique peaks per CL -- we look at targets
        if self.unique:
            self.selectUniquePeaks()
            #self.upsampleUniquePeaks()

        if self.upsampleRows:
            self.upsamplePeaks()

    def loadData_ChIP(self, chippath_file, type):
        """
            Load chip data can be in three ways:
                - (one) basset merged peaks and divided into training, testing, and Validation
                - (two) basset merged peaks and divided into training/valid and testing
                - (three) basset merged peaks and not divided at all.
        """

        self.loadData_ChIP_TestSplit(chippath_file)

    def loadData_ChIP_TestSplit(self, chippath_file):
        """
            #what were using
            Load ChIP-seq peaks across different cell lines for a specifc TF and antibody combination. (num_examples, 4, 1, 101)
            ChIPfile is h5 format with keys: ['target_labels', 'test_headers', 'test_in', 'test_out', 'train_in', 'train_out']
            The "training data" can then be further divided into training and validation set.

        h5_file, withATAC, Data, Label, balancedSampler=False, ATACLabel=None
        """
        h5_file = h5py.File(chippath_file, 'r') #open it once, not every single time

        if self.withATAC:
            print("with atac")
            self.train_set = H5Dataset(h5_file=h5_file, withATAC=self.withATAC, customSampler=self.customSampler,
                                        Data ='train_in', Label='train_out', ATACLabel='train_atac')
            self.test_set = H5Dataset(h5_file=h5_file, withATAC=self.withATAC, customSampler=self.customSampler,
                                        Data='test_in', Label='test_out', ATACLabel='test_atac')
            self.train_headers = h5_file['train_headers'] #instance names for the train sequences eg chr1:1-100
        else:
            self.train_set = H5Dataset(h5_file=h5_file, withATAC=self.withATAC, customSampler=self.customSampler,
                                        Data='train_in', Label='train_out')
            self.test_set = H5Dataset(h5_file=h5_file, withATAC=self.withATAC, customSampler=self.customSampler,
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

    ###############################################################################

    def removeTargets_ChIP(self):
        """
            Idea here that for a specifc TF there is ChIPseq data and RNaseq data.
            However, we are only interested in datasets with CLs found in both ChIP and RNA.
            As a result, we will need to remove ChIP-seq targets corresponding to CLs not found in RNA.
            And vice versa.
        """
        #Remember to keep order of the chipseq labels for training
        cellLines_ChIP = self.target_labels[:].astype('U13')
        cellLines_remove = np.setdiff1d(cellLines_ChIP, self.cellLines)
        CL_index = np.nonzero(np.in1d(cellLines_ChIP, cellLines_remove))[0]

        train_targets = np.delete(self.train_set.getTarget()[:], CL_index, axis=1)
        test_targets = np.delete(self.test_set.getTarget()[:], CL_index, axis=1)
        self.train_set.setTarget(train_targets)
        self.test_set.setTarget(test_targets)

        if self.withATAC:
            train_atac = np.delete(self.train_set.getATACseq()[:], CL_index, axis=1)
            test_atac = np.delete(self.test_set.getATACseq()[:], CL_index, axis=1)
            self.train_set.setATACseq(train_atac)
            self.test_set.setATACseq(test_atac)

        self.num_targets = self.test_set.getTarget().shape[1]
        print('LOAD DATA :: NUM TRAINING EXAMPLES : '  + str(self.numSeqs_train) + ' x ' + str(self.num_targets))
        print('LOAD DATA :: NUM TESTING EXAMPLES : '  + str(self.numSeqs_test) + ' x ' + str(self.num_targets))

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

    def loadData_RNA(self, rnaseqpath, TF, rnaType):
        """
            This is for normalized data generated using readRNA_RSEM_createH5

            Divide function -- organize.
            Load data corresponding to specific TF -- geneID.
            - TF/CL/RNAExperiment/RNAFiles

            Each RNAseq experiment has a file
            If rnaType == "gene_id":
                Columns represent the expected count per gene id across its respective RNAseq datasets are found.

            If rnaType == "transcript_id":
                Columns represent the expected count per transcript id across its respective RNAseq datasets are found.

            If rnaType == "exon_id":
                Columns represent the expected count per exon id across its respective RNAseq datasets are found.
        """
        h5f = h5py.File(rnaseqpath,'r')
        h5_TF = h5f[TF]

        #Get CLs corresponding to this TF
        CellLines = list(h5_TF.keys())
        df_RNAall = pd.DataFrame()

        for CL in CellLines:
            RNA_Experiments = list(h5_TF[CL].keys())
            for Exp in RNA_Experiments:
                df_RNAExp = pd.DataFrame(data = h5_TF[CL][Exp]['table'][:])

                #Need to change -- this is for normalized only
                del df_RNAExp['index']
                if 'Exon' in df_RNAExp.columns:
                    df_RNAExp.rename(columns={'Exon': rnaType}, inplace=True)
                elif 'Transcript' in df_RNAExp.columns:
                    df_RNAExp.rename(columns={'Transcript': rnaType}, inplace=True)
                elif 'Gene' in df_RNAExp.columns:
                    df_RNAExp.rename(columns={'Gene': rnaType}, inplace=True)

                df_RNAExp[rnaType] = df_RNAExp[rnaType].str.decode('utf-8')
                df_RNAExp = df_RNAExp.set_index(rnaType)

                RNAFiles = df_RNAExp.columns.tolist()
                tuples = []
                for File in RNAFiles:
                    tuples.append((CL, Exp, File))

                df_RNAExp.columns = pd.MultiIndex.from_tuples(tuples, names=('CellLine', 'RNA_Experiment', 'RNA_File'))
                df_RNAall = pd.concat([df_RNAall, df_RNAExp], axis=1, sort=False)

        self.RNA_FeatureCount = df_RNAall

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

    def merge_multiple_RNA_(self):
        """
            For a specific cell line, there can be multiple RNA seq experiments and for each of those experiments there are biological replicates.
            To deal with those, we have proposed a few methods:
                    - "A": Average across datasets for that cell line. -- what were doing now --
                    - "B": Randomly select an rna experiment from above list and take the mean of the datasets in experiment.
                    - "C": Randomly select an rna seq dataset from the above list of files.
                    - "D": PCA
                    - "E": Get all rnaseq experiments per CL.Add each experiment per CL as independent instance
                        such that seq1:CL1_rep1, seq1:CL1_rep2, seq1:CL1_rep3, seq1:CL1_rep4 ...

            Returns a dataframe where each column is a cell line, with the counts corresponding to the selected method of merge.
        """
        self.df_RNA_Summary = pd.DataFrame(columns=self.cellLines)

        #Way 1: Average across datasets for that cell line. -- other methods were explored in case of several datasets per CL
        for CL in self.cellLines:
            df_CL = self.RNA_FeatureCount[CL]
            RNA_Experiments = df_CL.columns.get_level_values(0).unique().tolist()
            RNA_Files = df_CL.columns.get_level_values(1).unique().tolist()
            averaged_Counts = df_CL.mean(axis =1)
            self.df_RNA_Summary[CL] = averaged_Counts

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

    def get_RNA_alternative_modified(self):
        """
            Call this at the beginning after loading RNAseq data.

            Instead of concatenating the exon level quantifications, we propose an alternative method.
            (A): Generate random vector per CL
            (B): Same random vector for all CLs
            (C): One hot encoding

        """
        print("LOAD DATA :: GET RNA ALTERNATIVE METHOD : " + str(self.RNA_Random) )
        if self.RNA_Random == "A":
            #Generate random vector per CL
            x = self.df_RNA_Summary.shape[0]
            y = self.df_RNA_Summary.shape[1]
            new_data = np.random.randint(1, 100, size=(x,y))
            df_new_matrix = pd.DataFrame(new_data, columns=self.cellLines)
            self.df_RNA_Summary = df_new_matrix

        elif self.RNA_Random == "B":
            #Same random vector for all CLs
            df_rand_same_matrix = sameVectorConcat()
            self.df_RNA_Summary = df_rand_same_matrix

        elif self.RNA_Random == "C" or self.RNA_Random == "E":
            #One hot encoding
            self.onehotCLs()
            if self.RNA_Random == "E":
                print("LOAD DATA : COMBINE CELL TYPE SPECIFIC AND CELL TYPE GENERAL IN ONE MODEL.")
                self.one_hot_dict[len(self.cellLines)] = [0]*len(self.cellLines) #Not telling it which CL it is
                self.cl_dict[len(self.cellLines)] = 'NoCL'

            print(self.one_hot_dict)
            print(self.cl_dict)
            self.df_RNA_Summary = pd.DataFrame(self.one_hot_dict).astype(int)
            print(self.df_RNA_Summary)

        elif self.RNA_Random == "test_both":
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
            #returns a one hot encoding + same random vector -- make into function
            self.df_RNA_Summary_g1 = self.sameVectorConcat()
            self.df_RNA_Summary_g2 = self.df_RNA_Summary.drop([len(self.cellLines)], axis=1)
            print(self.df_RNA_Summary_g1)
            print(self.df_RNA_Summary_g2)

    def get_all_Exp_per_CL(self):
        """
            Method 5: Include all experiments of all CLs
        """
        self.df_RNA_All = pd.DataFrame()

        for CL in self.cellLines:
            df_CL = self.RNA_FeatureCount[CL]
            RNA_Experiments = df_CL.columns.get_level_values(0).unique().tolist()
            RNA_Files = df_CL.columns.get_level_values(1).unique().tolist()
            df_CL = pd.DataFrame(df_CL.values, columns=[CL]*df_CL.values.shape[1])
            self.df_RNA_All = pd.concat([self.df_RNA_All, df_CL], axis=1)

        #self.df_RNA_All = self.df_RNA_All.loc[:, (self.df_RNA_All != 0).any(axis=0)]
        print(np.unique(np.array(self.df_RNA_All.columns.values)))

    def sample_batch_stage2(self, ChIPseq_batch, labels, atac_labels=None):
        """
            ChIPseq_batch: chipseq sequences
            labels: label for each instance per CL [labels are in matrix of size: #INSTANCES * #CL]
            type: training, validation, testing
            atac_labels: 0 -- not chromatin acessbile, 1 -- chromatin accessible

            Returns cell lines randomly selected according to batchsize -- this represents the rnaseq expression part
                and a chipseq dataset with randomly selected sequences according to batchsize

            This method is inspired by ChromDragoNN.
            For a batch size of BATCHSIZE, randomly select sample sequences from original training set and validation set
            for each of the ChIPseq and RNAseq.
        """

        #For each randomly selected instance in training set -- found using Basset method.
        #Number equal to number of peaks in training/validation set, randomly select a chipseq instance from the train/valid
        #Randomly select a cell line for each instance for the corresponding rnaseq counts
        CL_RNA_options_index_org = np.random.choice(len(self.cellLines), len(ChIPseq_batch)) #each batch
        labels = labels[np.arange(len(ChIPseq_batch)), CL_RNA_options_index_org]

        #set atac labels
        if self.withATAC:
            atac_dict = {0: [1,0], 1: [0,1]}
            atac_labels = atac_labels[np.arange(len(ChIPseq_batch)), CL_RNA_options_index_org]
            atac_labels = [ atac_dict[atac_labels[i].item()] for i in range(len(atac_labels))]
        else:
            atac_labels = None

        #Such that non cell specific vectors are concatenated
        if self.RNA_Random == "E":
            halfBatch = int(len(ChIPseq_batch)/2)
            CL_RNA_options_index = list(CL_RNA_options_index_org)
            CL_RNA_options_index = CL_RNA_options_index[:halfBatch] + [len(self.cellLines)]*(len(ChIPseq_batch) - halfBatch)

            if self.withATAC:
                assert(len(ChIPseq_batch) == len(atac_labels))
                for _ in range(halfBatch):
                    atac_labels[random.randint(0, len(ChIPseq_batch)-1)] = [0,0]

        if self.withATAC:
            atac_labels = torch.tensor(atac_labels, dtype=torch.int8)

        return CL_RNA_options_index, atac_labels, labels,  CL_RNA_options_index_org

    def sample_batch_stage2_Siamese(self, ChIPseq_batch, labels, type, atac_labels=None):
        """
        """
        CL_RNA_options_index = np.random.choice(len(self.cellLines), len(ChIPseq_batch)) #each batch
        labels = labels[np.arange(len(ChIPseq_batch)), CL_RNA_options_index]

        if self.withATAC:
            atac_dict = {0: [1,0], 1: [0,1]}
            atac_labels = atac_labels[np.arange(len(ChIPseq_batch)), CL_RNA_options_index]
            atac_labels = [ atac_dict[atac_labels[i].item()] for i in range(len(atac_labels))]
        else:
            atac_labels = None

        if type == "train":
            halfBatch = int(len(ChIPseq_batch)/2)
            CL_RNA_options_index = list(CL_RNA_options_index)
            CL_RNA_options_index = CL_RNA_options_index[:halfBatch] + [len(self.cellLines)]*(len(ChIPseq_batch) - halfBatch)

            if self.withATAC:
                assert(len(ChIPseq_batch) == len(atac_labels))
                for _ in range(halfBatch):
                    atac_labels[random.randint(0, len(ChIPseq_batch)-1)] = [0,0]

        if self.withATAC:
            atac_labels = torch.tensor(atac_labels, dtype=torch.int8)

        return CL_RNA_options_index, atac_labels, labels

    def get_train_instances(self):
        return self.instances_train, self.labels_train

###############################################################################################################################################
    def getChIP(self):
        return self.train_set, self.test_set

    def getallRNA(self):
        return self.RNA_FeatureCount

    def getSumRNA(self):
        #return all, same, one hot
        if self.RNA_Random == "test_both":
            return self.df_RNA_Summary, self.df_RNA_Summary_g1, self.df_RNA_Summary_g2
        else:
            return self.df_RNA_Summary

    def getCLs_Common(self):
        """
            We get ChIP-seq and RNA-seq for a specific TF.
            So we need to get the common CLs for both the ChIP and RNA.
            This is run at the beginning
        """
        cellLines_ChIP = self.target_labels[:].astype('U13') #From ChIP
        column_RNA = self.RNA_FeatureCount.columns.values #From RNA
        cellLines_RNA = np.unique(np.array([i[0] for i in column_RNA]))
        cellLines = np.intersect1d(cellLines_RNA, cellLines_ChIP)
        return cellLines

    def getCLs_ChIP(self):
        return self.target_labels[:].astype('U13')

    def getCLs_RNA(self):
         return self.df_RNA_Summary.columns.values

    def getNumCellTypes(self):
        return len(self.cellLines)

    def getcellLines(self):
        return self.cellLines

    def getNumExons(self):
        return len(self.df_RNA_Summary)
