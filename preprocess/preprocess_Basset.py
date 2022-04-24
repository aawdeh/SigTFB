#!/usr/bin/env python
import pandas as pd
import numpy as np
import subprocess
import os
import sys
from os import path
import math
from shutil import copyfile

################################################################################
# preprocess_Basset.py and seq_hdf5.py
#
# ChIP-seq files for a specific TF-AB are organized in the same directory [/TF/AB/files]
# For example:
# If TF-AB has three CLs [3 ChIP-seq bed files each corresponding to a CL], dir format
# .../TF/AB/CL1.bed
# .../TF/AB/CL2.bed
# .../TF/AB/CL2.bed
#
# For each TF and antibody combination, preprocess the associated bed files
# using the preprocess_features.py function in the Basset SW
#
# Preprocess a set of feature BED files for Basset analysis, potentially adding
# them to an existing database of features, specified as a BED file with the
# target activities comma-separated in column 4 and a full activity table file.
#
# make sure to load BEDTOOLS before
################################################################################
#Basset functions
FeatureScript = '/project/6006657/aaseel/Aim2/SW/Basset/src/preprocess_features.py' #need to change path
hdf5_script = '/project/6006657/aaseel/Aim2/SW/Basset/src/seq_hdf5.py'

################################################################################
# STEP 1 CREATE TEXT FILES TO USE AS INPUT FOR BASSET FUNCTION
################################################################################
def createTextFiles(dir, outputDir, name):
    '''
        This creates the text files of format for each TF and antibody combination
        CellName/CellID \t Path to Bed file

        Used as input for the preprocess_features function

        outputDir: path of directory to output text files
    '''
    TF, AB = name.split(".")
    chipFiles = os.listdir(os.path.join(dir, TF, AB))
    #if len(chipFiles) > 1:
    w = open(os.path.join(outputDir, TF + '.' + AB + '.txt'), 'w')
    print(TF + "." + AB)
    for file in chipFiles:
        CL_id = file.split('.')[0]
        w.write(CL_id + " " + os.path.join(dir, TF, AB, file) + '\n')
    w.close()

################################################################################
# STEP 2 CALL BASSET TO MERGE PEAKS
################################################################################
def call_Basset(TextFiles_Dir, outputDir, chromSize, name):
    '''
        This function is to call the preprocess_features.py in Basset.
        This is where we can control the length of the sequence.
    '''
    print('Merge peak using function in Basset SW.')

    #filesList = os.listdir(TextFiles_Dir)
    filesList = [name+".txt"]

    for file in filesList:
        TF, Antibody,_ = file.split(".")
        tf_outputDir = os.path.join(outputDir, TF)

        print(file)
        if not os.path.exists(tf_outputDir):
            os.makedirs(tf_outputDir)

        cmd = 'python ' + FeatureScript + ' -m 30 -s 101 -o ' + os.path.join(tf_outputDir, TF + '.' + Antibody) + ' -c ' + chromSize + ' ' + os.path.join(TextFiles_Dir, file)
        subprocess.call(cmd, shell=True, executable='/bin/bash')

################################################################################
# STEP 3 CONVERT MERGED BED TO FASTA
################################################################################
def convert2Fasta(MergedBed_Dir, Fasta_Dir, hg38):
    '''
        Convert merged bed to fasta
    '''
    print('Convert to fasta.')
    TFList = os.listdir(MergedBed_Dir)
    for TF in TFList:
        abList = os.listdir(os.path.join(MergedBed_Dir, TF))
        print(abList)

        for file in abList:
            if file.endswith(".bed"):
                print(file)
                TF_file = file.split('.')[0]
                AB_file = file.split('.')[1]

                tf_outputDir = os.path.join(Fasta_Dir, TF_file)
                print(tf_outputDir)
                if not os.path.exists(tf_outputDir):
                    os.makedirs(tf_outputDir)

                cmd = 'bedtools getfasta -fi ' + hg38 + ' -bed ' + os.path.join(MergedBed_Dir, TF_file, file) + ' -s -fo ' + os.path.join(tf_outputDir, TF_file + '.' + AB_file + '.fa')
                subprocess.call(cmd, shell=True, executable='/bin/bash')

################################################################################
# STEP 4 CONVERT FASTA TO H5
################################################################################
def convert2hdf5(Fasta_Dir, MergedBed_Dir, Hdf5_Dir):
    '''
        Use seq_hdf5.py to convert fasta file -> hdf5 file
    '''
    print('Convert to hdf5.')
    #hdf5_script = '/project/6019283/aaseel/Aim2/SW/Basset/src/seq_hdf5_modified.py'
    TFList = os.listdir(Fasta_Dir)

    for tf in TFList:

        abList = os.listdir(os.path.join(Fasta_Dir, tf))
        for file in abList:
            TF_file = file.split('.')[0]
            AB_file = file.split('.')[1]

            #Files
            fastaFile = os.path.join(Fasta_Dir, tf, file)
            activityFile = os.path.join(MergedBed_Dir, tf, TF_file + '.' + AB_file + '_act.txt')
            bedFile = os.path.join(MergedBed_Dir, tf, TF_file + '.' + AB_file + '.bed')
            h5File = os.path.join(Hdf5_Dir, tf, TF_file + '.' + AB_file + '.h5')

            #Make dir
            tf_outputDir = os.path.join(Hdf5_Dir, TF_file)
            if not os.path.exists(tf_outputDir):
                os.makedirs(tf_outputDir)

            #Need to find number of training, validation, testing
            df = pd.read_csv(fastaFile, header=None)
            numPeaks = len(df)//2
            print("Split data into training/valid and testing.")
            numTesting = int(math.floor(numPeaks * 0.05))
            numTrainingandValid = numPeaks - numTesting
            print(numPeaks, numTrainingandValid + numTesting, numTrainingandValid, numTesting)
            cmd = hdf5_script + ' -c -r -t ' + str(numTesting) + ' ' + fastaFile + ' ' + activityFile + ' ' + h5File
            subprocess.call(cmd, shell=True, executable='/bin/bash')


def main():
    """
        Files in bedDir are arranged such that you can access all CL files per TF [directory] per AB [directory].
    """

    redoChIPList = ["GABPA.ENCAB000AGR"]

    for chip in redoChIPList:
        bedDir='/project/6006657/aaseel/Aim2/Data/ChIP/NP_optimalIDR_Rearranged' #dir where bedfiles are
        bassetDir = '/project/6006657/aaseel/Aim2/Data/ChIP'
        chromSize = '/project/6006657/aaseel/Aim2/SW/Basset/data/genomes/hg38.chrom.sizes'
        hg38 = '/project/6006657/aaseel/Aim2/Data/hg38.fa'

        TextFiles_Dir = os.path.join(bassetDir, 'TextFiles')
        MergedBed_Dir = os.path.join(bassetDir, 'MergedBed')
        Fasta_Dir = os.path.join(bassetDir, 'FastaFiles')
        hdf5_Dir = os.path.join(bassetDir, 'hdf5Files')

        #Step1
        createTextFiles(bedDir, TextFiles_Dir, chip)

        #Step2
        call_Basset(TextFiles_Dir, MergedBed_Dir, chromSize, chip)

        #Step 3 -- convert to fasta
        #convert2Fasta(MergedBed_Dir, Fasta_Dir, hg38)

        #Step 4 -- convert fasta to hdf5
        #convert2hdf5(Fasta_Dir, MergedBed_Dir, hdf5_Dir)

if __name__== "__main__":
   main()
