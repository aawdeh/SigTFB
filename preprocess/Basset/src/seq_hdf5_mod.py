#!/usr/bin/env python
from __future__ import print_function
from optparse import OptionParser
import sys
import h5py
import numpy.random as npr
import numpy as np
import pandas as pd

import dna_io_mod

################################################################################
# seq_hdf5.py
#
# Make an HDF5 file for Torch input out of a FASTA file and targets text file,
# dividing the data into training, validation, and test.
################################################################################

################################################################################
# main
################################################################################
def main():
    usage = 'usage: %prog [options] <fasta_file> <targets_file> <atac_file> <out_file>'
    parser = OptionParser(usage)
    parser.add_option('-a', dest='add_features_file', default=None, help='Table of additional features')
    parser.add_option('-b', dest='batch_size', default=None, type='int', help='Align sizes with batch size')
    parser.add_option('-c', dest='counts', default=False, action='store_true', help='Validation and training proportions are given as raw counts [Default: %default]')
    parser.add_option('-e', dest='extend_length', type='int', default=None, help='Extend all sequences to this length [Default: %default]')
    parser.add_option('-r', dest='permute', default=False, action='store_true', help='Permute sequences [Default: %default]')
    parser.add_option('-s', dest='random_seed', default=1, type='int', help='numpy.random seed [Default: %default]')
    parser.add_option('-t', dest='test_pct', default=0, type='float', help='Test % [Default: %default]')
    parser.add_option('-v', dest='valid_pct', default=0, type='float', help='Validation % [Default: %default]')
    parser.add_option('--vt', dest='valid_test', default=False, action='store_true', help='Use validation as test, too [Default: %default]')
    (options,args) = parser.parse_args()

    if len(args) != 4:
        parser.error('Must provide fasta file, targets file, atac file and an output prefix')
    else:
        fasta_file = args[0]
        targets_file = args[1]
        atac_file = args[2]
        out_file = args[3]

    # seed rng before shuffle
    npr.seed(options.random_seed)

    #################################################################
    # load data
    #################################################################
    seqs, targets, atac_overlap = dna_io_mod.load_data_1hot(fasta_file, targets_file, atac_file,
                                                extend_len=options.extend_length, mean_norm=False, whiten=False, permute=False, sort=False)

    # reshape sequences for torch
    seqs = seqs.reshape((seqs.shape[0],4,1,seqs.shape[1]//4)) #Aseel changed this

    # read headers
    headers = []
    for line in open(fasta_file):
        if line[0] == '>':
            headers.append(line[1:].rstrip())
    headers = np.array(headers)

    # read labels -- these are the cell lines
    target_labels = open(targets_file).readline().strip().split('\t')

    # read additional features
    if options.add_features_file:
        df_add = pd.read_table(options.add_features_file, index_col=0)
        df_add = df_add.astype(np.float32, copy=False)

    # permute
    if options.permute:
        order = npr.permutation(seqs.shape[0])
        seqs = seqs[order]
        targets = targets[order]
        headers = headers[order]
        if options.add_features_file:
            df_add = df_add.iloc[order]

    # check proper sum
    if options.counts:
        assert(options.test_pct + options.valid_pct <= seqs.shape[0])
    else:
        assert(options.test_pct + options.valid_pct <= 1.0)

    #################################################################
    # divide data
    #################################################################
    if options.counts:
        test_count = int(options.test_pct)
        valid_count = int(options.valid_pct)
    else:
        test_count = int(0.5 + options.test_pct * seqs.shape[0])
        valid_count = int(0.5 + options.valid_pct * seqs.shape[0])

    train_count = seqs.shape[0] - test_count - valid_count
    train_count = batch_round(train_count, options.batch_size)
    print('%d training sequences ' % train_count, file=sys.stderr)

    test_count = batch_round(test_count, options.batch_size)
    print('%d test sequences ' % test_count, file=sys.stderr)

    valid_count = batch_round(valid_count, options.batch_size)
    print('%d validation sequences ' % valid_count, file=sys.stderr)

    i = 0
    train_seqs, train_targets, train_atac, train_headers = seqs[i:i+train_count,:], targets[i:i+train_count,:], atac_overlap[i:i+train_count,:], headers[i:i+train_count]
    i += train_count
    valid_seqs, valid_targets, valid_atac, valid_headers = seqs[i:i+valid_count,:], targets[i:i+valid_count,:], atac_overlap[i:i+valid_count,:], headers[i:i+valid_count]
    i += valid_count
    test_seqs, test_targets, test_atac, test_headers = seqs[i:i+test_count,:], targets[i:i+test_count,:], atac_overlap[i:i+test_count,:], headers[i:i+test_count]

    if options.add_features_file:
        i = 0
        train_add = df_add.iloc[i:i+train_count]
        i += train_count
        valid_add = df_add.iloc[i:i+valid_count]
        i += valid_count
        test_add = df_add.iloc[i:i+test_count]

    #################################################################
    # construct hdf5 representation
    #################################################################
    h5f = h5py.File(out_file, 'w')
    #fix
    target_labels = [header.encode('utf-8') for header in target_labels]
    train_headers = [header.encode('utf-8') for header in train_headers]
    test_headers = [header.encode('utf-8') for header in test_headers]
    valid_headers = [header.encode('utf-8') for header in valid_headers]

    h5f.create_dataset('target_labels', data=target_labels)

    if train_count > 0:
        h5f.create_dataset('train_in', data=train_seqs)
        h5f.create_dataset('train_out', data=train_targets)
        h5f.create_dataset('train_headers', data=train_headers)
        h5f.create_dataset('train_atac', data=train_atac)

    if valid_count > 0:
        h5f.create_dataset('valid_in', data=valid_seqs)
        h5f.create_dataset('valid_out', data=valid_targets)
        h5f.create_dataset('valid_atac', data=valid_atac)

    if test_count > 0:
        h5f.create_dataset('test_in', data=test_seqs)
        h5f.create_dataset('test_out', data=test_targets)
        h5f.create_dataset('test_headers', data=test_headers)
        h5f.create_dataset('test_atac', data=test_atac)

    elif options.valid_test:
        h5f.create_dataset('test_in', data=valid_seqs)
        h5f.create_dataset('test_out', data=valid_targets)
        h5f.create_dataset('test_headers', data=valid_headers)

    if options.add_features_file:
        h5f.create_dataset('add_labels', data=list(df_add.columns))

        if train_count > 0:
            h5f.create_dataset('train_add', data=train_add.as_matrix())
        if valid_count > 0:
            h5f.create_dataset('valid_add', data=valid_add.as_matrix())
        if test_count > 0:
            h5f.create_dataset('test_add', data=test_add.as_matrix())
        elif options.valid_test:
            h5f.create_dataset('test_add', data=valid_add.as_matrix())

    h5f.close()

def batch_round(count, batch_size):
    if batch_size != None:
        count -= (batch_size % count)
    return count

################################################################################
# __main__
################################################################################
if __name__ == '__main__':
    main()
