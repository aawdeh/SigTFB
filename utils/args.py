import argparse

def getArgs_Common(parser):
    parser.add_argument('--chipData', type=str, metavar='ChIP PATH', help="Path to chip data.",required=True)
    parser.add_argument('--TF', type=str, metavar='TF Name',required=True)
    parser.add_argument('--AB', type=str, metavar='Antibody',required=True)

    parser.add_argument('--typeChIP', default='two', type=str, metavar='Divide train/valid/test', help="indicates how the ChIP was divided: (one) train/valid/test or (two) train/test or (three) full.")
    parser.add_argument('--Encoding_Random', default="E", help='This is to see whether the addition of the cell type encoding to the network makes a difference. As a results, we try with (A) Random generated vector for each CL, (B) Same random vector for all CLs, (C) One hot encoding, (D) Random vector per instance. (E) One hot encoding + same random vector (non specific) , (test_both): Return two vectors -- One hot encoding & random vector.')

    #parser.add_argument('--cuda', '-c', default=True, help="GPU availibility.")
    parser.add_argument('--idxPath', default=None, type=str, metavar='Get indexes of training and valdiation cross valdiation')
    parser.add_argument('--saveModelPath', type=str, metavar='Save model PATH', help='Path to save models generated.',required=True)
    parser.add_argument('--hyperparameterPath', type=str, metavar='Save hyperparameter PATH', help='Path to save hyperparameter.',required=True)

    parser.add_argument('--customSampler_OneLoss', type=str, default="True", help='Custom balance distribution per cell line. Default is false.')
    parser.add_argument('--customSampler', type=str, default="True", help='Custom balance distribution per cell line. Default is false.')

    parser.add_argument('--unique', type=str, default="False", help='Unique per cell type. Default is false.')
    parser.add_argument('--shuffle', type=str, default="False", help='Include dinucleotide shuffle. Default is false.')
    parser.add_argument('--upsampleRows', type=str, default="False", help='Upsample unique row patterns. Default is false.')

    parser.add_argument('--lossFunction', default="BASSET", type=str, metavar='Loss Function')
    parser.add_argument('--learning_rate', default=0.0001, type=float, metavar='Learning Rate')
    parser.add_argument('--momentum_rate', default=0.0001, type=float, metavar='Momentum Rate')
    parser.add_argument('--weight_decay', default=0, type=float, metavar='Weight Decay')
    parser.add_argument('--dropout_prob', default=0.0, type=float, metavar='Dropout Probability. A prob of 0 means that theres no dropout.')
    parser.add_argument('--conv_std', default=1.0, type=float, metavar='Standard deviation of normal distribution for randomly selecting initial weights of convolutional layer.')
    parser.add_argument('--linear_std', default=1.0, type=float, metavar='Standard deviation of normal distribution for randomly selecting initial weights of linear fully connected layer.')
    parser.add_argument('--batch_size', default=64, type=int, metavar='Batch size per epoch.')
    parser.add_argument('--num_epochs', default=100, type=int, metavar='Number of epochs.')
    parser.add_argument('--num_channels', default=16, type=int, metavar='Number of channels for convolutional filter in stage 1.')
    parser.add_argument('--seq_length', default=101, type=int, metavar='Length of sequence.')
    parser.add_argument('--num_units', default=1, type=int, metavar='Length of exon or transcript or exon.')
    parser.add_argument('--num_cell_types', default=1, type=int, metavar='Number of cell types for a specific TF.')

    parser.add_argument('--saveModelBool', action='store_true', help='Save models.')
    parser.add_argument('--savePlotBool', action='store_true', help='Save plots.')
    parser.add_argument('--baseline', action='store_true', help='Conduct baseline comparsion of model. Here baseline refers to setting each same class for each instance across all instances.')
    parser.add_argument('--dataDivided', action='store_true', help='True, if the data is already divided into train, test and valid. False, otherwise.')
    parser.add_argument('--sampleData', action='store_true', help='Sample 10K from the train set and 1K from the valid set for testing purposes.')

    return parser

def getArgs_Stage1():
    parser = argparse.ArgumentParser(description='TFBS Prediction Stage 2')
    getArgs_Common(parser)

    args = parser.parse_args()
    return args

def getArgs_Stage2():
    print("Stage2")
    parser = argparse.ArgumentParser(description='TFBS Prediction Stage 2')
    getArgs_Common(parser)

    parser.add_argument('--modelname', default='Net', type=str, help='Model type: Net, Net_NoLayers, Net_OneLayer')
    parser.add_argument('--model_stage1_path', type=str, metavar='Load stage 1 model', help="Path to saved model of stage 1.")
    parser.add_argument('--freeze_pretrained_model', action='store_true', help='Freeze pretrained model.')
    parser.add_argument('--sigmoid_bool', action='store_true', help='Last layer for stage 2 model. If log_softmax is used, then NLLLoss is used. If log_sigmoid, then basset loss is used.')
    parser.add_argument('--dropout_bool', action='store_true', help='Add dropout to stage 2 model. Default is true.')
    parser.add_argument('--num_hidden_layers', type=int, default=1, help='Number of layers in fully connect network for stage 2 model.')
    parser.add_argument('--neurons_per_layer', type=int, default=50, help='Number of neurons per layer in fully connect network for stage 2 model.')
    parser.add_argument('--neurons_per_layer_pre_concat', type=int, default=50, help='Number of neurons in layer in fully connect network before concatenation for stage 2 model.')

    parser.add_argument('--concatRNA_epoch', action='store_true', help='Concat the same cell line for each sequence per epoch. Randomly select CL for each epoch.')
    parser.add_argument('--concatRNA_batch', action='store_true', help='Concat the same cell line for each sequence per batch. Randommly select CL for each batch.')

    args = parser.parse_args()
    return args
