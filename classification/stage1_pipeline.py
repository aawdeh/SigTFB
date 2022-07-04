import torch
from torch import nn
import numpy as np
import sys
from sys import path
import os
from torch.utils.data import DataLoader
from torch.utils.data.dataset import ConcatDataset
from torch.utils.data.sampler import SubsetRandomSampler
from ax.service.ax_client import AxClient

# ------------------------------------
# own modules
# ------------------------------------
path.append("/project/6006657/aaseel/Aim2/Scripts_GithubEdition")
#import stage1_utils
from stage1_utils import load_data, set_dataLoaders_sampler, train, evaluate, readHyperparameters, train_full, save_model
from DNN.utils import basset_loss as criterion_basset
from DNN.utils import torch_seed
from DNN.Models.stage1Model import CNN_Multilabel as Net
from DNN.args import getArgs_Stage1 as getArgs

#####################################################################################
# Hyperparameters
#####################################################################################
args = getArgs()
args.shuffle = eval(args.shuffle)
args.unique = eval(args.unique)
args.customSampler = eval(args.customSampler)
args.customSampler_OneLoss = eval(args.customSampler_OneLoss)

if True:
    args.TF = "ATF3"
    args.AB = "ENCAB000ADZ"
    args.chipData = "/project/6006657/aaseel/Aim2/Data/ChIP/Redo_Others/hdf5Files_TestSplit/ATF3/ATF3.ENCAB000ADZ.h5"
    #args.chipData = "/project/6006657/aaseel/Aim2/Data/ChIP/ChIPseq_hdf5Files_CLName_withATAC/" + args.TF + "/" + args.TF + "." + args.AB + ".h5"
    #args.chipData = "/project/6006657/aaseel/Aim2/Data/Test_Files/hdf5Files/" + args.TF + "/" + args.TF + "." + args.AB + ".h5"

    args.typeChIP = "two"
    args.Encoding_Random = 'E' #half time general half time specific
    args.TestingType = 'OneUnit_Stage2'
    args.num_epochs = 1
    args.batch_size = 4 #needs to be a multiple of the number of celllines

    args.customSampler = True
    args.customSampler_OneLoss = True

    # 6 conditions were testing for
    args.shuffle = False
    args.upsampleRows = False
    args.unique = False

    # 2 losses : bce and basset
    args.lossFunction = "BASSET"

print(args)
print("Ax Hyperparameter Optimization :: Stage 1 :: " + args.TF + "." + args.AB)
#####################################################################################
# Define search space
#####################################################################################
parameters=[
    {
        "name": "learning_rate",
        "type": "range",
        "bounds": [1e-6, 1e-3],
        "log_scale": True,
    },
    {
        "name": "weight_decay",
        "type": "range",
        "bounds": [1e-10, 1e-3],
        "log_scale": True,
    },
    {
        "name": "motif_std",
        "type": "range",
        "bounds": [1e-7, 1e-3],
        "log_scale": True,
    },
    {
        "name": "linear_std",
        "type": "range",
        "bounds": [1e-5, 1e-2],
        "log_scale": True,
    },
    {
        "name": "num_channels",
        "type": "range",
        "bounds": [16, 100],
        "value_type": "int"
    },
    {
        "name": "batch_size",
        "type": "choice",
        "values": [64, 128, 256],
    },
    {
        "name": "num_epochs",
        "type": "range",
        "bounds": [50, 200],
        "value_type": "int"
    },
]

#####################################################################################
# Inspired by https://github.com/facebook/Ax/blob/master/tutorials/tune_cnn.ipynb
#####################################################################################
torch_seed(12345)
dtype = torch.float
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
args.device = device
args, dl, dataset, test_set = load_data(args)
args.one_hot_dict, args.cl_dict = dl.getOneHotDictCL()

if args.lossFunction == "BCE":
    print("LOSS FUNCTION :: BCE.")
    if args.customSampler_OneLoss:
        args.criterion = nn.BCELoss(reduction="None")
    else:
        args.criterion = nn.BCELoss()
else:
    print("LOSS FUNCTION :: BASSET.")
    args.criterion = criterion_basset

#####################################################################################
# 2. Define function to optimize
#####################################################################################
def train_evaluate(parameterization):
    args.linear_std = parameterization.get('linear_std',1.0)
    args.motif_std = parameterization.get('motif_std',1.0)
    args.num_channels = parameterization.get('num_channels', 16)
    args.batch_size = parameterization.get("batch_size", 64)

    train_idx, validate_idx, train_loader, valid_loader, test_loader = set_dataLoaders_sampler(dataset, test_set, args)

    net = Net(args)
    model, _ = train(net=net, train_loader=train_loader, parameters=parameterization, args=args, dtype=dtype, device=device)

    return evaluate(
        net=model,
        data_loader=valid_loader,
        args=args,
        dtype=dtype,
        device=device,
    )

#####################################################################################
# 3. Run the optimization loop
#####################################################################################
print("Run Ax Optimization :: Stage 1 :: " + args.TF + "." + args.AB)
ax_client = AxClient()
ax_client.create_experiment(
    name="stage1_experiment",
    parameters=parameters,
    objective_name='train_evaluate',
    minimize=False)

for i in range(15):
    parameters, trial_index = ax_client.get_next_trial()
    ax_client.complete_trial(trial_index=trial_index, raw_data=train_evaluate(parameters))

#####################################################################################
# 4. Get best parameters
#####################################################################################
print("Best Hyperparameters :: Stage 1 :: " + args.TF + "." + args.AB)

best_parameters, values = ax_client.get_best_parameters()
for k in best_parameters.items(): #the best set of parameters
  print(k)
print()
print(best_parameters)

# the best score achieved.
means, covariances = values
print(means)
print(covariances)

with open(os.path.join(args.hyperparameterPath, "hyperparameters.txt"), "w") as f:
    f.write( "ACCURACY," + str(means['train_evaluate']) + "\n")
    for k in best_parameters.items():
        f.write(str(k[0]) +"," + str(k[1]) + "\n")
f.close()

print(ax_client.get_trials_data_frame().sort_values('trial_index').to_string())
ax_client.get_trials_data_frame().sort_values('trial_index').to_csv(os.path.join(args.hyperparameterPath, "dataframe.csv"), index=False)

#####################################################################################
# 5. Train model with best hyperparameters
#####################################################################################
model_best, size_conv_best, args_best = train_full(best_parameters, args, dataset, test_set)
save_model(model_best, size_conv_best, args_best)
