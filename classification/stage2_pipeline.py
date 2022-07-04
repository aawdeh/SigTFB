import torch
import numpy as np
import sys
import random
from sys import path
import os
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from ax.service.ax_client import AxClient

# ------------------------------------
# own modules
# ------------------------------------
path.append("/project/6006657/aaseel/Aim2/Scripts_GithubEdition")
from AxPytorch.Stage2.stage2_utils import train, evaluate
from AxPytorch.Stage1.stage1_utils import load_data, set_dataLoaders_sampler
from DNN.utils import load_model_checkpoint, save_model, torch_seed, saveIndex
from DNN.Models.Stage2.stage2Model import Net_Stage2
from AxPytorch.Stage2.ax_trainFull_stage2_other import saveModels
from DNN.args import getArgs_Stage2 as getArgs

#####################################################################################
# Hyperparameters
#####################################################################################
args = getArgs()
args.shuffle = eval(args.shuffle)
args.unique = eval(args.unique)
args.upsampleRows = eval(args.upsampleRows)
args.customSampler = eval(args.customSampler)
args.customSampler_OneLoss = eval(args.customSampler_OneLoss)

if True:
    #ATF1.ENCAB697XQW.pth.tar
    args.TF = "ATF7"
    args.AB = "ENCAB000BMO"
    args.chipData = "/project/6006657/aaseel/Aim2/Data/ChIP/hdf5Files_TestSplit/" + args.TF + "/" + args.TF + "." + args.AB + ".h5"

    args.model_stage1_path = "/home/aaseel/projects/def-tperkins/aaseel/Aim2/Results/Ax/Stage1/ACC_SAMPLER_CUSTOM/Models/" + args.TF + "." + args.AB + ".pth.tar"
    args.saveModelPath="/project/6006657/aaseel/Aim2/Results/Ax/Stage2/ACC_SAMPLER_CUSTOM/noCA/AUPRC/Models/AllHyperparameters/PartTrainTest"

    args.typeChIP = "two"
    args.Encoding_Random = 'E'
    args.batch_size = 4

    args.rc = False
    args.unique = False

print("Ax Hyperparameter Optimization :: Stage 2 :: " + args.TF + "." + args.AB)
#####################################################################################
# Define search space
#####################################################################################
parameters=[
    {
        "name": "learning_rate",
        "type": "range",
        "bounds": [1e-6, 1e-1],
        "log_scale": True,
    },
    {
        "name": "weight_decay",
        "type": "range",
        "bounds": [1e-10, 1e-1],
        "log_scale": True,
    },
    {
        "name": "conv_std",
        "type": "range",
        "bounds": [1e-7, 1e-1],
        "log_scale": True,
    },
    {
        "name": "linear_std",
        "type": "range",
        "bounds": [1e-5, 1e-1],
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
    {
        "name": "neurons_per_layer",
        "type": "range",
        "bounds": [1, 100],
        "value_type": "int"
    },
    {
        "name": "neurons_per_layer_pre_concat",
        "type": "range",
        "bounds": [1, 100],
        "value_type": "int"
    },
    {
        "name": "freeze_pretrained_model",
        "type": "choice",
        "values": [True, False],
    },
    {
        "name": "dropout_prob",
        "type": "range",
        "bounds": [1e-6, 1e-3],
        "log_scale": True,
    },
    {
        "name": "momentum_rate",
        "type": "range",
        "bounds": [1e-6, 1e-1],
        "log_scale": True,
    },
]

#####################################################################################
# Inspired by https://github.com/facebook/Ax/blob/master/tutorials/tune_cnn.ipynb
#####################################################################################
torch_seed(12345)
dtype = torch.float
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
args.device = device
print(device)
#####################################################################################
# 1. Load Data
#####################################################################################
args, dl, dataset, test_set = load_data(args)

#####################################################################################
# 2. Define function to optimize
#####################################################################################
def train_evaluate(parameterization):
    args.linear_std = parameterization.get('linear_std',1.0)
    args.conv_std = parameterization.get('conv_std',1.0)
    args.num_channels = parameterization.get('num_channels', 16)
    args.batch_size = parameterization.get("batch_size", 64)
    args.num_hidden_layers = parameterization.get("num_hidden_layers", 1)
    args.neurons_per_layer = parameterization.get("neurons_per_layer", 1)
    args.neurons_per_layer_pre_concat = parameterization.get("neurons_per_layer_pre_concat", 1)
    args.dropout_prob = parameterization.get("dropout_prob", 0)
    args.num_epochs = parameterization.get("num_epochs", 0)
    args.momentum_rate = parameterization.get("momentum_rate", 0)
    args.learning_rate = parameterization.get("learning_rate", 0)
    args.weight_decay = parameterization.get("weight_decay", 0)
    args.freeze_pretrained_model = parameterization.get("freeze_pretrained_model", False)

    print("Ax Train :: Stage 2 :: Set DataLoaders.")
    train_idx, validate_idx, train_loader, valid_loader, test_loader = set_dataLoaders_sampler(dataset, test_set, args)
    checkpoint_stage1, model_stage1, _ = load_model_checkpoint(args.model_stage1_path, 'cpu')

    net = Net_Stage2(model_stage1, args, checkpoint_stage1['size_conv_out']).to(device)
    net.model_stage1.load_state_dict(checkpoint_stage1['state_dict']) #check this

    model_stage1 = model_stage1.to(device)
    model = train(net=net, dl=dl, train_loader=train_loader, parameters=parameterization, args=args, dtype=dtype, device=device)

    return evaluate(
        net=model,
        dl=dl,
        data_loader=valid_loader,
        args=args,
        dtype=dtype,
        device=device,
    )

#####################################################################################
# 3. Run the optimization loop
#####################################################################################
print("Run Ax Optimization :: Stage 2 :: " + args.TF + "." + args.AB)
ax_client = AxClient()
ax_client.create_experiment(
    name="stage2_experiment",
    parameters=parameters,
    objective_name='train_evaluate',
    minimize=False)

print(ax_client)

for i in range(2):
    parameters, trial_index = ax_client.get_next_trial()
    args.trial_index = trial_index
    print(parameters)
    ax_client.complete_trial(trial_index=trial_index, raw_data=train_evaluate(parameters))

#####################################################################################
# 4. Get best parameters
#####################################################################################
print("Best Hyperparameters :: Stage 2 :: " + args.TF + "." + args.AB)

best_parameters, values = ax_client.get_best_parameters()
for k in best_parameters.items(): #the best set of parameters
  print(k)
print()

# the best score achieved.
means, covariances = values
print(means)
print(covariances)

# with open(os.path.join(args.hyperparameterPath, "hyperparameters.txt"), "w") as f:
#     f.write( "AUC," + str(means['train_evaluate']) + "\n")
#     for k in best_parameters.items():
#         f.write(str(k[0]) +"," + str(k[1]) + "\n")
# f.close()

print(ax_client.get_trials_data_frame().sort_values('trial_index').to_string())
ax_client.get_trials_data_frame().sort_values('trial_index').to_csv(os.path.join(args.hyperparameterPath, "dataframe.csv"), index=False)

#####################################################################################
# 5. Train model with best hyperparameters
#####################################################################################
saveModels(args, best_parameters)
