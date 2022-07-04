# SigTFB: Cell Type Specific DNA Signatures of Transcription Factor Binding

This repository contains code for our paper "Cell Type Specific DNA Signatures of Transcription Factor Binding". The models are implemented in PyTorch.

## Data

All associated data from our paper can be downloaded from [here](https://www.youtube.com)

### Preparing ChIP-seq data for each transcription factor (TF) and antibody (AB) across their corresponding cell types (CLs)

|Peaks | CL1 |  CL2 | CL3 | ... | CLN |
|----- | ----| ---- | ----| --- | --- |
|peak_1 |  1  |   0  |   0 | ... | 1  |
|peak_2 |  0  |   0  |   1 | ... | 0  |
|...			                            |
|peak_m |  0  |   1  |   1 | ... | 0  |


## Model Training 
We train a model for each TF-AB pair. 

### Stage 1
The stage 1 models predict TF binding across corresponding cell types for a specific TF-AB pair. The ``models`` directory contains the model (CNN_Multilabel) for stage 1 training. This function uses the Ax hyperparameter optimizer to select the best hyperparameters for the TF-AB model. Then trains the model using the best hyperparameters.

``python classification/stage1_pipeline.py --TF $TF --AB $AB 
                                          --chipData $chipData
                                          --hyperparameterPath $hyperparameterPath 
                                          --saveModelPath $saveModelPath``
       
 For other inputs, refer to:     
``python stage1_pipeline.py --help``     

### Stage 2

## Motif Enrichment

## Citation
If you use this code for your research, please cite our paper:

