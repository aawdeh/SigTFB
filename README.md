# SigTFB: Cell Type Specific DNA Signatures of Transcription Factor Binding (Work in progress)

This repository contains code for our paper "Cell Type Specific DNA Signatures of Transcription Factor Binding". The models are implemented in PyTorch.

## Data

All associated data from our paper can be downloaded from [here](https://doi.org/10.20383/103.0605)

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
The stage 1 models predict TF binding across corresponding cell types for a specific TF-AB pair. The ``models`` directory contains the model (CNN_Multilabel) for stage 1 training. This function uses the Ax hyperparameter optimizer to select the best hyperparameters for the TF-AB model, then trains the model using the best hyperparameters.

``python classification/stage1_pipeline.py --TF $TF --AB $AB 
                                          --chipData $chipData
                                          --hyperparameterPath $hyperparameterPath 
                                          --saveModelPath $saveModelPath``
       
 For other inputs, refer to:     
``python stage1_pipeline.py --help``     

### Stage 2
Using the pretrained stage 1 model for a specific TF-AB pair, the stage 2 model predicts TF binding for each cell type. It takes as input not only the genomic sequence for the ChIP-seq peak, but the cell type encoding as well. The ``models`` directory contains the model (NetModified_OneLayer_OneUnit) for stage 2 training. This function also uses the Ax hyperparameter optimizer to select the best hyperparameters for the TF-AB model, then trains the model using the best hyperparameters.

``python classification/stage2_pipeline.py --TF $TF --AB $AB 
                                          --chipData $chipData
                                          --model_stage1_path $model_stage1_path
                                          --hyperparameterPath $hyperparameterPath 
                                          --saveModelPath $saveModelPath``
       
 For other inputs, refer to:     
``python stage2_pipeline.py --help``     

## Citation
If you use this code for your research, please cite our paper:

