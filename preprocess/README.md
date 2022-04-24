# MERGE PEAKS ACROSS CELL LINES FOR A SPECIFIC TF-AB.

ChIP-seq files are organized such that if they correspond to the same TF and AB they are in same directory. 

For example:
If TF-AB has three CLs [3 ChIP-seq bed files each corresponding to a CL], they will organized in the following way:
 .../TF/AB/CL1.bed
 .../TF/AB/CL2.bed
 .../TF/AB/CL2.bed

|Peaks | CL1 |  CL2 | CL3 |
|----- | ----| ---- | ----|
|peak_1 |  1  |   0  |   0 |
|peak_2 |  0  |   0  |   1 |
|...			  |
|peak_m |  0  |   1  |   1 |

## Install 
- Bioinformatics:
	- bedtools (https://bedtools.readthedocs.io/en/latest/content/installation.html)
- Python 2.7
	- pip install pandas
	- pip install sklearn
	- pip install numpy
	- pip install h5py

## Input
- Genome sequence(https://www.gencodegenes.org/human/)

## Usage
Four main functions inspired by the Basset method used to merge peaks:

### Step 1. createTextFiles(...)	
								Creates text files for Basset to take as input. The text files contain the full paths of the ChIP-seq bed files.
								Format: CellName/CellID \t Path to bed file
							
### Step 2. call_Basset(...) 		
								Calls Basset preprocess_features.py to merge peaks. When calling the Basset function, the min peak
								overlap is set to 30 bps, and the sequence length set to 101.
								The outputs of this function are a merged bed file and an activity file -- where each row corresponds
								to a merged peak and the each column corresponds to a cell line. The activity file tells us whether 
								each peak is bound or unbound in each cell line.
								
### Step 3. convert2Fasta(...)		
								Converts the merged bed file per TF-AB to a fasta file using BEDTools [getfasta function]

### Step 4. convert2hdf5(...)		
								Calls Basset seq_hdf5.py to convert fasta file to hdf5 file to use as input for model.
								The peaks are randomly divided such that 5\% are used for testing, and the rest for training.
								The output consists of two hdf5 files -- one for testing and one for training.

