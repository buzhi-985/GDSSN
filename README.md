# GDSSN
# Introduction
GDSSN:graph convolution network-based deep self-supervised learning method for cancer outcome prediction with multi-omics data.The multi-omics data was input into GDSSN to obtain representative composite features that can best rebuild all input data. These features were then input to the elastic-net-regularized Cox proportional hazards (Cox-EN) model to estimate the patients’ prognosis risks. Finally, in order to reduce the number of features for cancer prognosis prediction, we employed XGboost for features selection, and reconstruct the cancer prognosis model.

# Requirements
python 3,pytorch
# Data preparation
In this study, we utilized cancer datasets from the TCGA portal (https://tcga-data.nci.nih.gov/tcga/). All these datasets were downloaded by using the R package “TCGA-assembler”(v1.0.3, (Wei, et al., 2018)).Then we use the information from KEGG gene connection pathways to screen the features of the multi-omics data and generate the adjacency matrix.
# Usage

Multiomics data and adjacency matrices are fed into GDSSN.py and then representative composite features are obtained that best reconstruct all input multiomics data for downstream analysis. The generated features are saved in the FinalFeat folder.

# Example

For ease of use, we present sample data: paad.h5ad and paad_matrix.csv, which are processed by preprocess\dataprocess.py.Then we use the following statement to launch the model GDSSN and get the reconstructed features.

```shell
python GDSSN.py --input_h5ad_path="preprocess/data/preprocessed/paad.h5ad" --input_edge_path="preprocess/data/preprocessed/paad_matrix.csv" --epoch
s 200 -b 512 --lr 0.01 --cos --gpu 0 --low_dim 200

```

