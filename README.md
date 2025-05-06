# ms_model_fiora

Information:

This is a based on the original paper:
Nowatzky, Y., Russo, F.F., Lisec, J. et al. FIORA: Local neighborhood-based prediction of compound mass spectra from single fragmentation events. Nat Commun 16, 2298 (2025). https://doi.org/10.1038/s41467-025-57422-4

This version is modify to fit data from TIMS-TOF 

## Requirements

Developed and tested with the following systems and versions:
* Debian GNU/Linux 11 (bullseye)
* Python 3.10.8
* GCC 11.2.0


## Installation

Installation guide for the Fiora Python package:

Clone the project folder 

    git clone https://github.com/Micholms/ms_model_fiora

(Optional) Create a new conda environment

    conda create -n fiora python=3.10.8
    conda activate fiora

Change into the project directory (`cd fiora`). Then, install the package by using the setup.py via

    pip install .

## Train model

Use the train_model.py file

    !python /notebooks/train_model.py -i test.csv -l 0.001 -e 50 -t "RGCNConv_1"     #-m ../../checkpoint_Mona_only.best.pt
    
Where 
-  -i is the input csv used for training, could with advantage be generated through the tims_tof_data_extraction packaged (see: https://github.com/Micholms/tims_tof_data_extraction)
-  -l is the starting learning rate
-   -e the number of epochs
-   -t the tag of the model
-   -m is an option to add a pre-trained model.
    
## Evaulate model
Evaluation of the model can be done through the 
"evaulation_model.ipynb" under "notebooks". 

## Predict new molecules


## Install Docker
