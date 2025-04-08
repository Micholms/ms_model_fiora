import sys
import torch


seed = 42
#torch.set_default_dtype(torch.float64)
torch.manual_seed(seed)
torch.set_printoptions(precision=2, sci_mode=False)


import pandas as pd
import numpy as np
import ast
import copy
import matplotlib.pyplot as plt 
import seaborn as sns

# Load Modules
sys.path.append("..")
from os.path import expanduser
home = expanduser("~")
from fiora.MOL.constants import DEFAULT_PPM, PPM, DEFAULT_MODES
from fiora.IO.LibraryLoader import LibraryLoader
from fiora.MOL.FragmentationTree import FragmentationTree
import fiora.visualization.spectrum_visualizer as sv
from fiora.MOL.Metabolite import Metabolite
from fiora.GNN.AtomFeatureEncoder import AtomFeatureEncoder
from fiora.GNN.BondFeatureEncoder import BondFeatureEncoder
from fiora.GNN.SetupFeatureEncoder import SetupFeatureEncoder
from fiora.GNN.GNNModules import GNNCompiler
from fiora.MS.SimulationFramework import SimulationFramework

from fiora.MOL.collision_energy import NCE_to_eV
from fiora.MS.spectral_scores import spectral_cosine, spectral_reflection_cosine, reweighted_dot
from fiora.MS.ms_utility import merge_annotated_spectrum
from sklearn.metrics import r2_score
import scipy
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')
from sklearn.model_selection import train_test_split
print(f'Working with Python {sys.version}')




MODEL_PATH=sys.argv[1]
try:
    model = GNNCompiler.load_from_state_dict(MODEL_PATH)
except:
    raise NameError("Error: Failed loading from state dict.")
   

# key map to read metadata from pandas DataFrame
metadata_key_map = {
                "name": "Name",
                 "collision_energy":  "CE", 
                "instrument": "Instrument_type",
                "ionization": "Ionization",
               "precursor_mz": "PrecursorMZ",
                "precursor_mode": "Precursor_type",      
                 "retention_time": "RT",
                 "ccs": "CCS"
                 }


def load_training_data(input_file):
    L = LibraryLoader()
    df = L.load_from_csv(input_file)
    return df
    
def restore_dict(df):
    dict_columns = ["peaks"]#, "summary"]
    for col in dict_columns:
        df[col] = df[col].apply(lambda x: ast.literal_eval(x.replace('nan', 'None')))
    df['group_id'] = df['group_id'].astype(int)
    
    return df

def train_val_test_split(keys, test_size=0.1, val_size=0.1, rseed=seed): ######################################### Dont forget to change ( 0.1 0.1)
    temp_keys, test_keys = train_test_split(keys, test_size=test_size, random_state=rseed)
    adjusted_val_size = val_size / (1 - test_size)
    train_keys, val_keys = train_test_split(temp_keys, test_size=adjusted_val_size, random_state=rseed)
    
    return train_keys, val_keys, test_keys
def add_grouping(df):
    print("Split into training, testing and validation")
    group_ids = df["group_id"].astype(int)
    keys = np.unique(group_ids)
    example_not_in_test_split = True
    
    for i in range(len(df)):
        train, val, test = train_val_test_split(keys, rseed=seed + i)
    df["dataset"] = df["group_id"].apply(lambda x: 'train' if x in train else 'validation' if x in val else 'test' if x in test else 'VALUE ERROR')
    return df

df = load_training_data(sys.argv[2])
df=restore_dict(df)
df=add_grouping(df)
print(df.groupby("lib")["group_id"].unique().apply(len))


df["Metabolite"] = df["SMILES"].apply(Metabolite)
df["Metabolite"].apply(lambda x: x.create_molecular_structure_graph())

node_encoder = AtomFeatureEncoder(feature_list=["symbol", "num_hydrogen", "ring_type"])
bond_encoder = BondFeatureEncoder(feature_list=["bond_type", "ring_type"])
setup_encoder = SetupFeatureEncoder(feature_list=["collision_energy", "molecular_weight"])
df["Metabolite"].apply(lambda x: x.compute_graph_attributes(node_encoder, bond_encoder))
df["summary"] = df.apply(lambda x: {key: x[name] for key, name in metadata_key_map.items()}, axis=1)

#df.apply(lambda x: x["Metabolite"].add_metadata(x["summary"], setup_encoder=None), axis=1)

print(df.groupby("dataset")["group_id"].unique().apply(len))

print("Reducing data to test set.")
df_train = df[df["dataset"] != "test"]
df_test = df[df["dataset"] == "test"]


CE_upper_limit = 100.0
weight_upper_limit = 1900.0

node_encoder = AtomFeatureEncoder(feature_list=["symbol", "num_hydrogen", "ring_type"])
bond_encoder = BondFeatureEncoder(feature_list=["bond_type", "ring_type"])
model_setup_feature_sets = None
if "setup_features_categorical_set" in model.model_params.keys():
    print("hi")
    model_setup_feature_sets = model.model_params["setup_features_categorical_set"]    
    # TODO Refactor this:
    for i, data in df_test.iterrows():
        df_test.loc[i]["summary"]["instrument"] = "HCD"
setup_encoder = SetupFeatureEncoder(feature_list=["collision_energy", "molecular_weight", "precursor_mode", "instrument"], sets_overwrite=model_setup_feature_sets)
rt_encoder = SetupFeatureEncoder(feature_list=["molecular_weight", "precursor_mode", "instrument"], sets_overwrite=model_setup_feature_sets)


def process_dataframes(df_train, df_test):

 
    setup_encoder.normalize_features["collision_energy"]["max"] = CE_upper_limit 
    setup_encoder.normalize_features["molecular_weight"]["max"] = weight_upper_limit 
    rt_encoder.normalize_features["molecular_weight"]["max"] = weight_upper_limit 

    df_test["Metabolite"].apply(lambda x: x.compute_graph_attributes(node_encoder, bond_encoder))
    df_test.apply(lambda x: x["Metabolite"].set_id(x["group_id"]) , axis=1)

    df["summary"] = df.apply(lambda x: {key: x[name] for key, name in metadata_key_map.items()}, axis=1)
    df_test.apply(lambda x: x["Metabolite"].add_metadata(x["summary"], setup_encoder, rt_encoder), axis=1)

    df_test["Metabolite"].apply(lambda x: x.fragment_MOL(depth=1))
    df_test.apply(lambda x: x["Metabolite"].match_fragments_to_peaks(x["peaks"]["mz"], x["peaks"]["intensity"], tolerance=x["ppm_peak_tolerance"]), axis=1)

    return df_train, df_test
    
df_train, df_test = process_dataframes(df_train, df_test)

from fiora.GNN.Trainer import Trainer
import torch_geometric as geom

if torch.cuda.is_available(): 
 dev = "cuda:0"
else: 
 dev = "cpu" 

print(f"Running on device: {dev}")
model.eval()
model = model.to(dev)

fiora = SimulationFramework(None, dev=dev, with_RT=True, with_CCS=True)
np.seterr(invalid='ignore')
def simulate_all(model, DF):
    return fiora.simulate_all(DF, model)

    
def test_model(model, DF):
    dft = simulate_all(model, DF)
    return dft
print(f"Testing the model")

df_test = test_model(model, df_test)

print("Done")
# Default score
score = "spectral_sqrt_cosine"
#^score = "spectral_sqrt_cosine_wo_prec"
avg_func = np.median
print(avg_func(df_test[score]))

fiora_res = {"model": "Fiora", "Test+": avg_func(df_test[df_test["Precursor_type"] == "[M+H]+"][score]), "Test-": avg_func(df_test[df_test["Precursor_type"] == "[M-H]-"][score])}#, "CASMI16+": avg_func(df_cas[df_cas["Precursor_type"] == "[M+H]+"][score]), "CASMI16-":avg_func(df_cas[df_cas["Precursor_type"] == "[M-H]-"][score])}#, "CASMI22+": avg_func(df_cas22[df_cas22["Precursor_type"] == "[M+H]+"][score]), "CASMI22-": avg_func(df_cas22[df_cas22["Precursor_type"] == "[M-H]-"][score])} 
             

summaryPos = pd.DataFrame( [fiora_res])#, cfm_id, ice_res])
print("Summary test sets")
print(summaryPos)