#Import packages
import sys
import torch

seed = 42
torch.manual_seed(seed)
torch.set_printoptions(precision=2, sci_mode=False)


import pandas as pd
import numpy as np
import ast
import copy

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
from fiora.GNN.Trainer import Trainer
import torch_geometric as geom
from sklearn.metrics import r2_score
import scipy
from rdkit import RDLogger
from fiora.GNN.GNNModules import GNNCompiler
from fiora.GNN.Losses import WeightedMSELoss, WeightedMSEMetric, WeightedMAELoss, WeightedMAEMetric
from fiora.MS.SimulationFramework import SimulationFramework
RDLogger.DisableLog('rdApp.*')
from sklearn.model_selection import train_test_split
print(f'Working with Python {sys.version}')


#Import data
# key map to read metadata from pandas DataFrame
metadata_key_map = {
                "name": "Name",
                "collision_energy":  "CE", 
                "instrument": "Instrument_type",
                "ionization": "Ionization",
                "precursor_mz": "PrecursorMZ",
                "precursor_mode": "Precursor_type",
                "retention_time": "RETENTIONTIME",
                "ccs": "CCS"
                }


#
# Load specified libraries and align metadata
#
test_run=False
    
def load_training_data(input_file):
    L = LibraryLoader()
    df = L.load_from_csv(input_file)
    return df
input_data=sys.argv[1]
df = load_training_data(input_data)  
dict_columns = ["peaks", "summary"]
for col in dict_columns:
    df[col] = df[col].apply(lambda x: ast.literal_eval(x.replace('nan', 'None')))
    #df[col] = df[col].apply(ast.literal_eval)



df['group_id'] = df['group_id'].astype(int)

import torch
from fiora.GNN.Trainer import Trainer
import torch_geometric as geom

if torch.cuda.is_available(): 
 dev = "cuda:0"
else: 
 dev = "cpu" 
 
print(f"Running on device: {dev}")

filter_spectra = True
CE_upper_limit = 100.0
weight_upper_limit = 1000.0


if test_run:
    df = df.iloc[5000:6000,:]
    #df = df.iloc[5000:20000,:]



df["Metabolite"] = df["SMILES"].apply(Metabolite)
df["Metabolite"].apply(lambda x: x.create_molecular_structure_graph())

node_encoder = AtomFeatureEncoder(feature_list=["symbol", "num_hydrogen", "ring_type"])
bond_encoder = BondFeatureEncoder(feature_list=["bond_type", "ring_type"])
setup_encoder = SetupFeatureEncoder(feature_list=["collision_energy", "molecular_weight", "precursor_mode", "instrument"])
rt_encoder = SetupFeatureEncoder(feature_list=["molecular_weight", "precursor_mode", "instrument"])

if filter_spectra:
    setup_encoder.normalize_features["collision_energy"]["max"] = CE_upper_limit 
    setup_encoder.normalize_features["molecular_weight"]["max"] = weight_upper_limit 
    rt_encoder.normalize_features["molecular_weight"]["max"] = weight_upper_limit 
df["Metabolite"].apply(lambda x: x.compute_graph_attributes(node_encoder, bond_encoder))

df["summary"] = df.apply(lambda x: {key: x[name] for key, name in metadata_key_map.items()}, axis=1)
df.apply(lambda x: x["Metabolite"].add_metadata(x["summary"], setup_encoder, rt_encoder), axis=1)

if filter_spectra:
    num_ori = df.shape[0]
    correct_energy = df["Metabolite"].apply(lambda x: x.metadata["collision_energy"] <= CE_upper_limit and x.metadata["collision_energy"] > 1) 
    df = df[correct_energy]
    correct_weight = df["Metabolite"].apply(lambda x: x.metadata["molecular_weight"] <= weight_upper_limit)
    df = df[correct_weight]    
    print(f"Filtering spectra ({num_ori}) down to {df.shape[0]}")
    print(df["Precursor_type"].value_counts())
#df["pp"] = df["Metabolite"].apply(lambda x: x.match_stats["precursor_prob"])

model_params = {
    'gnn_type': 'RGCNConv',
    'depth': 6,
    'hidden_dimension': 300,
    'dense_layers': 2,
    'embedding_aggregation': 'concat',
    'embedding_dimension': 300,
    'input_dropout': 0.2,
    'latent_dropout': 0.1,
    'node_feature_layout': node_encoder.feature_numbers,
    'edge_feature_layout': bond_encoder.feature_numbers,    
    'static_feature_dimension': geo_data[0]["static_edge_features"].shape[1],
    'static_rt_feature_dimension': geo_data[0]["static_rt_features"].shape[1],
    'output_dimension': len(DEFAULT_MODES) * 2, # per edge 
}
training_params = {
    'epochs': 200 if not test_run else 20, #180,
    'batch_size': 256, #128,
    'train_val_split': 0.90,
    'learning_rate': 0.0004,#0.001,
    'with_RT': True, # TODO CHANGED
    'with_CCS': True
}
model = GNNCompiler(model_params).to(dev)


def train_val_test_split(keys, test_size=0.1, val_size=0.1, rseed=seed):
    temp_keys, test_keys = train_test_split(keys, test_size=test_size, random_state=rseed)
    adjusted_val_size = val_size / (1 - test_size)
    train_keys, val_keys = train_test_split(temp_keys, test_size=adjusted_val_size, random_state=rseed)
    
    return train_keys, val_keys, test_keys

# Make sure that the example is in the test split


group_ids = df["group_id"].astype(int)
keys = np.unique(group_ids)
example_not_in_test_split = True

for i in range(100):
    train, val, test = train_val_test_split(keys, rseed=seed + i)
df["pp"] = df["Metabolite"].apply(lambda x: (x.precursor_count / (sum(x.compiled_countsALL) / 2.0)).tolist() )
df["dataset"] = df["group_id"].apply(lambda x: 'train' if x in train else 'validation' if x in val else 'test' if x in test else 'VALUE ERROR')

df_test = df[df["dataset"] == "test"]
df_train = df[df["dataset"].isin(["traini", "validation"])]
y_label = 'compiled_probsALL'
train_keys, val_keys = df[df["dataset"] == "train"]["group_id"].unique(), df[df["dataset"] == "validation"]["group_id"].unique()

trainer = Trainer(geo_data, y_tag=y_label, problem_type="regression", metric_dict={"mse": WeightedMSEMetric}, train_keys=train_keys, val_keys=val_keys, split_by_group=True, seed=seed, device=dev)
optimizer = torch.optim.Adam(model.parameters(), lr=training_params["learning_rate"])
#scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.98)    
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience = 8, factor=0.5, mode = 'min', verbose  = True)

loss_fn = WeightedMSELoss()
#loss_fn = torch.nn.MSELoss()

tag = "test"
checkpoints = trainer.train(model, optimizer, loss_fn, scheduler=scheduler, batch_size=training_params['batch_size'], epochs=training_params["epochs"], val_every_n_epochs=1, with_CCS=training_params["with_CCS"], with_RT=training_params["with_RT"], masked_validation=False, tag=tag)
