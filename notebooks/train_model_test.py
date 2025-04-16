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
sys.path.append("../'MS models'/fiora/")
from os.path import expanduser
home = expanduser("~")
from fiora.MOL.constants import DEFAULT_MODES
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

#
# Load specified libraries and align metadata
#

    
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
def get_encoders():
    CE_upper_limit = 100.0
    weight_upper_limit = 1900.0
    node_encoder = AtomFeatureEncoder(feature_list=["symbol", "num_hydrogen", "ring_type"])
    bond_encoder = BondFeatureEncoder(feature_list=["bond_type", "ring_type"])
    setup_encoder = SetupFeatureEncoder(feature_list=["collision_energy", "molecular_weight", "precursor_mode", "instrument"])#, sets_overwrite=overwrite_setup_features)
    rt_encoder = SetupFeatureEncoder(feature_list=["molecular_weight", "precursor_mode", "instrument"])#, sets_overwrite=overwrite_setup_features)
    
    setup_encoder.normalize_features["collision_energy"]["max"] = CE_upper_limit 
    setup_encoder.normalize_features["molecular_weight"]["max"] = weight_upper_limit 
    rt_encoder.normalize_features["molecular_weight"]["max"] = weight_upper_limit 
    return node_encoder,bond_encoder,setup_encoder,rt_encoder

    
def filter_CE_weight(df):
    print("Filtering based on CE and weight")
    CE_upper_limit = 100.0
    weight_upper_limit = 1900.0

    df["Metabolite"] = df["SMILES"].apply(Metabolite)
    df["Metabolite"].apply(lambda x: x.create_molecular_structure_graph())
    
    node_encoder,bond_encoder,setup_encoder,rt_encoder=get_encoders()
    
    df["Metabolite"].apply(lambda x: x.compute_graph_attributes(node_encoder, bond_encoder))
    df.apply(lambda x: x["Metabolite"].set_id(x["group_id"]) , axis=1)
    
    df["summary"] = df.apply(lambda x: {key: x[name] for key, name in metadata_key_map.items()}, axis=1)
    df.apply(lambda x: x["Metabolite"].add_metadata(x["summary"], setup_encoder, rt_encoder), axis=1)
    df.apply(lambda x: x["Metabolite"].set_loss_weight(x["loss_weight"]), axis=1)
    num_ori = df.shape[0]
    correct_energy = df["Metabolite"].apply(lambda x: x.metadata["collision_energy"] <= CE_upper_limit and x.metadata["collision_energy"] > 1) 
    df = df[correct_energy]
    correct_weight = df["Metabolite"].apply(lambda x: x.metadata["molecular_weight"] <= weight_upper_limit)
    df = df[correct_weight]    
    print(f"Filtering spectra ({num_ori}) down to {df.shape[0]}")
    
    return df


def filter_peaks(df):
    print("Filtering low peak counts")
    df["Metabolite"].apply(lambda x: x.fragment_MOL(depth=1))
    df.apply(lambda x: x["Metabolite"].match_fragments_to_peaks(x["peaks"]["mz"], x["peaks"]["intensity"], tolerance=x["ppm_peak_tolerance"]), axis=1)
    df["num_peak_matches"] = df["Metabolite"].apply(lambda x: x.match_stats["num_peak_matches"])
    print("Removed ", sum(df["num_peak_matches"] < 2), "due to less then 2 peaks")
    df = df[df["num_peak_matches"] >= 2]
    return df

def prep_data(df):
    df_test = df[df["dataset"] == "test"]
    df_train = df[df["dataset"].isin(["train", "validation"])]
    geo_data = df_train["Metabolite"].apply(lambda x: x.as_geometric_data().to(dev)).values
    return df_test,df_train,geo_data

def set_model_params(df):
    node_encoder,bond_encoder,setup_encoder,rt_encoder=get_encoders()
    geo_data=prep_data(df)[2]
    
    model_params = {
    'param_tag': 'default',
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
    
    # Keep track of encoded features
    'atom_features': node_encoder.feature_list,
    'atom_features': bond_encoder.feature_list,
    'setup_features': setup_encoder.feature_list,
    'setup_features_categorical_set': setup_encoder.categorical_sets,
    'rt_features': rt_encoder.feature_list,
    
    # Set default flags (May be overwritten below)
    'rt_supported': False,
    'ccs_supported': False,
    'version': "x.x.x"
    
    }

    training_params = {
    'epochs': args.epochs, 
    'batch_size': 128,
    #'train_val_split': 0.90,
    'learning_rate': args.lr, # 0.00001 currently for wMAE # Default for wMSE is 0.0004, #0.001,
    'with_RT': False, # Turn off RT/CCS for initial trainings round
    'with_CCS': False
    }
    return model_params,training_params
    
def train_val_test_split(keys, test_size=0.1, val_size=0.1, rseed=seed):
    temp_keys, test_keys = train_test_split(keys, test_size=test_size, random_state=rseed)
    adjusted_val_size = val_size / (1 - test_size)
    train_keys, val_keys = train_test_split(temp_keys, test_size=adjusted_val_size, random_state=rseed)
    
    return train_keys, val_keys, test_keys

def add_grouping(df):
    print("Split into training, testing and validation")
    group_ids = df["group_id"].astype(int)
    keys = np.unique(group_ids)
    example_not_in_test_split = True
    
    for i in range(100):
        train, val, test = train_val_test_split(keys, rseed=seed + i)
    df["dataset"] = df["group_id"].apply(lambda x: 'train' if x in train else 'validation' if x in val else 'test' if x in test else 'VALUE ERROR')
    return df


def simulate_all(model, DF):
    return fiora.simulate_all(DF, model)

    
def test_model(model, DF, score="spectral_sqrt_cosine", return_df=False):  ##### Change to true to save df w all predictions
    dft = simulate_all(model, DF)
    
    if return_df:
        return dft
    return dft[score].values
    
def train_new_model(continue_with_model=None):
        if continue_with_model:
            model = continue_with_model.to(dev)
            
        else:
            model = GNNCompiler(model_params).to(dev)
        y_label = 'compiled_probsALL'
     
        optimizer = torch.optim.Adam(model.parameters(), lr=training_params["learning_rate"])
        
        train_keys, val_keys = df[df["dataset"] == "train"]["group_id"].unique(), df[df["dataset"] == "validation"]["group_id"].unique()
        trainer = Trainer(geo_data, y_tag=y_label, problem_type="regression", train_keys=train_keys, val_keys=val_keys, metric_dict=metric_dict, split_by_group=True, seed=seed, device=dev)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience = 8, factor=0.5, mode = 'min', verbose = True)
    
        checkpoints = trainer.train(model, optimizer, loss_fn, scheduler=scheduler, batch_size=training_params['batch_size'], epochs=training_params["epochs"], val_every_n_epochs=1, with_CCS=training_params["with_CCS"], with_RT=training_params["with_RT"], masked_validation=False, tag=tag) 
        print(checkpoints)
        return model, checkpoints

import argparse
parser = argparse.ArgumentParser()

#-db DATABASE -u USERNAME -p PASSWORD -size 20
parser.add_argument("-l", "--lr", help="Start learning rate",type=float)
parser.add_argument("-e", "--epochs", help="Number of epochs",type=int)
parser.add_argument("-i", "--input", help="Data to train on (csv)")
parser.add_argument("-m", "--model", help="Model to fine tune, else leave blank")
parser.add_argument("-t", "--tag", help="Name of model")
args = parser.parse_args()



if __name__ == '__main__':
    input_data=args.input
    df = load_training_data(input_data)
    df=restore_dict(df)
    df=filter_CE_weight(df)
    df=filter_peaks(df)
    df=add_grouping(df)
    if torch.cuda.is_available(): 
        torch.cuda.empty_cache()
        dev = "cuda:0"
    else: 
        dev = "cpu" 
    
    
    print(f"Running on device: {dev}")

    print(f"Training model")
    
    fiora = SimulationFramework(None, dev=dev, with_RT=False, with_CCS=False)
    
    model_params,training_params=set_model_params(df)
    
    df_test,df_train,geo_data=prep_data(df)
    
    print(df.groupby("dataset")["group_id"].unique().apply(len))
    print(f"Prepared training/validation with {len(geo_data)} data points")
    np.seterr(invalid='ignore')
    tag = args.tag
    metric_dict= {"mse": WeightedMSEMetric} #WeightedMSEMetric
    loss_fn = WeightedMSELoss() # WeightedMSELoss()

    if args.model:
       
        MODEL_PATH=args.model  #"../Checkpoints/checkpoint_training.best.pt"
    
        try:
            model = GNNCompiler.load_from_state_dict(MODEL_PATH)
        except:
            raise NameError("Error: Failed loading from state dict.")   
            #model, checkpoints = train_new_model() # continue_with_model=model)
       
        model_params=model.model_params
        training_params=set_model_params(df)[1]
        model, checkpoints = train_new_model(continue_with_model=model)
    else: 
         model, checkpoints = train_new_model()
    model_end = copy.deepcopy(model)
    model = model_end#GNNCompiler.load(checkpoints["file"]).to(dev)
    
    score = "spectral_sqrt_cosine"
    
    val_results = test_model(model, df_train[df_train["dataset"]== "validation"], score=score)
    test_results = test_model(model, df_test, score=score)
    results = [{"model": model, "validation": val_results, "test": test_results}]
    print("Median ", score,"for the test set is", np.median(test_results))
    