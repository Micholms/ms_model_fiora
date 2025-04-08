import sys
print(f'Working with Python {sys.version}')
import seaborn as sns
import numpy as np
import pandas as pd
import importlib
import matplotlib.pyplot as plt
import seaborn as sns
import collections
import time
import os
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import Draw
import rdkit.Chem.Descriptors as Descriptors
from rdkit.Chem import PandasTools
import torch
seed = 42
torch.manual_seed(seed)
torch.set_printoptions(precision=2, sci_mode=False)
import ast

# Deep Learning
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import scipy

from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')

# Load Modules
sys.path.append("..")
from os.path import expanduser
home = expanduser("~")
import fiora.IO.mspReader as mspReader
import fiora.visualization.spectrum_visualizer as sv
import fiora.IO.molReader as molReader
from fiora.MOL.constants import ADDUCT_WEIGHTS, DEFAULT_PPM, PPM, DEFAULT_MODES
from fiora.MOL.collision_energy import align_CE
from fiora.MOL.Metabolite import Metabolite
from fiora.IO.LibraryLoader import LibraryLoader
from fiora.MOL.FragmentationTree import FragmentationTree 
from fiora.GNN.AtomFeatureEncoder import AtomFeatureEncoder
from fiora.GNN.BondFeatureEncoder import BondFeatureEncoder
from fiora.GNN.SetupFeatureEncoder import SetupFeatureEncoder

def read_rawfile(input_df):
    print("Reading ", input_df)
    df_msp = mspReader.read(input_df)
    df = pd.DataFrame(df_msp)
    return df

def filter_type(df):
    
    df = df[df["Spectrum_type"] == "MS2"]
    target_precursor_type = ["[M+H]+", "[M-H]-", "[M+H-H2O]+", "[M+Na]+"]
    df = df[df["Precursor_type"].apply(lambda ptype: ptype in target_precursor_type)]
    return df

def format_df(df):
    print("Formating dataset")
    df_mona=df.copy()
    df_mona['PrecursorMZ'] = df_mona["PrecursorMZ"].str.replace(',', '.')
    df_mona['PrecursorMZ'] = df_mona["PrecursorMZ"].astype('float')
    df_mona['Num Peaks'] = df_mona["Num Peaks"].astype('int')
    return df_mona

def get_mol(df,ID):
    x = df.loc[ID]
    comments=x["Comments"].split()
    smiles=False
    for j in comments:
        if smiles:
            continue
        else:
            if "SMILES" in str(j):
                if str(j)[5]=="S":
                    smiles=str(j[7:-1])
                else:
                    smiles=str(j[8:-1])
    if smiles:        
        mol=Chem.MolFromSmiles(smiles)
    else: 
        mol=np.nan
        smiles=np.nan
    return mol,smiles

def convert_info(df):
    print("Fetch SMILES and interpret InChIKeys")
    df_mona=df.copy()
    df_mona= df_mona[~df_mona["InChIKey"].isnull()] 
    MOL=[]
    SMILES=[]
    for i in df_mona.index:
        mol,smiles=get_mol(df_mona,i)
           
        MOL=MOL+[mol]
        SMILES=SMILES+[smiles]
    df_mona["MOL"] = MOL
    df_mona["SMILES"]=SMILES
    print(f"Successfully interpreted {sum(df_mona['MOL'].notna())} from {df_mona.shape[0]} entries. Dropping the rest.")
    df_mona = df_mona[df_mona['MOL'].notna()]
    #df_mona["SMILES"] = df_mona["MOL"].apply(Chem.MolToSmiles)
    df_mona["InChI"] = df_mona["MOL"].apply(Chem.MolToInchi)
    df_mona["K"] = df_mona["MOL"].apply(Chem.MolToInchiKey)
    df_mona["ExactMolWeight"] = df_mona["MOL"].apply(Chem.Descriptors.ExactMolWt)
    return df_mona

def check_InChIKey(df_mona):
    print("Checking if interpreted InChIKeys are correct")
    correct_keys = df_mona.apply(lambda x: x["InChIKey"] == x["K"], axis=1) #Check so that computed keys are same as original
    s = "confirmed!" if correct_keys.all() else "not confirmed !! Attention!"
    print(f"Confirming whether computed and provided InChI-Keys are correct. Result: {s} ({correct_keys.sum()/len(correct_keys):0.2f} correct)")
    half_keys = df_mona.apply(lambda x: x["InChIKey"].split('-')[0] == x["K"].split('-')[0], axis=1)
    s = "confirmed!" if half_keys.all() else "not confirmed !! Attention!"
    print(f"Checking if main layer InChI-Keys are correct. Result: {s} ({half_keys.sum()/len(half_keys):0.3f} correct)")
    
    print("Dropping all other.")
    df_mona["matching_key"] = df_mona.apply(lambda x: x["InChIKey"] == x["K"], axis=1)
    df_mona = df_mona[df_mona["matching_key"]]
    return df_mona
    
def filter_peaks(df,MIN_PEAKS,PRECURSOR_TYPES):
    print("Filtering based on number of peaks")
    df_mona=df.copy()
    df_mona = df_mona[df_mona["Num Peaks"] > MIN_PEAKS]
    df_mona["theoretical_precursor_mz"] = df_mona["ExactMolWeight"] + df_mona["Precursor_type"].map(ADDUCT_WEIGHTS)
    df_mona = df_mona[df_mona["Precursor_type"].apply(lambda ptype: ptype in PRECURSOR_TYPES)]
    df_mona["precursor_offset"] = df_mona["PrecursorMZ"] - df_mona["theoretical_precursor_mz"]
    print(f"Shape {df_mona.shape}")
    return df_mona
    
def compute_graph(df):
    print("Computing molecular structure graph")
    df_mona=df.copy()
    TOLERANCE = 200 * PPM
    df_mona["Metabolite"] = df_mona["SMILES"].apply(Metabolite)
    df_mona["Metabolite"].apply(lambda x: x.create_molecular_structure_graph())
    df_mona["Metabolite"].apply(lambda x: x.compute_graph_attributes())
    to_drop=[]
    for i,name in enumerate(df_mona.index):
        try:
            df_mona.loc[name,"Metabolite"]=df_mona.loc[name,"Metabolite"].fragment_MOL()
        except:
            to_drop=to_drop+[name]
    df_mona=df_mona.drop(index=to_drop)
    df_mona.apply(lambda x: x["Metabolite"].match_fragments_to_peaks(x["peaks"]["mz"], x["peaks"]["intensity"], tolerance=TOLERANCE), axis=1)
    return df_mona    
    
def CE_filtering(df):
    print("Filtering on Collision energy")
    df_mona=df.copy()
    df_mona["CE"] = df_mona.apply(lambda x: align_CE(x["Collision_energy"], x["theoretical_precursor_mz"]), axis=1) #modules.MOL.collision_energy.align_CE) 
    df_mona["CE_type"] = df_mona["CE"].apply(type)
    df_mona["CE_derived_from_NCE"] = df_mona["Collision_energy"].apply(lambda x: "%" in str(x))
    
    print("Distinguish CE absolute values (eV - float) and normalized CE (in % - str format)")
    print(df_mona["CE_type"].value_counts())
    
    print("Removing all but absolute values")
    df_mona = df_mona[df_mona["CE_type"] == float]
    df_mona = df_mona[~df_mona["CE"].isnull()]

    print(f'Detected {len(df_mona["CE"].unique())} unique collision energies in range from {np.min(df_mona["CE"])} to {max(df_mona["CE"])} eV')
    
    df_mona=compute_graph(df_mona)
    return df_mona
    

    
def match_framgments_and_peaks(df):
    print("Matching fragments to peaks")
    df_mona=df.copy()
    df_mona["peak_matches"] = df_mona["Metabolite"].apply(lambda x: getattr(x, "peak_matches"))
    df_mona["num_peaks_matched"] = df_mona["peak_matches"].apply(len)
    
    def get_match_stats(matches, mode_count={m: 0 for m in DEFAULT_MODES}):
        num_unique, num_conflicts = 0, 0
        for mz, match_data in matches.items():
            ion_modes = match_data["ion_modes"]
            if len(ion_modes) == 1:
                num_unique += 1
            elif len(ion_modes) > 1:
                num_conflicts += 1
            for c in ion_modes:
                mode_count[c[0]] += 1
        return num_unique, num_conflicts, mode_count
    
    df_mona["match_stats"] = df_mona["peak_matches"].apply(lambda x: get_match_stats(x))
    df_mona["num_unique_peaks_matched"] = df_mona.apply(lambda x: x["match_stats"][0], axis=1)
    df_mona["num_conflicts_in_peak_matching"] = df_mona.apply(lambda x: x["match_stats"][1], axis=1)
    df_mona["match_mode_counts"] = df_mona.apply(lambda x: x["match_stats"][2], axis=1)
    u= df_mona["num_unique_peaks_matched"].sum() 
    s= df_mona["num_conflicts_in_peak_matching"].sum() 
    print(f"Total number of uniquely matched peaks: {u} , conflicts found within {s} matches ({100 * s / (u+s):.02f} %))")
    print(f"Total number of conflicting peak to fragment matches: {s}")
    return df_mona 
    
def add_metadata(df):
    print("Adding various metadata")
    orbitrap_nametags = ["Orbitrap"]
    qtof_nametags = ["QTOF", "LC-ESI-QTOF", "ESI-QTOF"]
    df["Instrument_type"] = df["Instrument_type"].apply(lambda x: "HCD" if x in orbitrap_nametags else "Q-TOF" if x in qtof_nametags else x)
    df["RETENTIONTIME"] = np.nan
    df["CCS"] = np.nan
    df["PPM_num"] = 50
    df["ppm_peak_tolerance"] = df["PPM_num"] * PPM
    df["lib"] = "MoNA"
    df["origin"] = "MoNA"
    df["Ionization"] = "ESI"
   
    return df

def add_identifiers(df):
    print("Assigning unique metabolite identifiers.")
    
    metabolite_id_map = {}
    
    for metabolite in df["Metabolite"]:
        is_new = True
        for id, other in metabolite_id_map.items():
            if metabolite == other:
                metabolite.set_id(id)
                is_new = False
                break
        if is_new:
            new_id = len(metabolite_id_map)
            metabolite.id = new_id
            metabolite_id_map[new_id] = metabolite
    
    df["group_id"] = df["Metabolite"].apply(lambda x: x.get_id())
    df["num_per_group"] = df["group_id"].map(df["group_id"].value_counts())
    
    for i, data in df.iterrows():
        data["Metabolite"].set_loss_weight(1.0 / data["num_per_group"])
    print(f"Found {len(metabolite_id_map)} unique molecular structures.")

    
def precursor_processing(df):
    print("Processing precursors")
    df["loss_weight"] = df["Metabolite"].apply(lambda x: x.loss_weight)
    df["Precursor_offset"] = df["PrecursorMZ"] - df.apply(lambda x: x["Metabolite"].ExactMolWeight + ADDUCT_WEIGHTS[x["Precursor_type"]], axis=1)
    df["Precursor_abs_error"] = abs(df["Precursor_offset"])
    df["Precursor_rel_error"] = df["Precursor_abs_error"] / df["PrecursorMZ"]
    df["Precursor_ppm_error"] = df["Precursor_abs_error"] / (df["PrecursorMZ"] * PPM)
    print((df["Precursor_ppm_error"] > df["PPM_num"]).sum(), "found with misaligned precursor. Removing these.")
    
    df = df[df["Precursor_ppm_error"] <= df["PPM_num"]]

if __name__ == '__main__':
    input_data=sys.argv[1]
    df = read_rawfile(input_data)
    df=filter_type(df)
    df=format_df(df)
    df=convert_info(df)
    df=check_InChIKey(df)
    df=filter_peaks(df, 2,["[M+H]+", "[M-H]-"])
    df=compute_graph(df)
    df=CE_filtering(df)
    df=match_fragments_and_peaks(df)
    df=add_metadata(df)
    df=add_identifiers(df)
    df=precursor_processing(df)



    



    df.to_csv("../datasplits_Feb25_test.csv")














    