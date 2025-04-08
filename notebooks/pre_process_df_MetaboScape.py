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
import fiora.IO.mgfReader as mgfReader
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

def find_closest(numbers, target):
    return min(numbers, key=lambda x: abs(x- target))

def find_match(numbers,target, tolerance):
    closest=find_closest(numbers, target)
  
    if abs(closest-target)<tolerance:
        return closest


def count_matches(data,msms_list,target_list,tolerance):
    data=data.copy()
    match=[]
    not_found=[]
    target_mass=target_list["Exact Mass"]
    target_smiles=target_list["SMILES"]
    target_INCHIKEY=target_list["INCHIKEY"]
    target_name=target_list["Sample Name"]
    for i in target_list.index:
        m=find_match(msms_list,target_mass[i],tolerance)
        if m:
            match.append(target_mass[i])
            j=data.index[data['PEPMASS']==m].tolist()
            
            data.loc[j,"Target"]=float(target_mass[i])
            data.loc[j,"SMILES"]=target_smiles[i]
            data.loc[j,"INCHIKEY"]=target_INCHIKEY[i]
            data.loc[j,"Name"]=target_name[i]


            
           
        else:
            not_found.append(target_mass[i])
    #print(data.dropna(subset="Target").sort_values("Target"))
    print("Found", len(match), "/", len(target_mass), "matches at", tolerance)
    return match,not_found,data


def read_rawfile(input_df):
    print("Reading ", input_df)
    df_mgf = mgfReader.read(input_df)
    df = pd.DataFrame(df_mgf)
    return df
def filter_type(df):
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
    smiles=x["SMILES"]
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
    df_mona["Metabolite"].apply(lambda x: x.fragment_MOL(depth=1))
    df_mona.apply(lambda x: x["Metabolite"].match_fragments_to_peaks(x["peaks"]["mz"], x["peaks"]["intensity"], tolerance=TOLERANCE), axis=1)
    return df_mona    
    
def CE_filtering(df):
    print("Filtering on Collision energy")
    df_mona=df.copy()
    df_mona["Collision_energy"] ="20eV" 
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
    


def add_metadata(df):
    print("Adding various metadata")
    orbitrap_nametags = ["Orbitrap"]
    qtof_nametags = ["QTOF", "LC-ESI-QTOF", "ESI-QTOF"]
    df["Instrument_type"]="TimsTof"
    df["Instrument_type"] = df["Instrument_type"].apply(lambda x: "HCD" if x in orbitrap_nametags else "Q-TOF" if x in qtof_nametags else x)
    df["RETENTIONTIME"] = np.nan
    df["CCS"] = np.nan
    df["PPM_num"] = 50
    df["ppm_peak_tolerance"] = df["PPM_num"] * PPM
    df["lib"] = "TimsTof"
    df["origin"] = "TimsTof"
    df["Ionization"] = "ESI" #??????
   
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
    return df
    
def precursor_processing(df):
    
    print("Processing precursors")
    df["loss_weight"] = df["Metabolite"].apply(lambda x: x.loss_weight)
    df["Precursor_offset"] = df["PrecursorMZ"] - df.apply(lambda x: x["Metabolite"].ExactMolWeight + ADDUCT_WEIGHTS[x["Precursor_type"]], axis=1)
    df["Precursor_abs_error"] = abs(df["Precursor_offset"])
    df["Precursor_rel_error"] = df["Precursor_abs_error"] / df["PrecursorMZ"]
    df["Precursor_ppm_error"] = df["Precursor_abs_error"] / (df["PrecursorMZ"] * PPM)
    print((df["Precursor_ppm_error"] > df["PPM_num"]).sum(), "found with misaligned precursor. Removing these.")
    
    df = df[df["Precursor_ppm_error"] <= df["PPM_num"]]
    return df
def count_peaks(df):
    for i in (df.index):
        df.loc[i,"Num Peaks"]=int(len(df.loc[i,"peaks"]["mz"]))
    return df


    
def match_fragments_and_peaks(df):
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
    
import argparse
parser = argparse.ArgumentParser()

#-db DATABASE -u USERNAME -p PASSWORD -size 20
parser.add_argument("-t", "--target", help="Target values with metadata")
parser.add_argument("-m", "--metadata", help="MSMS metadata")
parser.add_argument("-s", "--spectrum", help="MSMS spectrum file (.mgf)")
parser.add_argument("-o", "--output", help="Output file name/path")

args = parser.parse_args()


if __name__ == '__main__':
    #input_data=sys.argv[1]

    values_df=  pd.read_csv(args.target ,sep=";")#../../Tims_Tof_data/Tims_Dummy_test/test_values.csv", sep=";")
    msms_E01= pd.read_csv(args.metadata) #pd.read_csv("../../Tims_Tof_data/Tims_Dummy_test/E01.msmsonly.csv")
    

# All data

    data_dict={"E01 msms": msms_E01}
    target_data="E01 msms"
    tolerance=0.01

    print("Targeting", target_data)

    #Sort out target
    target_list=values_df[values_df["Position"]==target_data.split()[0]]

    #Get list of found values in MetaboScape
    data_in=data_dict[target_data]
    msms_list=data_in["PEPMASS"]

    match,not_found,data_out=count_matches(data_in,msms_list,target_list,tolerance)
    matched_msms=data_out.dropna(subset="Target").sort_values("Target")
    matched_msms=matched_msms.dropna(axis=1, how="all")


    
    test=read_rawfile(args.spectrum)  #"../../Tims_Tof_data/Tims_Dummy_test/E01.gnps.mgf")
    test["FEATURE_ID"]=test["FEATURE_ID"].astype(int)
    
    df=test.merge(matched_msms, right_on="FEATURE_ID", left_on="FEATURE_ID", how="right")
    df.columns=['SCANS', 'FEATURE_ID', 'PrecursorMZ', 'MSLEVEL', 'CHARGE', 'POLARITY',
       'RTINMINUTES', 'Precursor_type', 'peaks', 'SHARED_NAME', 'RT', 'PrecursorM', 'CCS','ADDUCT',
       'MaxIntensity', '5_P2-E-1_1_9', 'Target', 'SMILES','InChIKey',"Name"]
    
    #df = read_rawfile(input_data)
    
    
    df=filter_type(df)
    df=count_peaks(df)
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
    df=df.drop(columns=['SCANS', 'FEATURE_ID','SHARED_NAME','RETENTIONTIME','RTINMINUTES'])
    print(len(df))

    output_name=args.output#../dummy_E01_test.csv"
    print("Saving as", output_name)
    df.to_csv(output_name)














    