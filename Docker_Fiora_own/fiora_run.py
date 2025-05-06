
import streamlit as st
import pandas as pd
import subprocess
import os, sys
import time
from io import StringIO
import shutil
import zipfile
import numpy as np
import matplotlib.pyplot as plt
import fiora.IO.mgfReader as mgfReader

import csv
#from comparison_SPC_files import *

try:
    import zlib
    zip_compression = zipfile.ZIP_DEFLATED
except:
    zip_compression = zipfile.ZIP_STORED


__version__ = '0.0.1'
version_date = '2025-01-14'


def datetime_stamp():
    stamp = '%4d-%02d-%02d_%02d.%02d.%02d' % time.localtime()[:6]
    return stamp

def remove_res():

    current_path=os.getcwd()
    if "pred_results.csv" in os.listdir(current_path):
        os.remove("pred_results.csv")
        st.write("Removed old result")
    elif "pred_results.csv" in os.listdir("param_file_dir"):
        os.remove("param_file_dir/pred_results.csv")
        st.write("Removed old result")
    if "eval_results.csv" in os.listdir(current_path):
        os.remove("eval_results.csv")
        st.write("Removed old result")
    elif "eval_results.csv" in os.listdir("param_file_dir"):
        os.remove("param_file_dir/eval_results.csv")
        st.write("Removed old result")

def make_dir_if_absent(d):
    if not os.path.exists(d):
        os.makedirs(d)

def ShowErrorMsg(hdr, msg):
    st.text('ShowErrorMsg '+ hdr+ '\n'+msg)

def get_output_filename(scriptBaseName, extension='.mat', outputFilesLocationDir=''):
    output_filename = os.path.splitext(scriptBaseName)[0]  + '_output_' +datetime_stamp()  + extension
    return os.path.join(outputFilesLocationDir, output_filename)


def WriteInputFileNamesToFile(ListOfFiles, OutputFileName, inputFilesLocationDir='' ):
    try:
        file01 = open(OutputFileName, mode='w')
        try:
            for s in ListOfFiles:
                file01.write(os.path.join(inputFilesLocationDir, s) +"\n")
            st.write("Saved", ListOfFiles, "to",OutputFileName)
        finally:
            file01.close()

    except IOError as e:
       ShowErrorMsg('Error', 'Error saving file\n' + str(e))


def WriteInputFiles(datafiles, outputFileDir=''):
    trace = False
    for data_file in data_files:
        outputFilePath = os.path.join(outputFileDir, data_file.name)


        try:
            file03 = open(outputFilePath, mode='wb')
            try:
                file_content = data_file.getbuffer()
                file03.write(file_content)
            finally:
                file03.close()
        except IOError as e:
            ShowErrorMsg('Error', 'Error writing input files\n' + str(e))


def WriteParamFile(param_file, outputFileDir=''):
    outputFilePath = os.path.join(outputFileDir, param_file.name)

    try:
        file04 = open(outputFilePath, mode='wb')
        try:
            file_content = param_file.getbuffer()
            file04.write(file_content)
        finally:
            file04.close()
    except IOError as e:
        ShowErrorMsg('Error', 'Error writing input files\n' + str(e))


def writePyScriptToFile(scriptFname, script_content, outputFileDir='' ):
    outputFileName = os.path.join(outputFileDir, scriptFname)
    try:
        file02 = open(os.path.join(outputFileDir, scriptFname), mode='w')
        try:
            file02.write(script_content)
        finally:
            file02.close()
    except IOError as e:
        ShowErrorMsg('Error', 'Error writing py file ', outputFileName+'\n' + str(e))


def get_scripts(verbose=0):
    script_codes = st.file_uploader("Upload scripts", type=["py"], accept_multiple_files=True)

    script_lst = ['Script not yet activated']
    if script_codes:
        for py_script in script_codes:
            py_script_details = {"FileName":py_script.name,"FileType":py_script.type,"FileSize":py_script.size}
            if py_script_details["FileType"] == "text/x-python":
                script_lst.append(py_script_details["FileName"])
                if verbose > 1:
                    st.write(py_script_details)
                stringio = StringIO(py_script.getvalue().decode('utf-8'))
                script_txt = stringio.read()
                # st.write(read_scr)
                writePyScriptToFile(py_script_details["FileName"], script_txt)
    return script_lst


def get_parameters(form):
    trace = False
    form.subheader('Input data set:')
    single_parameter_file = form.file_uploader("Upload a data set ",
                                               type=["csv"],
                                               accept_multiple_files=False)
    if single_parameter_file:
        param_file = single_parameter_file
        file_details = {"FileName":param_file.name,"FileType":param_file.type,"FileSize":param_file.size}
        parameters = file_details["FileName"]

    else:
        param_file = None
        parameters = None

    if trace:
        print('ID (number)', type(parameters), type(param_file))


    return parameters, param_file

def get_input_files(form, accepted_file_types=["dir", "pt","json"], verbose=0):
    st.write(os.getcwd())
    trace = False
    form.header('Input model:')
    data_files = form.file_uploader("Upload a model", type=accepted_file_types, accept_multiple_files=True)

    if trace:
        form.text('data_files')
        form.text(data_files)

    listOfInputFileNames = []
    if data_files:
        for f_entry in data_files:
            file_details = {"FileName":f_entry.name,"FileType":f_entry.type,"FileSize":f_entry.size}
            listOfInputFileNames.append(file_details["FileName"])
            if verbose > 0:
                form.write(file_details)
                if verbose > 1:
                    if file_details["FileType"] == "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet":
                        df = pd.read_excel(f_entry)
                        form.write(df)
                    elif file_details["FileType"] == "text/csv":
                        form.write(f_entry)
                    elif file_details["FileType"] == "text/plain":
                        stringio=StringIO(f_entry.getvalue().decode('utf-8'))
                        read_data=stringio.read()
                        form.write(read_data)
    return listOfInputFileNames, data_files




def save_ip_df(stack):
    #wn_new=np.arange(400, 4002, 2)
    #wn=[str(i) for i in wn_new]

    header = ['smiles','phase', "ID"] + wn
    with open('./processed_df.csv', 'w', newline='\n') as csvfile:
            csvwriter = csv.writer(csvfile, delimiter=',')
            csvwriter.writerow(header)
            for row in stack:
                csvwriter.writerow(row)



##Plotting scripts

def gen_histogram(d_set, metric):
    n, bins, patches = plt.hist(x=d_set, bins=np.arange(0, 1.1, 0.05), color='darkmagenta',
                            alpha=0.7, rwidth=0.85)
    plt.grid(axis='y', alpha=0.75)
    if metric=="sis":
        plt.xlabel('Spectral Information Similarity (SIS)')
    else:
        plt.xlabel('Cosine Similarity (Cos_s)')
    plt.ylabel('Frequency')
    m = np.mean(d_set)
    std = np.std(d_set)
    title = r"$\mu$ = " + str(np.round(m, 4)) + r"   $\sigma$ = " + str(np.round(std, 4) )
    plt.title(title)
    st.pyplot(plt.gcf(), clear_figure=True)


def read_rawfile(input_df):
    print("Reading ", input_df)
    df_msp = mgfReader.read(input_df)
    df = pd.DataFrame(df_msp)
    return df



def plot_several_eval(data,i, metric):

    data, true, pred, IDs=prep_data(data, metric)

    IDs=list(IDs)
    pred_iso=get_isomers(pred,IDs[i])
    true_iso=get_isomers(true,IDs[i])


    x=np.arange(400,4002,2)
    fig, ax = plt.subplots(2,2, figsize=(10,10))
    y, y_true, SIS=get_all(pred_iso, true_iso, 0, metric)


    name=str(pred_iso.iloc[0,-1]) +". " +metric+": "+ str(round(SIS, 4))
    ax[0,0].plot(x, y/max(y), label="Pred")
    ax[0,0].plot(x, y_true/max(y_true), label="True")

    ax[0,0].set_title(name)
    ax[0,0].legend()
    if len(pred_iso)>1:
        y, y_true, SIS=get_all(pred_iso, true_iso, 1, metric)
        name=str(pred_iso.iloc[1,-1])+". " +metric+": "+ str(round(SIS, 4))
        ax[1,0].plot(x, y/max(y), label="Pred")
        ax[1,0].plot(x, y_true/max(y_true), label="True")
        ax[1,0].set_title(name)
        ax[1,0].legend()
    if len(pred_iso)>2:
        y, y_true, SIS=get_all(pred_iso, true_iso, 2, metric)
        name=str(pred_iso.iloc[2,-1]) +". " +metric+": "+ str(round(SIS, 4))
        ax[0,1].plot(x, y/max(y), label="Pred")
        ax[0,1].plot(x, y_true/max(y_true), label="True")
        ax[0,1].set_title(name)
        ax[0,1].legend()
    if len(pred_iso)>3:
        y, y_true, SIS=get_all(pred_iso, true_iso, 3, metric)
        name=str(pred_iso.iloc[3,-1]) +". " + metric+": "+ str(round(SIS, 4))
        ax[1,1].plot(x, y/max(y), label="Pred")
        ax[1,1].plot(x, y_true/max(y_true), label="True")
        ax[1,1].set_title(name)
        ax[1,1].legend()
    fig.suptitle(str(pred_iso.iloc[0,-1]))
    st.pyplot(plt.gcf(), clear_figure=True)



if __name__ == '__main__':

    st.title('Fiora ')
    st.text('Version '+ __version__)

    file_cache_dir = 'file_cache_dir'
    param_file_dir = 'param_file_dir'
    make_dir_if_absent(file_cache_dir)
    make_dir_if_absent(param_file_dir)

    trace = False
    verbose = 1

    script_dict={"Predict":"fiora-predict"}
    script_lst = ['Script not yet activated']

    script_lst.append("Predict")
    #script_lst.append("Evaluate")
    st.sidebar.header('Active script:')

    active_script = st.selectbox('Select script', script_lst)

    if active_script!='Script not yet activated':

        active_script=script_dict.get(active_script)
        st.sidebar.write(active_script)
        ScriptBaseName_NoExt = os.path.splitext(active_script)[0]

    else: st.sidebar.write(active_script)

    form = st.form('Run script')

    #listOfInputFileNames, data_files = get_input_files(form)



    parameters, parameter_file = get_parameters(form)

    form.divider()


    form.subheader('Begin run')
    form.write(active_script)


    is_submitted = form.form_submit_button('Run script')

    if is_submitted:

        shutil.rmtree(file_cache_dir)
        make_dir_if_absent(file_cache_dir)
        outputDataDirectory = os.getcwd()

        # Have param file in separate dir to avoid saving it with the return results
        shutil.rmtree(param_file_dir)
        make_dir_if_absent(param_file_dir)
        outputParamDirectory = param_file_dir

        if (active_script != 'Script not yet activated'):
            remove_res()

            script_string =[active_script,"--model","default"]
            script_string =[active_script,"--model","checkpoint_all_wells_mona_ft_rounded.best.pt"] #todo: update with best model
            #script_string.append("--annotation")


            if parameters:
             #
                WriteParamFile(parameter_file, outputFileDir = os.getcwd())


            output_file_name="pred_results.mgf"
            script_string.append("-i")
            script_string.append(parameters) #input file
            script_string.append("-o")
            script_string.append(output_file_name)


            script_entries = ''
            for s in script_string:
                script_entries = script_entries + s+'\n'

            subprocess.run(script_string)
            outputDataDirectory = file_cache_dir

        else:

            form.text('Insufficient data to process')

        current_path=os.getcwd()
       # st.write(os.listdir(current_path),os.listdir("param_file_dir"), os.listdir('file_cache_dir'))
        #combine to find pred_result
        what="None"
        if "pred" in str(active_script):

            what="Pred"
            if "pred_results.mgf" in os.listdir(current_path):
                res=read_rawfile("pred_results.mgf")
            elif "pred_results.mgf" in os.listdir("param_file_dir"):
                res=read_rawfile("param_file_dir/pred_results.mgf")
        elif "eval" in str(active_script):
            what="Eval"
            if "eval_results.csv" in os.listdir(current_path):
                res=pd.read_csv("eval_results.csv")
            elif "eval_results.csv" in os.listdir("param_file_dir"):
                res=pd.read_csv("param_file_dir/eval_results.csv")



        if what=="Pred":
            st.write(res)
            st.write("Plotting the 1th to 10th spectrum")
            for s in range(len(res)):
                if s<11:
                    try:

                        res_dict=dict(res.iloc[s,-1])

                        mz=list(res_dict["mz"])
                        int_list=np.array(res_dict["intensity"])
                        fig, ax = plt.subplots()

                        ax.stem(mz, int_list/max(int_list), markerfmt=" ")
                        st.pyplot(fig)


                    except:
                        st.write("Error plotting")
                        res_dict=dict(res.iloc[0,-1])

                        mz=res_dict["mz"]
                        int_list=res_dict["intensity"]



        elif what=="Eval":

            st.write(res.iloc[:,-4:])



        output_csv = res.to_csv(index=False).encode('utf-8')
        csv_name="results_" + what + '_' +datetime_stamp() +".csv"
        st.write("Download results as CSV:")
        st.download_button('Download CSV', output_csv, file_name=csv_name, mime='text/csv')
