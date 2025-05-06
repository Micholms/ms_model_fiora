#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 31 22:51:30 2024

@author: mats_j
"""
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import streamlit as st
import pandas as pd
import subprocess
import os, sys
import time
from io import StringIO
import shutil

import zipfile
try:
    import zlib
    zip_compression = zipfile.ZIP_DEFLATED
except:
    zip_compression = zipfile.ZIP_STORED
    

__version__ = '0.0.1'
version_date = '2024-11-09'


def datetime_stamp():
    stamp = '%4d-%02d-%02d_%02d.%02d.%02d' % time.localtime()[:6]
    return stamp


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
        finally:
            file01.close()                
    except IOError as e:
       ShowErrorMsg('Error', 'Error saving file\n' + str(e))
       

def WriteInputFiles(datafiles, outputFileDir=''):
    trace = False
    for data_file in data_files:
        outputFilePath = os.path.join(outputFileDir, data_file.name)
        if trace:
            print('WriteInputFiles', outputFilePath)
            
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
    print('WriteParamFile', outputFilePath)
    #full_local_path = os.path.abspath(outputFileDir)
    #st.text('full_local_path')
    #st.write(full_local_path)
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
    form.subheader('Optional parameters:')        
    single_parameter_file = form.file_uploader("Upload a parameter file", 
                                               type=["csv", "txt", "xlsx", "pth"], 
                                               accept_multiple_files=False)       
    if single_parameter_file:
        param_file = single_parameter_file
        file_details = {"FileName":param_file.name,"FileType":param_file.type,"FileSize":param_file.size}
        parameters = file_details["FileName"]
    else:
        param_file = None
        parameters = form.text_input('Alt. enter parameters:') 
        form.text(parameters)
    if trace:
        print('get_parameters', type(parameters), type(param_file))
    return parameters, param_file



def get_input_files(form, accepted_file_types=['csv', 'txt', 'xlsx', 'mat', 'tif', 'png'], verbose=0):
    trace = False
    form.header('Input files:')    
    data_files = form.file_uploader("Upload a Dataset", type=accepted_file_types, accept_multiple_files=True)
    
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


 def zip_results(dir_name, output_name):
     final_zipped_name = output_name+'.zip'
     temp_zipped_name = output_name+'.zip.part'
     if not os.path.exists(final_zipped_name):
         zipf = zipfile.ZipFile(temp_zipped_name, mode='w', allowZip64=True)
         arcname = os.path.basename(output_name)
        
         zipf.write(dir_name, arcname, compress_type=zip_compression)
         zipf.close()
         os.rename(temp_zipped_name, final_zipped_name)
     return final_zipped_name


def zip_results(dir_name, output_name):
    final_zipped_name = shutil.make_archive(output_name, 'zip', dir_name)    
    return os.path.basename(final_zipped_name)

        
if __name__ == '__main__':
    
    st.title('Files2Script_webb')
    st.text('Version '+ __version__)

    file_cache_dir = 'file_cache_dir'
    param_file_dir = 'param_file_dir'
    make_dir_if_absent(file_cache_dir)
    make_dir_if_absent(param_file_dir)
    # outputDataDirectory = file_cache_dir



    trace = False
    verbose = 1
    zip_fname = ''
 

    script_lst = get_scripts()
    
    st.sidebar.header('Active script:')
    active_script = st.selectbox('Select script', script_lst)
    st.sidebar.write(active_script)
    ScriptBaseName_NoExt = os.path.splitext(active_script)[0]

    form = st.form('Run script')
    #is_submitted = False
    listOfInputFileNames, data_files = get_input_files(form)
                     
    form.divider()
    parameters, parameter_file = get_parameters(form)     
    form.divider()

    form.subheader('Begin run' )
    form.write(active_script)
    #st.write(is_submitted)        
    is_submitted = form.form_submit_button('Run script')
    form.write(is_submitted) 
    if is_submitted:
        shutil.rmtree(file_cache_dir)
        make_dir_if_absent(file_cache_dir)
        outputDataDirectory = file_cache_dir
        
        # Have param file in separate dir to avoid saving it with the return results
        shutil.rmtree(param_file_dir)
        make_dir_if_absent(param_file_dir)
        outputParamDirectory = param_file_dir
        
        if listOfInputFileNames and (active_script != 'Script not yet activated'):
            full_local_path = os.path.abspath(outputDataDirectory)
            if verbose > 0:
                st.text('full_local_path')
                st.write(full_local_path)
            ScriptInputFileName = os.path.join( outputDataDirectory, ScriptBaseName_NoExt + '_input_' +datetime_stamp()+'.txt')
            dummy_outpt = 'dummy_output_'+datetime_stamp()+'.txt'
            WriteInputFileNamesToFile( listOfInputFileNames, ScriptInputFileName, inputFilesLocationDir=full_local_path)
            WriteInputFiles(data_files, outputFileDir = outputDataDirectory)
            # The function of the output file depends on the called script
            # It may be used as a pointer to a directory for many output files, not using the base filename provided
            output_filename = get_output_filename(active_script, outputFilesLocationDir=full_local_path)
            script_string = [f"{sys.executable}", active_script, ScriptInputFileName, output_filename]
            if parameters:
                if isinstance(parameters, list):
                    script_string.extend(parameters)
                else: # is a filename
                    full_param_path = os.path.abspath(outputParamDirectory)
                    script_string.append(os.path.join(full_param_path, parameters))
                    WriteParamFile(parameter_file, outputFileDir = outputParamDirectory)

            script_entries = ''
            for s in script_string:
                if trace:
                    print('script_entries', s, type(s))
                script_entries = script_entries + s+'\n'                   
            form.text(script_entries)
            
            subprocess.run(script_string)
            zip_outpt_name = os.path.splitext(active_script)[0] + '_' +datetime_stamp()
            zip_fname = zip_results(outputDataDirectory, zip_outpt_name)
        else:
            #print('Insufficient data to process')
            form.text('Insufficient data to process')
        
        st.text('zip_fname')
        st.text(zip_fname)
        if zip_fname:
            #zip_fname = zip_results(outputDataDirectory, 'f2s_downloads')
            with open(zip_fname, "rb") as fp:
                btn = st.download_button( label="Download ZIP",
                                          data=fp,
                                          file_name=zip_fname,
                                          mime="application/zip")


        current_path=os.getcwd()

        #combine to find pred_result

        res=pd.read_csv("app/pred_results.csv")



        df_plot=res.iloc[:,:-3].T #or -2  #transposed

        df_plot["x"]=np.arange(400,4002,2)

        if len(res)<6:

            for i in range(len(res)):

                name=res.iloc[i,-1]

                st.write("Plot for: ", name)

                st.line_chart(df_plot, "x", i, x_label="Wavenumber", y_label="Predicted Intensity")
        else: st.write("Too many")






        output_csv = res.to_csv(index=False).encode('utf-8')
        st.download_button('Download CSV', output_csv, file_name="results.csv", mime='text/csv')
        
        
        
        
        
        
        
        
        
