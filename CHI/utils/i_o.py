import pandas as pd
from langchain_text_splitters import RecursiveCharacterTextSplitter
import streamlit as st
import numpy as np
import re
from utils.stopwords import *
from datetime import datetime
from pathlib import Path
import torch
import json
import io
import zipfile
import os
from typing import Union
import tempfile

dict_timeFormat = {
    "2024":         "%Y",
    "9":            "%m",
    "09":           "%m",
    "Sep":         "%b",
    "九月":         "%B",
    "6":            "%d",
    "06":           "%d",
}


def load_stopwords(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        stopwords = set(f.read().splitlines())  # Read lines and create a set
    return ExpandStopwordList(list(stopwords))


def load_files(uploaded_files):
    txt_files = []
    csv_files = []
    for file in uploaded_files:
        if file.name.endswith('.txt'):
            txt_files.append(file)
        elif file.name.endswith('.csv'):
            csv_files.append(file)
            
    # prepare ALL_DOCs
    DOCs = {}
    for file in txt_files:
        try:
            chi_punc = "！？｡。＂＃＄％＆＇（）＊＋，－／：；＜＝＞＠［＼］＾＿｀｛｜｝～｟｠｢｣､、〃《》「」『』【】〔〕〖〗〘〙〚〛〜〝〞〟〰〾〿–—‘’‛“”„‟…‧﹏."
            DOCs[file.name] = {}
            DOCs[file.name]['content'] = re.sub(r"[%s]+" %chi_punc, " ", file.getvalue().decode("utf-8"))
        except:
            st.error("Error: " + file.name)
            
    for file in csv_files:
        try: 
            df = pd.read_csv(file)
        except:
            st.error("Error: " + file.name)
        for i in range(len(df)):
            DOCs[df.iloc[i, 0]] = {}
            DOCs[df.iloc[i, 0]]['content'] = df.iloc[i, 1]

    #DOCs = dict(sorted(DOCs.items()))
    st.success("文件上載完成，請稍等片刻，待右上角`Running...`完成，然後到側邊欄-參數設定完成模型訓練或上載")
    return DOCs


def process_files(docs):
    new_docs = {}
    for doc_id in docs.keys():
        new_docs[doc_id] = {}
        chi_punc = "！？｡。＂＃＄％＆＇‘’“” '（）＊＋，－／：；＜＝＞＠［＼］＾＿｀｛｜｝～｟｠｢｣､、〃《》「」『』【】〔〕〖〗〘〙〚〛〜〝〞〟〰〾〿–—‘’‛“”„‟…‧﹏."
        new_docs[doc_id]['content'] = re.sub(r"[%s]+" %chi_punc, " ", docs[doc_id]['content'])
    return new_docs


def parseTimeFormat(timeFormat):
    Y_M_D = (False, False, False)
    # replace year
    if '2024' in timeFormat:
        timeFormat = timeFormat.replace('2024', dict_timeFormat['2024'])
        Y_M_D = (True, Y_M_D[1], Y_M_D[2])
    
    # replace month
    for month in ['09', '9', '九月', 'Sep']:
        if month in timeFormat:
            timeFormat = timeFormat.replace(month, dict_timeFormat[month])
            Y_M_D = (Y_M_D[0], True, Y_M_D[2])
            break

    # replace day
    for day in ['06', '6']:
        if day in timeFormat:
            timeFormat = timeFormat.replace(day, dict_timeFormat[day])
            Y_M_D = (Y_M_D[0], Y_M_D[1], True)
            break

    if Y_M_D == (False, False, False):
        timeFormat = None
    # print(timeFormat)
    # print(Y_M_D)
    return timeFormat, Y_M_D


def Str2Time(timeStr, timeFormat, Y_M_D):  
    date_obj = datetime.strptime(timeStr, timeFormat) 
    
    if Y_M_D == (True, True, True):
        dateInfo = (date_obj.year, date_obj.month, date_obj.day)
        return dateInfo

    if Y_M_D == (True, True, False):
        dateInfo = (date_obj.year, date_obj.month, '00')
        return dateInfo

    if Y_M_D == (True, False, True):
        dateInfo = (date_obj.year, '00', date_obj.day)
        return dateInfo

    if Y_M_D == (False, True, True):
        dateInfo = ('0000', date_obj.month, date_obj.day)
        return dateInfo

    if Y_M_D == (True, False, False):
        dateInfo = (date_obj.year, date_obj.month, '00')
        return dateInfo

    if Y_M_D == (False, True, False):
        dateInfo = ('0000', date_obj.month, '00')
        return dateInfo
     
    return None


def load_files_time(uploaded_files, format, timeFormat = None):
    # parse time format
    Y_M_D = (False, False, False)
    if format == '自定義':
        if timeFormat is None:
            st.error("Error: Please input a customized time format.")
            return None
        else:
            timeFormat = re.sub(r'(\d+)(st|nd|rd|th)', r'\1', timeFormat) # <d>rd, <d>st, <d>st` -> `<d>th`
            timeFormat, Y_M_D = parseTimeFormat(timeFormat)  
            if timeFormat is None:
                st.error("Error: Invalid time format.")  
                return None
    
    # seperate txt and csv files
    csv_files = []
    for file in uploaded_files:
        if file.name.endswith('.txt'):
            st.error("Error: " + file.name)
            st.error("Only csv files are accepted.")
            return None
        elif file.name.endswith('.csv'):
            csv_files.append(file)

    # prepare ALL_DOCs
    DOCs = {}
    for file in csv_files:
        try: 
            df = pd.read_csv(file)
        except:
            st.error("Error: " + file.name)
        for i in range(len(df)):
            if str(df.iloc[i, 0]) == 'nan' or str(df.iloc[i, 1]) == 'nan' or (format == 'Year_Month_Day' and (str(df.iloc[i, 2]) == 'nan' or str(df.iloc[i, 3]) == 'nan' or str(df.iloc[i, 4]) == 'nan')) or (format == 'Timestamp' and str(df.iloc[i, 2]) == 'nan') or (format == 'Customized' and str(df.iloc[i, 2]) == 'nan'):
                continue
            DOCs[df.iloc[i, 0]] = {}
            DOCs[df.iloc[i, 0]]['content'] = df.iloc[i, 1]
            if format == '年月日':
                DOCs[df.iloc[i, 0]]['time'] = (int(df.iloc[i, 2]), int(df.iloc[i, 3]), int(df.iloc[i, 4]))
                DOCs[df.iloc[i, 0]]['timestamp'] = 0
            elif format == '時間戳':
                DOCs[df.iloc[i, 0]]['timestamp'] = int(df.iloc[i, 2])
            elif format == '自定義':
                timeStr = str(df.iloc[i, 2])
                # replace <d>rd, <d>st, <d>st with <d>th
                # print(timeStr)
                timeStr = re.sub(r'(\d+)(st|nd|rd|th)', r'\1', timeStr)
                # print(timeStr)
                try: 
                    dateInfo = Str2Time(timeStr, timeFormat, Y_M_D)
                except:
                    st.error("Error: " + file.name)
                    return None
                # print(dateInfo)
                if dateInfo == None:
                    return None
                DOCs[df.iloc[i, 0]]['time'] = dateInfo
                DOCs[df.iloc[i, 0]]['timestamp'] = 0
                
    DOCs = dict(sorted(DOCs.items()))
    st.success("文件上載完成，請稍等片刻，待右上角`Running...`完成，然後到側邊欄-參數設定完成模型訓練或上載")
    return DOCs


def process_files_time(docs):
    new_docs = {}
    for doc_id in docs.keys():
        chi_punc = "！？｡。＂＃＄％＆＇‘’“” '（）＊＋，－／：；＜＝＞＠［＼］＾＿｀｛｜｝～｟｠｢｣､、〃《》「」『』【】〔〕〖〗〘〙〚〛〜〝〞〟〰〾〿–—‘’‛“”„‟…‧﹏."
        new_docs[doc_id] = {}
        new_docs[doc_id]['content'] =  re.sub(r"[%s]+" %chi_punc, " ", docs[doc_id]['content'])
        new_docs[doc_id]['time'] = docs[doc_id]['time']
        new_docs[doc_id]['timestamp'] = docs[doc_id]['timestamp']
    return new_docs 


def unzip_model(zip_file):
    if zip_file:
        temp_dir = tempfile.mkdtemp()
        #path = os.path.join(temp_dir, zip_file.name)
        with zipfile.ZipFile(zip_file, 'r') as zip_ref:
            zip_ref.extractall(temp_dir)
        #st.write(os.listdir(temp_dir))
        return temp_dir
    else:
        return None


def zip_model(folder_path):
    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, 'w') as zip_file:
        for root, dirs, files in os.walk(folder_path):
            for file in files:
                # Create the complete filepath of the file in the directory
                file_path = os.path.join(root, file)
                # Add the file to the zip file
                zip_file.write(file_path, os.path.relpath(file_path, folder_path))
    return zip_buffer
