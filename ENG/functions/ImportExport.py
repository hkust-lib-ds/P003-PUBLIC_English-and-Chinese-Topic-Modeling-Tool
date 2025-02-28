import streamlit as st
import pandas as pd
import json
import zipfile
from bertopic import BERTopic
from functions.Sidebar_NewRun import ModelToRun, ModelToRun_nodata
from functions.Sidebar_UpdateModel import LabelTopics
from functions.Main_overTime import TimestampText
from io import BytesIO
from datetime import datetime
import re
import base64

def processDoc(uploaded_files):
    # seperate txt and csv files
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
            DOCs[file.name] = {}
            DOCs[file.name]['content'] = file.getvalue().decode("utf-8")
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

    DOCs = dict(sorted(DOCs.items()))
    
    st.success("Finished Processing.")
    return DOCs




######################################################################################################
dict_timeFormat = {
    "2024":         "%Y",
    "9":            "%m",
    "09":           "%m",
    "Sep":         "%b",
    "September":    "%B",
    "6":            "%d",
    "06":           "%d",
}

def parseTimeFormat(timeFormat):
    Y_M_D = (False, False, False)
    # replace year
    if '2024' in timeFormat:
        timeFormat = timeFormat.replace('2024', dict_timeFormat['2024'])
        Y_M_D = (True, Y_M_D[1], Y_M_D[2])
    
    # replace month
    for month in ['09', '9', 'September', 'Sep']:
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
                                 

# format = 'Year_Month_Day' or 'Timestamp' or 'YMD&Timestamp' or 'Customized'
def processDoc_time(uploaded_files, format = 'Year_Month_Day', timeFormat=None): 
    
    # parse time format
    Y_M_D = (False, False, False)
    if format == 'Customized':
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
            if format == 'Year_Month_Day':
                DOCs[df.iloc[i, 0]]['time'] = (int(df.iloc[i, 2]), int(df.iloc[i, 3]), int(df.iloc[i, 4]))
                DOCs[df.iloc[i, 0]]['timestamp'] = 0
            elif format == 'Timestamp':
                DOCs[df.iloc[i, 0]]['timestamp'] = int(df.iloc[i, 2])
            elif format == 'Customized':
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
    
    st.success("Finished Processing.")
    return DOCs
######################################################################################################

def ExportModelData(RUN, TRAIN_DOCs, CUR_RUN):
    # doc_info json
    doc_info = {}
    for doc in TRAIN_DOCs:
        doc_info[doc] = {}
        doc_info[doc]['content'] = TRAIN_DOCs[doc]['content']
        doc_info[doc]['topic_prob_isrepresentative'] = None

    # Topic info json
    topic_info = {}
    for topic in RUN['TOPICs']:
        topic_info[topic] = {}
        topic_info[topic]['label'] = RUN['TOPICs'][topic]['LABEL']
        topic_info[topic]['word_score'] = RUN['TOPICs'][topic]['WORDs']
        topic_info[topic]['RepresentativeDoc_prob'] = RUN['TOPICs'][topic]['RepDocs']
        topic_info[topic]['BelongDoc_prob'] = RUN['TOPICs'][topic]['BelDocs']

        for d in RUN['TOPICs'][topic]['RepDocs']:
            doc_info[d]['topic_prob_isrepresentative'] = (topic, RUN['TOPICs'][topic]['RepDocs'][d], True)
        for d in RUN['TOPICs'][topic]['BelDocs']:
            doc_info[d]['topic_prob_isrepresentative'] = (topic, RUN['TOPICs'][topic]['BelDocs'][d], False)

    # doc_topic_probMatrix csv
    ApproDistribution = RUN['ApproDistribution']
    doc_topic_probMatrix = {}
    for index, docID in enumerate(list(TRAIN_DOCs.keys())):
        doc_topic_probMatrix[docID] = ApproDistribution[index]
    columns = list(RUN['TOPICs'].keys())
    columns = [i for i in columns if i != "Topic -1"]
    doc_topic_probMatrix = pd.DataFrame.from_dict(doc_topic_probMatrix, orient='index', columns=columns)
    doc_topic_probMatrix.index.name = 'doc_id'
    doc_topic_probMatrix.reset_index(inplace=True)


    doc_topic_probMatrix = doc_topic_probMatrix.to_csv(index=False)

    # csv alternative for json
    doc_info_csv = pd.DataFrame.from_dict(doc_info, orient='index')
    doc_info_csv.index.name = 'doc_id'
    doc_info_csv.reset_index(inplace=True)
    doc_info_csv = doc_info_csv.to_csv(index=False)

    topic_info_csv = pd.DataFrame.from_dict(topic_info, orient='index')
    topic_info_csv.index.name = 'topic_id'
    topic_info_csv.reset_index(inplace=True)
    topic_info_csv = topic_info_csv.to_csv(index=False)

    # Create a zip file in memory
    zip_buffer = BytesIO()
    with zipfile.ZipFile(zip_buffer, 'w') as zipf:
        zipf.writestr(f"{CUR_RUN}_doc_topic_probMatrix.csv", doc_topic_probMatrix)
        zipf.writestr(f"{CUR_RUN}_doc_info.csv", doc_info_csv)
        zipf.writestr(f"{CUR_RUN}_topic_info.csv", topic_info_csv)
        zipf.writestr(f"{CUR_RUN}_doc_info.json", json.dumps(doc_info))
        zipf.writestr(f"{CUR_RUN}_topic_info.json", json.dumps(topic_info))

    zip_buffer.seek(0)

    st.download_button(
        label=f"Download {CUR_RUN}.zip",
        data=zip_buffer,
        file_name=f"{CUR_RUN}.zip",
        mime='application/zip'
    )

    st.success("Prediction data exported successfully.")

def ExportPredictData(RUN, PREDICT_DOCs, CUR_RUN, PREDICT_RES):
    # doc_info json
    doc_info = {}
    for doc in PREDICT_DOCs:
        doc_info[doc] = {}
        doc_info[doc]['content'] = PREDICT_DOCs[doc]['content']
        doc_info[doc]['topic_prob'] = (PREDICT_DOCs[doc]['TopicProb'][0], float(PREDICT_DOCs[doc]['TopicProb'][1]))

    # csv alternative for json
    doc_info_csv = pd.DataFrame.from_dict(doc_info, orient='index')
    doc_info_csv.index.name = 'doc_id'
    doc_info_csv.reset_index(inplace=True)
    doc_info_csv = doc_info_csv.to_csv(index=False)

    # doc_topic_probMatrix csv
    ApproDistribution = PREDICT_RES['_ApproMatrix_']
    doc_topic_probMatrix = {}
    for index, docID in enumerate(list(PREDICT_DOCs.keys())):
        doc_topic_probMatrix[docID] = ApproDistribution[index]
    columns = ['doc_id'] + [i for i in list(RUN['TOPICs'].keys()) if i != "Topic -1"]
    doc_topic_probMatrix = pd.DataFrame.from_dict(doc_topic_probMatrix, orient='index', columns=columns[1:])
    doc_topic_probMatrix.index.name = 'doc_id'
    doc_topic_probMatrix.reset_index(inplace=True)
    doc_topic_probMatrix = doc_topic_probMatrix.to_csv(index=False)

    # Create a zip file in memory
    zip_buffer = BytesIO()
    with zipfile.ZipFile(zip_buffer, 'w') as zipf:
        zipf.writestr(f"{CUR_RUN}_pred_doc_info.json", json.dumps(doc_info))
        zipf.writestr(f"{CUR_RUN}_pred_doc_info.csv", doc_info_csv)
        zipf.writestr(f"{CUR_RUN}_pred_doc_topic_probMatrix.csv", doc_topic_probMatrix)

    zip_buffer.seek(0)

    st.download_button(
        label=f"Download {CUR_RUN}_prediction.zip",
        data=zip_buffer,
        file_name=f"{CUR_RUN}_prediction.zip",
        mime='application/zip'
    )

    st.success("Prediction data exported successfully.")


def DownloadRun(RUN, CUR_RUN):
    # export model
    model = RUN['MODEL']
    model.save(f"{CUR_RUN}.pickle", serialization="pickle")
    with open(f"{CUR_RUN}.pickle", 'rb') as f:
        st.download_button(
            label=f"Download {CUR_RUN}.pickle",
            data=f,
            file_name=f"{CUR_RUN}.pickle",
            mime='application/octet-stream'
        )
    st.success("Model exported successfully.")

def UploadRun(uploaded_pickle, TRAIN_DOCs):
    path = uploaded_pickle.name
    with open(path, 'wb') as f:
        f.write(uploaded_pickle.getbuffer()) 
    try:
        model = BERTopic.load(path)
        if TRAIN_DOCs:
            original_labels = {topic: label for topic, label in zip(sorted(set(model.topics_)), model.custom_labels_)}
            original_labels = {value.split(" | ")[0]: value.split(" | ")[1] if len(value.split(" | ")) == 2 else None for key, value in original_labels.items()}
            original_topic_label_text = ''
            for key, value in original_labels.items():
                if value:
                    original_topic_label_text += f"{key}: {value},"
            if original_topic_label_text.endswith(','):
                original_topic_label_text = original_topic_label_text[:-1]
            run = ModelToRun(model, TRAIN_DOCs)
            run = LabelTopics(run, TRAIN_DOCs, original_topic_label_text)
        else:
            run, TRAIN_DOCs = ModelToRun_nodata(model)
        st.success("Model loaded successfully.")
    except Exception as e:
        run = None
        st.error(f"Error: {e}.")
    return run, TRAIN_DOCs


def SampleTrainDoc():
    TrainDoc = 'Data/Doc_withoutTime.csv'
    csv = pd.read_csv(TrainDoc)
    csv = csv.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()  # Encoding the CSV data
    href = f'<a href="data:file/csv;base64,{b64}" download="SampleTrainDoc.csv">Download a CSV sample for training Doc here</a>'
    return href

def SamplePredictDoc():
    PredictDoc = 'Data/PredictDoc.csv'
    csv = pd.read_csv(PredictDoc)
    csv = csv.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()  # Encoding the CSV data
    href = f'<a href="data:file/csv;base64,{b64}" download="SamplePredictDoc.csv">Download a CSV sample for prediction Doc here</a>'
    return href

def SampleTimeDoc_YMD():
    TimeDoc = 'Data/Doc_Time_YMD.csv'
    csv = pd.read_csv(TimeDoc)
    csv = csv.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()  # Encoding the CSV data
    href = f'<a href="data:file/csv;base64,{b64}" download="SampleTimeDoc_YMD.csv">Download a CSV sample for time (YMD) Doc here</a>'
    return href

def SampleTimeDoc_Timestamp():
    TimeDoc = 'Data/Doc_Time_timestamp.csv'
    csv = pd.read_csv(TimeDoc)
    csv = csv.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()  # Encoding the CSV data
    href = f'<a href="data:file/csv;base64,{b64}" download="SampleTimeDoc_Timestamp.csv">Download a CSV sample for time (timestamp) Doc here</a>'
    return href

def SampleTimeDoc_Customized():
    TimeDoc = 'Data/Doc_Time_customized.csv'
    csv = pd.read_csv(TimeDoc)
    csv = csv.iloc[:, :3]
    csv = csv.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()  # Encoding the CSV data
    href = f'<a href="data:file/csv;base64,{b64}" download="SampleTimeDoc_Customized.csv">Download a CSV sample for time (customized) Doc here</a>'
    return href
