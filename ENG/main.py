import streamlit as st
from streamlit_extras.stylable_container import stylable_container
from bertopic import BERTopic
from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd

###############################################################
# set page config
###############################################################

st.set_page_config(
    page_title="English Topic Modeling Tool | HKUST Library",
    page_icon='favicon.ico',
    layout="wide"
)

#########################################################################################################################################
#########################################################################################################################################

# Custom CSS 
st.markdown(
    """
    <style>
    tr {
        text-align: left! important;
    }
    </style>
    """,
    unsafe_allow_html=True
)

###############################################################
# functions and Utils import
###############################################################
from functions.ImportExport import processDoc, ExportModelData, ExportPredictData, UploadRun, DownloadRun, processDoc_time, SampleTrainDoc, SamplePredictDoc, SampleTimeDoc_Customized, SampleTimeDoc_YMD, SampleTimeDoc_Timestamp
from functions.Main_overTime import GenerateTimestamp, Display_Timestamp, TrainModelOverTime, ExportTopicOverTimeData,TimestampText
from functions.Main_DisplayTopicModel import DisplayTopic
from functions.Main_DisplayDoc import DisplayDoc, DisplayTrain_table, DisplayDoc_nodata
from functions.Main_prediction import PredictDoc, DisplayPredict
# from functions.Sidebar_Topic import 
from functions.Sidebar_NewRun import TrainNewModel,ModelToRun, UploadStopWords, LabelTopics_nodata
from functions.Sidebar_UpdateModel import ReduceTopic, LabelTopics, MergeTopic

from utils.ConstValue import *

###############################################################
# global variables
###############################################################
ALL_RUNs = {}                                                                           # {RunName: Run} detains in sidebar_newRun; ; Doc_Timestamp: {docID : stamp}, { TOPIC_TIME_RES = {Model_df, Figs} }
CUR_RUN = None
CUR_TOPIC = None                                                                        # None means all topics  
TRAIN_DOCs = {}                                                                         # {DocID: {'content': 'content', 'time': (year, month, day), 'timestamp': timestamp}}
PREDICT_DOCs = {}                                                                       # {DocID: {'content': 'content'}, 'TopicProb': (topic, prob)}
PREDICT_RES = {}                                                                        # {_ApproMatrix_, _ApproMatrix_fig_}
ALL_STOP_WORDS = { 'DEFAULT': CountVectorizer(stop_words="english").get_stop_words() }  # {name: [stop words]}
STOP_WORDS = ALL_STOP_WORDS['DEFAULT']


###############################################################
# session_state initialization and load data
###############################################################
session_state = st.session_state.setdefault("session_state", {})

# initialization of session_state
default_session_state = {
    "ALL_RUNs": ALL_RUNs,
    "TRAIN_DOCs": TRAIN_DOCs,
    "PREDICT_DOCs": PREDICT_DOCs,
    "PREDICT_RES": PREDICT_RES,
    "CUR_RUN": CUR_RUN,
    "CUR_TOPIC": CUR_TOPIC,
    "STOP_WORDS": STOP_WORDS,
    "ALL_STOP_WORDS": ALL_STOP_WORDS,
}

# load previous data from session_state
for key, value in default_session_state.items():
    if key in session_state:
        globals()[key] = session_state[key]
    else:
        globals()[key] = value

if "ConfirmTrainDoc" not in session_state:
    session_state["ConfirmTrainDoc"] = False

if "ModelOnly" not in session_state:
    session_state["ModelOnly"] = False

if "TopicOverTime" not in session_state:
    session_state["TopicOverTime"] = False # False or formats: Year_Month_Day, Timestamp, YMD&Timestamp

if "PredictUploader" not in session_state:
    session_state["PredictUploader"] = None

if "SearchTopic" not in session_state:
    session_state["SearchTopic"] = None

if 'PredictUsedRun' not in session_state:
    session_state['PredictUsedRun'] = None

if 'Timestamp_text' not in session_state:
    session_state['Timestamp_text'] = None


#########################################################################################################################################
#########################################################################################################################################
# main page
st.title("English Topic Modeling Tool")

st.markdown(
    """
    <div style='background-color: #f0f0f5; padding: 20px; border-radius: 10px;'>
        <p style='font-size: 18px;'>
            Welcome to our Topic Modeling Tool <img src='https://img.icons8.com/emoji/48/000000/waving-hand-emoji.png' width='30' style='vertical-align: middle; margin-right: 10px;'/> <br/>
            <span style='font-size: 22px; font-weight: bold;'>Topic modeling</span> is a technique in natural language processing and machine learning
            that is used to <span style='font-size: 20px; font-weight: bold;'>discover abstract topics within a collection of documents</span>.
            It helps in identifying the underlying themes present in a large corpus of text by analyzing the patterns of word usage.
            <br/><br/>
            Our tool offers two main functionalities:
        </p>
        <ul style='font-size: 18px;'>
            <li><span style='font-size: 22px; font-weight: bold;'>Train Model with Training Data</span></li>
            <li><span style='font-size: 22px; font-weight: bold;'>Use Model without Training Data</span></li>
        </ul>


    New to our tool? Be sure to check out our [![Manual guide](https://img.shields.io/badge/Manual_Guide-red.svg)](https://github.com/hkust-lib-ds/P003-PUBLIC_English-and-Chinese-Topic-Modeling-Tool/ENG/README.md)  for guidance.

    PS: This tool is designed exclusively for English text. If you need to work with Chinese text, please run our [![Manual guide](https://img.shields.io/badge/Chinese_Modeling_Tool-blue.svg)](https://github.com/hkust-lib-ds/P003-PUBLIC_English-and-Chinese-Topic-Modeling-Tool/CHI/README.md) instead.
   
    </div>
    """, unsafe_allow_html=True
)


#########################################################################################################################################
#########################################################################################################################################
# session 1: Import Training Documents
st.markdown("---")
st.subheader("Upload Documents")

# upload training documents
with st.popover("Guidance for File Format"):
    st.write("**Option 1:** multiple :red[**txt files**] with each file for a document. :red[(Please use Unique File Name as the DocID)]. ")
    st.write("_Example:_")
    st.image("Data/txt_format.png",width=300)
    st.write("**Option 2:** :red[**csv file**] with each line for a document.")
    st.write("_Example:_")
    st.image("Data/csv_format.png", width=300)
    st.markdown(SampleTrainDoc(), unsafe_allow_html=True)
    st.write("**Option 3:** Any combination of 1 & 2.")
    st.write("**Option 4** (if use function 'Topic over time'): csv file with each line for a document.")
    st.write("_Example:_")
    st.image("Data/time_YMD.png", width=300)
    st.markdown(SampleTimeDoc_YMD(), unsafe_allow_html=True)
    st.image("Data/time_Timestamp.png", width=300)
    st.markdown(SampleTimeDoc_Timestamp(), unsafe_allow_html=True)
    st.image("Data/time_customized.png", width=300)
    st.markdown(SampleTimeDoc_Customized(), unsafe_allow_html=True)

uploaded_files = st.file_uploader("Upload training document txt or csv", accept_multiple_files=True, type=["txt", "csv"])
help_mes = 'Select a format according to the uploaded file. \nChoose "No Time Format" if you are not going to use function "Topic Over Time". '
help_mes += '\nTimestamp generation is provided for "Year_Month_Day" and "Customized".'

TopicOverTime = st.selectbox("Select a format for time: ", ["No Time Format", "Year_Month_Day", "Timestamp", "Customized"], index=0, disabled=session_state["ConfirmTrainDoc"], help=help_mes)
if TopicOverTime == "Customized":
    # use 2024-09-06; 2024 Sep 6th; 2024 Septmber 6th for user to provide format
    with st.popover("Guidance for Time Format"):
        table = {
            'Year': ['2024'],
            'Month': ['9', '09', 'Sep', 'September'],
            'Day': ['6', '06'],
            'Symbols': ['-', '/', 'th', 'st', 'rd', 'nd', '.' ,',', '<empty_space>'],
        }
        table = {key: '    '.join(value) for key, value in table.items()}
        table = pd.DataFrame(table.items(), columns=['Element', 'Format'])
        remark = {
            "Order": "Any order is allowed. The format will be detected according to the position of: '2024', '9', '09', '6', '06', 'Sep', 'September'.",
            "Abbreviations": "Abbreviations of month are must use the first three letters.",
            'Information of missing': "If infomation of month and/or day is missing, will be set as '00'; if year is missing, will be set as '0000'.",
            "Limitation": "If the above strategry does not satisfy your need, please reformat you dates.",
        }
        remark = pd.DataFrame(remark.items(), columns=['Remark', 'Guidance (double click to view details)'])

        st.write("Please use the combiniation of the followings to provide the time format:")
        st.write(table)
        st.write("Additionla Information:")
        st.write(remark)

    timeFormat = st.text_input("Enter the time format: ", placeholder="Examples: 2024-09-06; Sep 6th 2024, September 6th, 09/06/2024 ...", disabled=session_state["ConfirmTrainDoc"])
else:
    timeFormat = None
    
with stylable_container(
            key="ConfirmTrainDoc_Button",
            css_styles="""
                button {
                    background-color: #681172;
                    color: white;
                }
                """,
        ):
    ConfirmTrainDoc = st.button("Confirm Training Documents", disabled=session_state["ConfirmTrainDoc"])

if ConfirmTrainDoc and uploaded_files and not session_state["ConfirmTrainDoc"]:
    if TopicOverTime == "No Time Format":
        TRAIN_DOCs = processDoc(uploaded_files)
    else:
        TRAIN_DOCs = processDoc_time(uploaded_files, TopicOverTime, timeFormat)
        if TopicOverTime != "Timestamp":
            session_state['Timestamp_text'] = TimestampText(TRAIN_DOCs)
        session_state['TopicOverTime'] = TopicOverTime

    session_state["ConfirmTrainDoc"] = True


#########################################################################################################################################
#########################################################################################################################################
# sidebar
with st.sidebar:
    st.markdown("""
                <a href="https://library.hkust.edu.hk/"><img src="https://library.hkust.edu.hk/wp-content/themes/hkustlib/hkust_alignment/core/assets/library/library_logo.png_transparent_bkgd_h300.png" alt="HKUST Library Logo" style="width:200px;"/></a>
                """, unsafe_allow_html=True)

    """
    [![Manual guide](https://img.shields.io/badge/Manual_Guide-red.svg)](https://github.com/hkust-lib-ds/P003-PUBLIC_English-and-Chinese-Topic-Modeling-Tool/ENG/README.md)
    [![GitHub repo](https://badgen.net/badge/icon/GitHub/black?icon=github&label)](https://github.com/hkust-lib-ds/P003-PUBLIC_English-and-Chinese-Topic-Modeling-Tool) 

    """

st.sidebar.markdown("<p style='font-weight:600;font-size:20px;margin-bottom:-8px;'>Sidebar</p>", unsafe_allow_html=True)

# Sidebar tabs
tab_NewRun, tab_Topic, tab_updateModel = st.sidebar.tabs([ "Define Model", "View Topic", "Update Topic"])

#########################################################################################################################################

with tab_NewRun:
    st.markdown("### Upload a Model")
    uploadRun_file = st.file_uploader("Upload a model file." , type = ["pickle"], help='If want to use training data together with a uploaded model, please upload both documents before clicking "Confirm".')
    with stylable_container(
            key="Confirm",
            css_styles="""
                button {
                    background-color: #681172;
                    color: white;
                }
                """,
        ):
        confirm = st.button("Confirm")
    if uploadRun_file and confirm:
        run, TRAIN_DOCs = UploadRun(uploadRun_file, TRAIN_DOCs)
        if run:
            ALL_RUNs[uploadRun_file.name.split(".")[0]] = run
            session_state['ModelOnly'] = not session_state['ConfirmTrainDoc'] # if have not upload many data and upload model, then model only
            session_state["ConfirmTrainDoc"] = True


    if not TRAIN_DOCs:
        st.markdown("### Train a New Model")
        st.warning("Please upload training documents in the right-hand side's main page first.")
    elif not session_state["ModelOnly"]:
        st.markdown("### Train a New Model")
        NewRunName = st.text_input("Enter the name of the new model", value="Model 1")
        
        # parameters
        embedding_model = st.selectbox("Select an embedding model: ", ["DEFAULT: all-MiniLM-L6-v2", "all-MiniLM-L12-v2", "all-mpnet-base-v2", "all-distilroberta-v1"], index=0, help="(Details in: https://www.sbert.net/docs/pretrained_models.html) You can chose one according to accuracy, speed, etc.")
        top_n_words = st.number_input("Enter the number of representative words: ", min_value=1, max_value=20, value=10, help='The number of words to represent one topic. (Suggested: no greater than 10)')
        # table for n-gram examples
        table = {
            '1-gram': 'I, am, a, student',
            '2-gram': 'I am, am a, a student',
            '3-gram': 'I am a, am a student',
            '4-gram': 'I am a student',
        }
        table = pd.DataFrame(table.items(), columns=['N-gram', 'Tokens'])
        help_ngram = "N-gram: how many consecritive words to be treated as one token. (Suggested: upper bound no greater than 3)\nExamples:\n"
        help_ngram = help_ngram + table.to_markdown(index=False)
        n_gram_range = st.slider("Enter the range of n-gram: ", 1, 6, (1, 1), help=help_ngram)
        min_topic_size = st.number_input("Enter the minimum number of topics: ", min_value=5, max_value=30, value=10, help='The minimum number of topics to be generated. (Suggested: not too large)')
        nr_topics = st.selectbox("The topic reduction method", ["No Reduction", "Automatic Reduction"], index=0, help='No Reduction: keep the initial result; \nAutomatic Reduction: automatically reduce the number of topic using HDBSCAN.\nSuggested: No Reduction, becuase you can reduce later in the session "Update Model".')

        uploadSW_file = st.file_uploader("Or upload stop words with txt file ", type = ["txt"], help="Each words should be seperated by a comma (,). \nWhite space will be ingored.")
        if uploadSW_file:
            uploadSW = uploadSW_file.getvalue().decode("utf-8")
            newSW = UploadStopWords(uploadSW)
            ALL_STOP_WORDS[uploadSW_file.name] = newSW
            stop_words = newSW

        # stop words
        stop_words = st.selectbox("Select a list of stop words: ", ["None"]+list(ALL_STOP_WORDS.keys()), index=0, help='Stop words: what to exlude in the model training. (Suggested: None if your documents are in sufficient length and quantity.)')
        if stop_words == "None":
            stop_words = None
        else:
            stop_words = ALL_STOP_WORDS[stop_words]
        
        if stop_words:
            st.markdown("### Stop Words List for New Model")
            st.markdown(f"<div style='background-color: #cbe2e0; padding: 10px; border-radius: 5px; height: 150px; overflow-y: auto;'>{', '.join(stop_words)}</div>", unsafe_allow_html=True)
            st.write("")

        with stylable_container(
                key="TrainModel_Button",
                css_styles="""
                    button {
                        background-color: #681172;
                        color: white;
                    }
                    """,
            ):
            TrainModel_Button = st.button("Train Model")
        if TrainModel_Button:
            model = TrainNewModel(TRAIN_DOCs, top_n_words, n_gram_range, min_topic_size, nr_topics, embedding_model, stop_words)
            run = ModelToRun(model, TRAIN_DOCs)
            ALL_RUNs[NewRunName] = run
            st.success("Finished Training.")

#########################################################################################################################################

with tab_updateModel:

    if not ALL_RUNs:
        st.warning("To proceed to the next step, please go to the left-hand sidebar, do \"Upload a Model\" or \"Train a New Model\" first.")

    else:
        st.markdown("### Update Model")
        Update_Run = st.selectbox("Select a model to update", list(ALL_RUNs.keys()), index=0)

        # label topics
        st.markdown("### Label Topics of Current Model")
        topic_label = st.text_input("Enter the topics and labels.", placeholder="Topic 1: Health, Topic 2: Economy", help='Format: "Topic 1: Health, Topic 2: Economy".')
        with stylable_container(
                key="LabelTopic_Button",
                css_styles="""
                    button {
                        background-color: #681172;
                        color: white;
                    }
                    """,
            ):
            LabelTopic_Button = st.button("Label Topics")
        if LabelTopic_Button:
            if not session_state['ModelOnly']:
                updatedRUN = LabelTopics(ALL_RUNs[Update_Run], TRAIN_DOCs, topic_label)
            else: 
                updatedRUN = LabelTopics_nodata(ALL_RUNs[Update_Run], topic_label)
            if ALL_RUNs[Update_Run]['TOPIC_TIME_RES']:
                updatedRUN['TOPIC_TIME_RES'] = TrainModelOverTime(TRAIN_DOCs, ALL_RUNs[Update_Run]['MODEL'], session_state["Timestamp_text"], ALL_RUNs[Update_Run]) 
            ALL_RUNs[Update_Run] = updatedRUN
            st.success("Finished setting labels")

        if not session_state['ModelOnly']:
            # reduce topics
            st.markdown("### Reduce the Number of Topics")
            ReduceToNum = st.number_input("Reduce the number of topics to: ", min_value=1, max_value=len(ALL_RUNs[Update_Run]['TOPICs']), value=int((len(ALL_RUNs[Update_Run]['TOPICs']))/2))
            with stylable_container(
                    key="ReduceTopic_Button",
                    css_styles="""
                        button {
                            background-color: #681172;
                            color: white;
                        }
                        """,
                ):
                ReduceTopic_Button = st.button("Reduce Topics")
            if ReduceTopic_Button:
                updatedRun = ReduceTopic(ALL_RUNs[Update_Run], TRAIN_DOCs, ReduceToNum)
                ALL_RUNs[Update_Run] = updatedRun
                session_state["SearchTopic"] = None
                session_state["predictUploader"] = None
                session_state['PredictUsedRun'] = None

            # merge topics
            st.markdown("### Merge Topics")
            topicsToMerge = st.text_input("Enter the numerical IDs of the topics to merge using lists: ", placeholder="[[1, 2], [3, 4]]", help=' e.g. "[1, 2, 3]" will merge topics 1, 2 and 3; "[[1, 2], [3, 4]]" will merge topics 1 and 2, and separately merge topics 3 and 4.')
            with stylable_container(
                    key="MergeTopic_Button",
                    css_styles="""
                        button {
                            background-color: #681172;
                            color: white;
                        }
                        """,
                ):
                MergeTopic_Button = st.button("Merge Topics")
            if MergeTopic_Button and topicsToMerge:
                topicsToMerge = eval(topicsToMerge)
                if type(topicsToMerge) != list:
                    st.warning("Please enter a list.")
                else:
                    updatedRUN = MergeTopic(ALL_RUNs[Update_Run], TRAIN_DOCs, topicsToMerge)
                    ALL_RUNs[Update_Run] = updatedRUN
                    session_state["SearchTopic"] = None
                    session_state["predictUploader"] = None
                    session_state['PredictUsedRun'] = None
            

#########################################################################################################################################

with tab_Topic:

    if not ALL_RUNs:
        st.warning("To proceed to the next step, please go to the left-hand sidebar, do \"Upload a Model\" or \"Train a New Model\" first.")
    
    else:
        st.markdown("### Current Model")
        CUR_RUN = st.selectbox("Select a model to view", list(ALL_RUNs.keys()), index=0)
        if CUR_RUN != session_state["CUR_RUN"]:
            session_state["SearchTopic"] = None
            session_state["predictUploader"] = None
            if session_state["TopicOverTime"]:
                for docID in TRAIN_DOCs:
                    TRAIN_DOCs[docID]['timestamp'] = ALL_RUNs[CUR_RUN]['Doc_timestamp'][docID]
                if session_state["TopicOverTime"] != "Timestamp":
                    session_state['Timestamp_text'] = TimestampText(TRAIN_DOCs)

        st.markdown("### Display Filtering")
        radioList = ["All TOPICS"] 
        for topic in ALL_RUNs[CUR_RUN]['TOPICs']:
            if ALL_RUNs[CUR_RUN]['TOPICs'][topic]['LABEL']:
                radioList.append(f"{topic} | {ALL_RUNs[CUR_RUN]['TOPICs'][topic]['LABEL']}")
            else:
                radioList.append(f"{topic}")
        CUR_TOPIC = st.radio("Select a topic", radioList, index=0)
        if CUR_TOPIC == "All TOPICS":
            CUR_TOPIC = None
        elif '|' in CUR_TOPIC:
            CUR_TOPIC = CUR_TOPIC.split("|")[0].strip()
        else:
            CUR_TOPIC = CUR_TOPIC

        st.markdown("### Search Topic with a Keyword")
        search_keyword = st.text_input("Enter ONE keyword to search", value=session_state["SearchTopic"][0] if session_state["SearchTopic"] else "")
        top_n = st.number_input("Number of topics to return", min_value=1, max_value=len(ALL_RUNs[CUR_RUN]['TOPICs']), value=1)
        with stylable_container(
                key="SearchTopic_Button",
                css_styles="""
                    button {
                        background-color: #681172;
                        color: white;
                    }
                    """,
            ):
            SearchTopic_Button = st.button("Search Topic")
        if SearchTopic_Button:
            model = ALL_RUNs[CUR_RUN]['MODEL']
            similar_topics, similarity = model.find_topics(search_keyword, top_n=top_n)
            searchDict = {}
            for topic in similar_topics:
                TopicName = f"Topic {topic}"
                if ALL_RUNs[CUR_RUN]['TOPICs'][TopicName]['LABEL']:
                    TopicName = f"{TopicName} | {ALL_RUNs[CUR_RUN]['TOPICs'][TopicName]['LABEL']}"
                searchDict[TopicName] = similarity[similar_topics.index(topic)]
            session_state["SearchTopic"] = (search_keyword, searchDict)

        if session_state["SearchTopic"] and search_keyword == session_state["SearchTopic"][0]:
            st.write(f"Search Results for '{search_keyword}':")
            st.write(pd.DataFrame(session_state["SearchTopic"][1].items(), columns=['Topic', 'Similarity']))


#########################################################################################################################################
#########################################################################################################################################
# Main page: session 2: Data Display

st.markdown("---")
st.subheader("• " + "Discovered Topics")

if not ALL_RUNs:
    st.warning("To proceed to the next step, please go to the left-hand sidebar, do \"Upload a Model\" or \"Train a New Model\" first.")

else:
    #########################################################################################################################################
    #########################################################################################################################################
    # export
    col_downRun, col_downData, col_note = st.columns(3)

    with col_downRun:
        with stylable_container(
                key="DownloadRun_Button",
                css_styles="""
                    button {
                        background-color: #157739;
                        color: white;
                    }
                    """,
            ):
            DownloadRun_Button = st.button("Download Model")
        if DownloadRun_Button:
            DownloadRun(ALL_RUNs[CUR_RUN], CUR_RUN)

    with col_downData:
        with stylable_container(
                    key="ExportModelData_Button",
                    css_styles="""
                        button {
                            background-color: #157739;
                            color: white;
                        }
                        """,
                ):
            ExportModelData_Button = st.button("Export Model Data")
        if ExportModelData_Button:
            if session_state['ModelOnly']:
                st.warning("Using model only. Cannot Export Model Data. Can only download the model.")
            else:
                ExportModelData(ALL_RUNs[CUR_RUN], TRAIN_DOCs, CUR_RUN)

    # Remark for display
    with col_note:
        with st.popover("Remark for Display"):
            table = {
                "Topic": "Display : topic name, topic label (if any), representative words.",
                "Document": "Display: document ID, topic assisned to with corresponding probability, document content with representitive words of assigned topic being bold.",
                "Color of box": "Each colors is corresponding to a topic.",
                "Filtering of Display": "Select a topic in sidebar to display corresponding content only.",
            }
            table = pd.DataFrame(table.items(), columns=['Element', 'Remark'])
            st.markdown(
                f"""
                <div style="overflow-x: auto; width: 600px;">
                    {table.to_html(index=False)}
                </div>
                """,
                unsafe_allow_html=True
            )


    #########################################################################################################################################
    #########################################################################################################################################
    # visualization
    st.markdown("### Visualization")
    if len(ALL_RUNs[CUR_RUN]['TOPICs']) < 5:
        st.warning("The number of topics is too small (less than 5). Some visualization may not be available.")

    if not CUR_TOPIC:
        TABs_vis = st.tabs(list(ALL_RUNs[CUR_RUN]['FIGs'].keys()))
        for i, figKey in enumerate(ALL_RUNs[CUR_RUN]['FIGs'].keys()):
            if not ALL_RUNs[CUR_RUN]['FIGs'][figKey]:
                st.warning(f"No figure. Maybe the number of topics or training documents is too small.")
                continue
            with TABs_vis[i]:
                if figKey == "Topic WordCloud":
                    wordcloud = ALL_RUNs[CUR_RUN]['FIGs'][figKey]
                    st.image(wordcloud.to_array(), caption="WordCloud for all topics")
                elif figKey == 'Document-Topic Heatmap':
                    tabs = st.tabs([f"P.{p}" for p in range(1, len(ALL_RUNs[CUR_RUN]['FIGs'][figKey])+1)])
                    for j, heatmap in enumerate(ALL_RUNs[CUR_RUN]['FIGs'][figKey]):
                        with tabs[j]:
                            st.pyplot(heatmap)
                elif figKey == 'Table Summary for Training Documents':
                    st.markdown("##### Training Documents Information")
                    DisplayTrain_table(ALL_RUNs[CUR_RUN], TRAIN_DOCs, CUR_TOPIC)
                else:
                    st.plotly_chart(ALL_RUNs[CUR_RUN]['FIGs'][figKey])

    else: # one topic
        TABs_vis = st.tabs(list(ALL_RUNs[CUR_RUN]['TOPICs'][CUR_TOPIC]['Figs'].keys()))
        for i, figKey in enumerate(ALL_RUNs[CUR_RUN]['TOPICs'][CUR_TOPIC]['Figs'].keys()):
            if not ALL_RUNs[CUR_RUN]['TOPICs'][CUR_TOPIC]['Figs'][figKey]:
                st.warning(f"No figure. Maybe the number of topics or training documents is too small.")
                continue
            with TABs_vis[i]:
                if figKey == 'WordCloud':
                    wordcloud = ALL_RUNs[CUR_RUN]['TOPICs'][CUR_TOPIC]['Figs'][figKey]
                    if not ALL_RUNs[CUR_RUN]['TOPICs'][CUR_TOPIC]['LABEL']:
                        st.image(wordcloud.to_array(), caption=f"WordCloud for {CUR_TOPIC}")
                    else:
                        st.image(wordcloud.to_array(), caption=f"WordCloud for {CUR_TOPIC} | {ALL_RUNs[CUR_RUN]['TOPICs'][CUR_TOPIC]['LABEL']}")
                    continue
                elif figKey == 'Document-Topic Heatmap':
                    tabs = st.tabs([f"P.{p}" for p in range(1, len(ALL_RUNs[CUR_RUN]['TOPICs'][CUR_TOPIC]['Figs'][figKey])+1)])
                    for j, heatmap in enumerate(ALL_RUNs[CUR_RUN]['TOPICs'][CUR_TOPIC]['Figs'][figKey]):
                        with tabs[j]:
                            st.pyplot(heatmap)
                elif figKey == 'Table Summary for Training Documents':
                    DisplayTrain_table(ALL_RUNs[CUR_RUN], TRAIN_DOCs, CUR_TOPIC)
                elif figKey == 'Table Summary for Words and Scores':
                    st.markdown("##### Words and Scores")
                    statistics = ALL_RUNs[CUR_RUN]['TOPICs'][CUR_TOPIC]['WORDs']
                    df = pd.DataFrame(statistics.items(), columns=['Word', 'Score'])
                    st.table(df)
                else:
                    st.plotly_chart(ALL_RUNs[CUR_RUN]['TOPICs'][CUR_TOPIC]['Figs'][figKey])
    
    #########################################################################################################################################
    # display
    st.markdown("### Details")

    col_topic, col_doc = st.columns(2)

    # display topic models and documents with text
    with col_topic:
        st.markdown("#### Topics")
        st.markdown('---')
        DisplayTopic(ALL_RUNs[CUR_RUN], CUR_TOPIC)

    with col_doc:
        st.markdown("#### Documents")
        if not session_state["ModelOnly"]:
            DisplayDoc(ALL_RUNs[CUR_RUN], CUR_TOPIC, TRAIN_DOCs)
        else:
            DisplayDoc_nodata(ALL_RUNs[CUR_RUN], CUR_TOPIC, TRAIN_DOCs)

    #########################################################################################################################################
    #########################################################################################################################################
    # session 3: Prediction
    st.markdown("---")
    st.subheader("• " + "Apply the Model to New Documents")

    # upload prediction documents
    with st.popover("Guidance for File Format"):
        st.write("**Option 1:** multiple :red[**txt files**] with each file for a document. :red[(Please use Unique File Name as the DocID)]. ")
        st.write("_Example:_")
        st.image("Data/txt_format.png",width=300)
        st.write("**Option 2:** :red[**csv file**] with each line for a document.")
        st.write("_Example:_")
        st.image("Data/csv_format.png", width=300)
        st.markdown(SampleTrainDoc(), unsafe_allow_html=True)
        st.write("**Option 3:** Any combination of 1 & 2.")

    
    uploaded_files = st.file_uploader("Upload document for prediction txt or csv", accept_multiple_files=True, type = ["txt", "csv"])

    if not uploaded_files:
        # clean all prediction data
        PREDICT_DOCs = {}
        PREDICT_RES = {}
        session_state["PredictUploader"] = None

    if uploaded_files and uploaded_files != session_state["PredictUploader"]:
        PREDICT_DOCs = processDoc(uploaded_files)
        PREDICT_RES = {}
        session_state["PredictUploader"] = uploaded_files
            

    col1, col2 = st.columns(2)
    with col1:
        with stylable_container(
                key="PredictDoc_Button",
                css_styles="""
                    button {
                        background-color: #681172;
                        color: white;
                    }
                    """,
            ):
            PredictDoc_Button = st.button("Predict using current model")
        if PredictDoc_Button:
            PREDICT_DOCs, PREDICT_RES = PredictDoc(ALL_RUNs[CUR_RUN], PREDICT_DOCs)
            session_state["PredictUsedRun"] = CUR_RUN
    with col2:
        with stylable_container(
                key="ExportPredictData_Button",
                css_styles="""
                    button {
                        background-color: #157739;
                        color: white;
                    }
                    """,
            ):
            ExportPredictData_Button = st.button("Export Prediction Results")
        if ExportPredictData_Button:
            ExportPredictData(ALL_RUNs[CUR_RUN], PREDICT_DOCs, CUR_RUN, PREDICT_RES)

    #########################################################################################################################################
    # Display Prediction Results
    if CUR_RUN == session_state["PredictUsedRun"] and PREDICT_RES:
        st.markdown(f"### Prediction Results using Model: {CUR_RUN}")
        DisplayPredict(ALL_RUNs[CUR_RUN], CUR_TOPIC, PREDICT_DOCs, PREDICT_RES)


#########################################################################################################################################
#########################################################################################################################################
# session 4: Topic Modelling Over Time
if session_state["TopicOverTime"] and CUR_RUN and not session_state["ModelOnly"]:

    st.markdown("---")
    st.subheader("• " + "Track Topics Changes Over Time")

    if  session_state["TopicOverTime"] != "Timestamp":
        st.markdown("### Generate Timestamp")
        cluster_K = st.number_input("Set K value for K-mean clustering.", min_value=1, max_value=100, value=1, help=' K: How many cluster you want to split the documents according to time. \nPlease choose an appropriate value of K according to the scatter plot below.')
        
        col1, col2 = st.columns(2)
        with col1:
            with stylable_container(
                    key="GenerateTimestamp_Button",
                    css_styles="""
                        button {
                            background-color: #681172;
                            color: white;
                        }
                        """,
                ):
                GenerateTimestamp_Button = st.button("Generate Timestamp")
        if GenerateTimestamp_Button:
            TRAIN_DOCs = GenerateTimestamp(TRAIN_DOCs, cluster_K)
            ALL_RUNs[CUR_RUN]['TOPIC_TIME_RES'] = None
            Doc_timestamp = {}
            for docID in TRAIN_DOCs:
                Doc_timestamp[docID] = TRAIN_DOCs[docID]['timestamp']
            ALL_RUNs[CUR_RUN]['Doc_timestamp'] = Doc_timestamp
            session_state['Timestamp_text'] = TimestampText(TRAIN_DOCs)
            st.success("Finished generating timestamp.")

        with col2:
            with st.popover("Remark for Display"):
                table = {
                    "dots in the plot": "Each dot represents a document.",
                    "side panel": "Timestamps with the corresponding periods of time.",
                    "Color": "Each color corresponds to a timestamp.",
                    "X-axis": "The origin is the earliest date. Other = 365*years + 30*month + 1*days.",}
                table = pd.DataFrame(table.items(), columns=['Element', 'Remark (Double click to view details)'])
                st.markdown(
                f"""
                <div style="overflow-x: auto; width: 600px;">
                    {table.to_html(index=False)}
                </div>
                """,
                unsafe_allow_html=True
                )

        fig = Display_Timestamp(TRAIN_DOCs, session_state["TopicOverTime"], session_state['Timestamp_text'])
        if fig:
            st.plotly_chart(fig)
            st.markdown("### Train Topic Model Over Time")

    col1, col2, col3 = st.columns(3)
    with col1:
        with stylable_container(
                key="TrainModelOverTime_Button",
                css_styles="""
                    button {
                        background-color: #681172;
                        color: white;
                    }
                    """,
            ):
            TrainModelOverTime_Button = st.button("Find Changes Over Time")
        if TrainModelOverTime_Button:
            TOPIC_TIME_RES = TrainModelOverTime(TRAIN_DOCs, ALL_RUNs[CUR_RUN]['MODEL'], session_state["Timestamp_text"], ALL_RUNs[CUR_RUN]) 
            ALL_RUNs[CUR_RUN]['TOPIC_TIME_RES'] = TOPIC_TIME_RES    

    with col2:
        with stylable_container(
                key="ExportModelData_time_Button",
                css_styles="""
                    button {
                        background-color: #157739;
                        color: white;
                    }
                    """,
            ):
            ExportModelData_time_Button = st.button("Export Data Over Time")
        if ExportModelData_time_Button:
            ExportTopicOverTimeData(ALL_RUNs[CUR_RUN], TRAIN_DOCs, CUR_RUN, session_state['Timestamp_text'])

    with col3:
        with st.popover("Remark for Display"):
            table = {
                "Topic Models Over Time": "For each timestamp, a topic model is trained.",
                "Timestamp": "Each timestamp corresponds to a peroiod of time.",
                "Bar Chart": "x-axis: timestamp; y-axis: frequency, i.e. number of documents; color: distinguish topics.",
            }
            table = pd.DataFrame(table.items(), columns=['Element', 'Remark'])
            st.markdown(
                f"""
                <div style="overflow-x: auto; width: 600px;">
                    {table.to_html(index=False)}
                </div>
                """,
                unsafe_allow_html=True
            )
        

    # display topic model over time
    if not CUR_TOPIC and ALL_RUNs[CUR_RUN]['TOPIC_TIME_RES']:
        tabs = st.tabs(list(ALL_RUNs[CUR_RUN]['TOPIC_TIME_RES']['Figs'].keys()))
        for i, figKey in enumerate(ALL_RUNs[CUR_RUN]['TOPIC_TIME_RES']['Figs'].keys()):
            with tabs[i]:
                if not ALL_RUNs[CUR_RUN]['TOPIC_TIME_RES']['Figs'][figKey]:
                    st.warning(f"No figure. Maybe the number of topics or training documents is too small.")
                    continue
                if figKey == 'Individual Bar Chart':
                    try:
                        fig = ALL_RUNs[CUR_RUN]['TOPIC_TIME_RES']['Figs'][figKey]['']
                        st.plotly_chart(fig)
                    except:
                        st.warning("No figure. Maybe too many topics to display.")
                else:
                    st.plotly_chart(ALL_RUNs[CUR_RUN]['TOPIC_TIME_RES']['Figs'][figKey])
    elif CUR_TOPIC and ALL_RUNs[CUR_RUN]['TOPIC_TIME_RES']:
        tabs = st.tabs(['Individual Bar Chart'])
        with tabs[0]:
            try:
                fig = ALL_RUNs[CUR_RUN]['TOPIC_TIME_RES']['Figs']['Individual Bar Chart'][CUR_TOPIC]
                st.plotly_chart(fig)
            except:
                st.warning("No figure. Maybe too many topics to display.")


#########################################################################################################################################
#########################################################################################################################################
# save data to session_state
for key, value in default_session_state.items():
    session_state[key] = globals()[key]