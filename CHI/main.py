import streamlit as st
from streamlit_extras.stylable_container import stylable_container
from utils.i_o import *
from utils.topic import *
from utils.display import *
from utils.visualization import *
from utils.stopwords import *
import os
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
from bert import *
from bertopic import BERTopic
from sentence_transformers import SentenceTransformer
from utils.overtime import *
import shutil
import seaborn
import plotly.io as pio
import pandas as pd


# Colors
# HKUST Blue: #013365 
# HKUST Gold: #996600
# HKUST Yellow: #f5cc4a

# Initialize session state variables if they don't exist
if 'Exist_training_model' not in st.session_state:
    st.session_state.Exist_training_model = False
if 'raw_topics' not in st.session_state:
    st.session_state.raw_topics = {}
if 'topics' not in st.session_state:
    st.session_state.topics = {}
if 'trained_model' not in st.session_state:
    st.session_state.trained_model = None
if 'originalDoc' not in st.session_state:
    st.session_state.originalDoc = {}
if 'trainingDoc' not in st.session_state:
    st.session_state.trainingDoc = {}
if 'stop_words' not in st.session_state:
    st.session_state.stop_words = None
if 'filtered_topic' not in st.session_state:
    st.session_state.filtered_topic = None
if 'predictingDoc' not in st.session_state:
    st.session_state.predictingDoc = {}
if 'predicted_topics' not in st.session_state:
    st.session_state.predicted_topics = {}
if 'predicted_df' not in st.session_state:
    st.session_state.predicted_df = None
if 'predicted_probs' not in st.session_state:
    st.session_state.predicted_probs = []
if 'model_count' not in st.session_state:
    st.session_state.model_count = 0
if 'top_n_words' not in st.session_state:
    st.session_state.top_n_words = 0
if 'Model_is_uploaded' not in st.session_state:
    st.session_state.Model_is_uploaded = False
if 'Topic_over_time' not in st.session_state:
    st.session_state.Topic_over_time = ''
if 'Timestamp_text' not in st.session_state:
    st.session_state.Timestamp_text = {}
if 'Doc_timestamp' not in st.session_state:
    st.session_state.Doc_timestamp = {}
if 'topic_time' not in st.session_state:
    st.session_state.topic_time = {}
if 'training_doc_time' not in st.session_state:
    st.session_state.training_doc_time = {}
if 'stop_words_modified' not in st.session_state:
    st.session_state.stop_words_modified = False
if 'time_format' not in st.session_state:
    st.session_state.time_format = None

st.set_page_config(
    page_title="中文主題分析工具 Chinese Topic Modeling Tool | HKUST Library",
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

#####################
#####################
##### Main Page #####
#####################
#####################

st.caption("Chinese Topic Modeling Tool")
st.title("中文文本主題分析工具")

st.markdown(
    """
    <div style='background-color: #f0f0f5; padding: 20px; border-radius: 10px;'>
        <p style='font-size: 16px;'>
            <span style='font-size: 22px; font-weight: bold;'>主題模型/主題建模/主題分析 (Topic Modeling)</span> 是一種自然語言處理 (NLP) 技術，透過分析文本中的文字的出現概率和分佈，使研究人員能夠從語料庫中提取有意義的主題，進而進行主題聚類或文本分類。我們這個工具使用了 BERTopic 這個主題分佈模型，透過預先訓練的BERT模型來對文本中的語義進行聚類，從而得出每個文檔的主題。
            <br/><br/>
            Topic modeling is a natural language processing technique (NLP), a statistical method used for discovering abstract topics within a collection of text. It helps in organizing, understanding, and summarizing large datasets by identifying patterns and themes present in the text. The model analyzes word co-occurrences and assigns topics to documents, allowing researchers to extract meaningful insights from the corpora. This tool utilizes the BERTopic model for topic distribution, clustering the semantics in the text through a pre-trained BERT model to identify the topics of each document.
        </p>

    第一次使用這個工具？ 請參閱我們的 [![使用手冊](https://img.shields.io/badge/使用手冊-red.svg)](https://github.com/hkust-lib-ds/P003-PUBLIC_English-and-Chinese-Topic-Modeling-Tool/blob/main/CHI/README.md)。

    注意: 這個工具是專為中文文本設計，如果您需要處理英文文本，請使用我們的 [![Manual guide](https://img.shields.io/badge/English_Modeling_Tool-blue.svg)](https://github.com/hkust-lib-ds/P003-PUBLIC_English-and-Chinese-Topic-Modeling-Tool/blob/main/ENG/README.md)。
   
    </div>
    """, unsafe_allow_html=True
)

#############################
# Import training documents #
#############################

st.markdown("---")
st.subheader("上載文件 Upload Documents")

# upload training documents
with st.popover("上傳格式指引"):
    st.write("**方法 1:** 多個 :red[**txt 檔案**]，每個檔案對應一個文本。 :red[（每個檔案必須有不同的名稱）] ")
    st.write("**方法 2:** :red[**csv 檔案**]，每行對應一個文本。標題列: `docID,content` 或 `docID,content,time`")
    st.write("**方法 3:** 如果想使用功能「隨時間的主題」，請上傅 csv 格式的檔案，每行對應一個文本。")
    st.image("manual-img/3.png")


uploaded_files = None

if st.session_state.Model_is_uploaded:
    st.markdown('模型為用戶上載，由於Bertopic模型不儲存文本信息，故無法獲取具體訓練文本。如您希望展示文本信息，請於下方上載您先前用於探索主題的文本 \n\n Unable to get the information of the training documents as they are not saved within Bertopic. Please upload the files you used to discover these topics.')
    uploaded_files = st.file_uploader("上載您先前用於探索主題的文件 \n\n Upload the documents you used to discover these topics", accept_multiple_files=True, type=["txt", "csv"])
else:
    uploaded_files = st.file_uploader("上載您希望用於探索主題的文件 \n\n Upload your documents for discovering topics", accept_multiple_files=True, type=["txt", "csv"])

st.write(":red[請待右上角`Running...`完成，然後才再上載文件/進行下一步]")
st.session_state.Topic_over_time = st.selectbox("如果您希望追蹤主題隨時間的變化，請選擇您的時間格式；如無相關需求，默認保持“不追蹤” \n\n Topic over time: select a time format if want to train a model over time. ", 
                                                ["不追蹤", "年月日", "時間戳", "自定義"], index=0)
if st.session_state.Topic_over_time == "自定義":
    # use 2024-09-06; 2024 Sep 6th; 2024 Septmber 6th for user to provide format
    with st.popover("關於時間格式的指引 Guidance for Time Format"):

        st.markdown(
                f"""
                ### 自定義時間格式 Customized Date Format
                <table>
                    <tr>
                        <th>年 Year</th>
                        <td><code>2024</code></td>
                    </tr>
                    <tr>
                        <th>月 Month</th>
                        <td><code>9</code> <code>09</code> <code>Sep</code> <code>September</code></td>
                    </tr>
                        <th>日 Day</th>
                        <td><code>6</code> <code>06</code></td>
                    </tr>
                    <tr>
                        <th>符號 Symbols</th>
                        <td><code>-</code> <code>/</code> <code>.</code> <code>,</code> <code><empty_space></code> <code>st</code> <code>nd</code> <code>rd</code> <code>th</code></td>
                    </tr>
                </table>
                """,
                unsafe_allow_html=True
            )

        st.markdown("""
                - 月份的縮寫必須使用前三個字母
                - 如果缺少月份和/或日期的信息，將設置為 `00`；如果缺少年份，將設置為 `0000`
                - 如果上述策略無法滿足您的需求，請重新格式化您的日期
                - 任何順序都是允許的，格式將根據元素的位置進行檢測
                - 在下方空格，請以2024年9月6日為例，輸入您的時間格式，程式將根據您輸入的元素位置進行檢測，例如 : `2024-09-06` / `2024 Sep 6` / `6th Septmber 2024`
                """)
    st.session_state.time_format = st.text_input("請依據上方指示, 輸入您的時間格式")


with stylable_container(
        key="ConfirmTrainDoc_Button",
        css_styles="""
            button {
                background-color: #f5cc4a;
                color: #013365;
            }
            """,
):
    ConfirmTrainDoc = st.button("確認上載文件 \n\n Confirm uploading files")

if ConfirmTrainDoc and uploaded_files:
    if st.session_state.Topic_over_time == '不追蹤':
        st.session_state.originalDoc = load_files(uploaded_files)
        st.session_state.trainingDoc = process_files(st.session_state.originalDoc)
        #st.write(st.session_state.trainingDoc)
        #st.write(st.session_state.originalDoc)
    else:
        st.session_state.originalDoc = load_files_time(uploaded_files, st.session_state.Topic_over_time, st.session_state.time_format)
        #st.write(st.session_state.originalDoc)
        st.session_state.trainingDoc = process_files_time(st.session_state.originalDoc)
        # st.write(st.session_state.trainingDoc)

####################
####################
##### Side Bar #####
####################
####################

with st.sidebar:
    st.markdown("""
                <a href="https://library.hkust.edu.hk/"><img src="https://library.hkust.edu.hk/wp-content/themes/hkustlib/hkust_alignment/core/assets/library/library_logo.png_transparent_bkgd_h300.png" alt="HKUST Library Logo" style="width:200px;"/></a>
                """, unsafe_allow_html=True)
    """
    [![Manual guide](https://img.shields.io/badge/使用手冊Manual-red.svg)](https://github.com/hkust-lib-ds/P003-PUBLIC_English-and-Chinese-Topic-Modeling-Tool/blob/main/CHI/README.md)
    [![GitHub repo](https://badgen.net/badge/icon/GitHub/black?icon=github&label)](https://github.com/hkust-lib-ds/P003-PUBLIC_English-and-Chinese-Topic-Modeling-Tool) 
    """
    

st.sidebar.markdown("<p style='font-weight:600;font-size:20px;margin-bottom:-8px;'>側邊欄</p>", unsafe_allow_html=True)
tab_Settings, tab_Topic, tab_Model = st.sidebar.tabs(["參數設定 \n\n Parameters", "主題展示 \n\n Topics", "上載模型 \n\n Models"])




#########################################
# Get the customized settings for model #
#########################################
with tab_Model:
    st.markdown("清除您當前的分析紀錄 \n\n Clear your current topic model")
    with stylable_container(
            key="RefreshModel_Button",
            css_styles="""
                button {
                    background-color: #f5cc4a;
                    color: #013365;
                }
                """,
    ):
        RefreshModel_Button = st.button("復原 \n\n Refresh")
    if RefreshModel_Button:
        st.session_state.Exist_training_model = False
        st.session_state.raw_topics = {}
        st.session_state.topics = {}
        st.session_state.trained_model = None
        st.session_state.originalDoc = {}
        st.session_state.trainingDoc = {}
        st.session_state.stop_words = None
        st.session_state.filtered_topic = None
        st.session_state.predictingDoc = {}
        st.session_state.predicted_topics = {}
        st.session_state.predicted_df = None
        st.session_state.predicted_probs = []
        st.session_state.model_count = 0
        st.session_state.top_n_words = 0
        st.session_state.Model_is_uploaded = False
        st.session_state.Topic_over_time = ''
        st.session_state.Timestamp_text = {}
        st.session_state.Doc_timestamp = {}
        st.session_state.topic_time = {}
        st.session_state.training_doc_time = {}
        st.session_state.modified_stopwords = []
        st.session_state.time_format = None

    st.markdown("---")
    st.markdown("上載您先前保存的分析模型 \n\n Upload your saved topic model")
    uploaded_model = st.file_uploader("上載您先前保存的分析模型 \n\n Upload your saved topic model", type = ['zip'])


    #path = st.text_input('請輸入您的模型路徑 \n\n Please enter the filepath for your model')
    with stylable_container(
            key="UploadModel_Button",
            css_styles="""
                button {
                    background-color: #f5cc4a;
                    color: #013365;
                }
                """,
    ):
        UploadModel_Button = st.button("上載模型 \n\n Upload the model")
    if UploadModel_Button and uploaded_model:
        temp_dir = unzip_model(uploaded_model)
        #temp_dir = tempfile.mkdtemp()
        #with zipfile.ZipFile(uploaded_model, 'r') as zip_ref:
            #zip_ref.extractall(path = temp_dir)
        #folder = os.listdir(temp_dir)[0]
        #path = temp_dir + '/' + folder
        #st.write(temp_dir)
        #st.write(os.listdir(temp_dir))
        st.session_state.trained_model = BERTopic.load(temp_dir)
        shutil.rmtree(temp_dir)
        st.session_state.Exist_training_model = True
        st.session_state.raw_topics = st.session_state.trained_model.get_topics()
        st.session_state.topics = processed_topics(st.session_state.raw_topics, st.session_state.trained_model)
        #st.write(st.session_state.topics)
        st.session_state.Model_is_uploaded = True

#########################################
# Get the customized settings for model #
#########################################

with tab_Settings:
    st.markdown("## 參數 Parameters")
    # Get the parameters for the model training
    min_topic_size = st.number_input("請輸入主題需要包含的文檔數量最小值 \n\n Please enter min_topic size", 
                                     min_value = 1, max_value = 30, value = 5,
                                     help = '當一個主題的文檔數量少於您所定義的值時，該主題可能會被合併到其他主題或被忽略。 \n\n It defines the threshold below which a topic may either be merged with other topics or disregarded altogether.')
    st.session_state.top_n_words = st.number_input("請輸入每個主題的代表詞匯數量 \n\n Please enter top_n_words", 
                                                   min_value = 1, max_value = 20, value = 10,
                                                   help = '該參數用於指定每個主題中顯示的關鍵詞數量。 \n\n It is a parameter that specifies the number of top words to be displayed for each topic.')
    max_topic_num = st.number_input("請輸入您希望呈現的最大主題數量 \n\n Please enter max_topic_num", 
                                    min_value = 1, max_value = 20, value = 10,
                                    help = '該參數用於設置模型可以生成的最大主題數量。 \n\n It is a parameter that sets the maximum number of topics that the model can generate.')
    # Get the stopwords settings
    st.session_state.stop_words = load_stopwords("data/baidu_stopwords.txt")
    #if st.session_state.stop_words:
    choose_another_list = st.radio('目前的停用詞表如下所示，請問您需要更換停用詞表嗎？ \n\n Current stopwords list is as follows, would you like to choose another list?',
                                    ['需要 yes', '不需要 no'], index = 1
                                    )
    if choose_another_list == '需要 yes':
        stop_words = st.selectbox("選擇合適的停用詞表 \n\n Choose your stopwords list", 
                                ["默認：百度停用詞表 \n\n Default: Baidu stopwords", "哈工大停用詞表 \n\n HIT stopwords", "中文停用詞表 \n\n CN stopwords", "四川大學機器智能實驗室停用詞庫 \n\n SCU stopwords", "自定義 \n\n Custimization"],
                                help = '停用詞是指在處理文本數據之前通常會被過濾掉的常見詞語，例如“和”、“的”、“是”等。 \n\n Stopwords are common words that are typically filtered out before processing text data, such as "and," "the," "is," and similar terms.')
        #if stop_words == "默認：百度停用詞表 \n\n Default: Baidu stopwords":
            #st.session_state.stop_words = load_stopwords("data/baidu_stopwords.txt")
        if stop_words == "哈工大停用詞表 \n\n HIT stopwords":
            st.session_state.stop_words = load_stopwords("data/hit_stopwords.txt")
        elif stop_words == "中文停用詞表 \n\n CN stopwords":
            st.session_state.stop_words = load_stopwords("data/cn_stopwords.txt")
        elif stop_words == "四川大學機器智能實驗室停用詞庫 \n\n SCU stopwords":
            st.session_state.stop_words = load_stopwords("data/scu_stopwords.txt")
        elif stop_words == "自定義 \n\n Custimization":
            customized_stopwords = st.file_uploader("上傳您的自定義停用詞表 每一行為一個停用詞 \n\n Upload your customized stopwords list with one word per line.", type=["txt"])
            if customized_stopwords:
                st.session_state.stop_words = load_stopwords(customized_stopwords) 
            else:
                #st.session_state.stop_words = load_stopwords("data/baidu_stopwords.txt")
                st.warning("未檢測到您上傳有效的自定義停用詞表，默認使用百度停用詞表 \n\n No effective customized stopwords list. Use Baidu stopwords by default.")
    else:
        modified_stopwords = st.text_input("如果您需要增加/減少停用詞，請在下方輸入: \n\n Typy in the stopwords you would like to append/delete.",
                                        help = "您可以檢視下方顯示的停用詞表以及探索主題的結果，增加特定的停用詞或移除不必要的停用詞。請以空格分隔多個詞匯，輸入內容無繁體/簡體中文之分。 \n\n You can append or remove specific stopwords according to the display above or the results of your model. Please seperate by space if you have multiple inputs. There's no need to distinguish between Traditional/Simplified Chinese.")
        modified_stopwords = modified_stopwords.split()
        col1, col2 = st.columns(2)
        with col1:
            with stylable_container(
                    key="AddWord_Button",
                    css_styles="""
                        button {
                            background-color: #013365;
                            color: white;
                        }
                        """,
            ):
                AddWord_Button = st.button("加入 \n\n Append")
        with col2:
            with stylable_container(
                    key="DeleteWord_Button",
                    css_styles="""
                        button {
                            background-color: #013365;
                            color: white;
                        }
                        """,
            ):
                DeleteWord_Button = st.button("移除 \n\n Remove")

        if AddWord_Button and modified_stopwords:
            st.session_state.stop_words = AddStopwords(st.session_state.stop_words, modified_stopwords)
            st.session_state.stop_words_modified = True
        elif DeleteWord_Button and modified_stopwords:
            st.session_state.stop_words = DeleteStopwords(st.session_state.stop_words, modified_stopwords)
            st.session_state.stop_words_modified = True

    st.markdown("### 現在使用的停用詞表 \n\n Current stopwords list")
    st.markdown(f"<div style='background-color: #cbe2e0; padding: 10px; border-radius: 5px; height: 150px; overflow-y: auto;'>{', '.join(st.session_state.stop_words)}</div>", unsafe_allow_html=True)
    st.write("")
 
    with stylable_container(
            key="TrainModel_Button",
            css_styles="""
                button {
                    background-color: #f5cc4a;
                    color: #013365;
                }
                """,
    ):
        TrainModel_Button = st.button("探索主題 \n\n Discover topics")
    
    #st.write(st.session_state.stop_words[-1])
     
    if TrainModel_Button:  
        #st.write(st.session_state.trainingDoc)
        doc = {}
        #st.write(st.session_state.trainingDoc)
        for key in st.session_state.trainingDoc.keys():
            doc[key] = st.session_state.trainingDoc[key]['content']
        #doc[] = [st.session_state.trainingDoc[key]['content'] for key in st.session_state.trainingDoc.keys()]
        st.session_state.trained_model = training_model(doc, st.session_state.stop_words, st.session_state.top_n_words, min_topic_size, max_topic_num)
        st.session_state.raw_topics = st.session_state.trained_model.get_topics()
        #st.write(st.session_state.raw_topics)
        st.session_state.topics = processed_topics(st.session_state.raw_topics, st.session_state.trained_model)
        #st.write(st.session_state.topics)
        st.session_state.Exist_training_model = True
        st.session_state.Model_is_uploaded = False
        st.success("訓練完成！ \n\n Training finished!")
        if len(st.session_state.topics) < max_topic_num:
            st.warning("主題詞數量少於最大期望數量，無需減少；如果您希望生成更多主題詞，請參考使用教程進行參數調整。 \n\n Not enough topic words to be reduced. Modify the parameters if more topic words are expected.")


#######################################
# Prepare for the topic visualization #
#######################################
with tab_Topic:

    st.markdown("## 主題 Topic")
    if not st.session_state.Exist_training_model:
        st.warning("當前無已訓練模型，請在側邊欄-參數設定完成模型訓練或上載。 \n\n No trained model, please train or upload your model in the sidebar.")
    else:
        st.markdown("---")
        st.markdown("### 篩選 Filter")
        radioList = ["所有主題 All topics"]
        for topic in st.session_state.topics:
            if st.session_state.topics[topic]['LABEL']:
                radioList.append(f"{topic} | {st.session_state.topics[topic]['LABEL']}")
            else:
                radioList.append(f"{topic}")
        filtered_topic = st.radio("選擇您所需要的主題 \n\n Choose your topic", radioList, index=0)
        if filtered_topic == '所有主題 All topics':
            st.session_state.filtered_topic = None
        elif '|' in filtered_topic:
            st.session_state.filtered_topic = filtered_topic.split("|")[0].strip()
        else:
            st.session_state.filtered_topic = filtered_topic

        st.markdown("---")    
        st.markdown("### 命名 Label")
        topic_id = st.number_input("選擇需要命名的主題編號 \n\n Choose the topic ID for labeling", min_value=-1, max_value=len(st.session_state.topics)-2, value=-1)
        topic_label = st.text_input("輸入選定主題的自定義名稱 \n\n Input your label")
        with stylable_container(
                key="LabelTopic_Button",
                css_styles="""
                    button {
                        background-color: #013365;
                        color: white;
                    }
                    """,
        ):
            LabelTopic_Button = st.button("主題命名 \n\n Labeling")
        
        if LabelTopic_Button:
            topic_label_mapping = {}
            topic_label_mapping[topic_id] = topic_label
            st.session_state.trained_model.set_topic_labels(topic_label_mapping)
            st.session_state.topics[f'Topic {topic_id}']['LABEL'] = topic_label


            

###########
# Display #
###########

st.markdown("---")
st.subheader("• " + "探索主題 Discover Topics")
#st.write(st.session_state.trainingDoc)
# Todo
if not st.session_state.Exist_training_model:
    st.warning("當前無已訓練模型，請在側邊欄-參數設定完成模型訓練或上載 \n\n No trained model, please train or upload your model in the sidebar.")

elif not st.session_state.trainingDoc:
    st.markdown("### 可視化 Visualization")
    tab_Barchart, tab_intertopicDistMap, tab_wordCloud, tab_similarityMatrix = st.tabs([
            "長條圖 \n\n Barchart",
            "主題間距圖 \n\n Intertopic Distance Map", 
            "詞雲庫 \n\n WordCloud", 
            "相似矩陣 \n\n Similarity Matrix"])
    
    with tab_Barchart:
        st.markdown("##### 長條圖 Barchart")
        top_n_topics = st.number_input('請輸入您希望呈現的主題數量 \n\n Choose the number of topics in the barchart', min_value = 1, max_value = len(st.session_state.topics) - 1, value = min(8, len(st.session_state.topics)), key = '11')
        top_n_words = st.number_input('請輸入您希望每個主題呈現的關鍵詞數量 \n\n Choose the number of words per topic in the barchart', min_value = 3, max_value=st.session_state.top_n_words, value = 5)
        fig = st.session_state.trained_model.visualize_barchart(top_n_topics=top_n_topics, n_words=top_n_words, custom_labels=True)
        st.write(fig)
        buf = io.BytesIO()
        pio.write_image(fig, buf, format='png')
        buf.seek(0)  # Rewind the buffer
        with stylable_container(
        key="downloadBarchart_Button",
        css_styles="""
            button {
                background-color: #157739;
                color: white;
            }
            """,
        ):
            exportBarchart_Button = st.download_button("下載長條圖 \n\n Download the barchart", 
                                                       data = buf, file_name = 'Barchart', mime = 'png')

    with tab_intertopicDistMap:
        st.markdown("##### 主題間距圖 Intertopic Distance Map")
        if len(st.session_state.topics) < 5:
            st.warning("主題數量較少 主題間距圖無法適用 請調整訓練參數 以增加主題數量")
        else:
            top_n_topics = st.number_input('請輸入您希望呈現的主題數量 \n\n Choose the number of topics in the barchart', min_value = 1, max_value = len(st.session_state.topics) - 1, value = min(8, len(st.session_state.topics)), key = '22')
            fig = st.session_state.trained_model.visualize_topics(top_n_topics=top_n_topics, custom_labels=True)
            st.write(fig)
            buf = io.BytesIO()
            pio.write_image(fig, buf, format='png')
            buf.seek(0) 
            with stylable_container(
            key="downloadIntertopicDistMap_Button",
            css_styles="""
                button {
                    background-color: #157739;
                    color: white;
                }
                """,
            ):
                exportBarchart_Button = st.download_button("下載主題間距圖 \n\n Download the Intertopic Distance Map", 
                                                           data = buf, file_name = 'Intertopic Distance Map', mime = 'png')
            
    with tab_wordCloud:
        st.markdown("##### 詞雲庫 WordCloud")
        topic = st.number_input("請選擇您需要生成詞雲圖的主題編號 \n\n Choose your topic ID for the WordCloud", min_value = -1, max_value = len(st.session_state.topics) - 2, value = 0, key = '33')
        max_num_words = st.number_input('請選擇您需要呈現的最大詞匯數量 \n\n Choose you maximum number of words', min_value = min(3, st.session_state.top_n_words), max_value = st.session_state.top_n_words, value = st.session_state.top_n_words, key = '333')
        min_word_length = st.number_input('請選擇您需要呈現的最小詞匯長度 \n\n Choose you minimum length of words', min_value = 1, max_value = 10, value = 1, key = '3333')
        plt = visualize_wordcloud(st.session_state.trained_model, topic, max_num_words, min_word_length)
        st.pyplot(plt)
        buf = io.BytesIO()
        plt.savefig(buf, format = 'png')
        buf.seek(0)  # Rewind the buffer
        plt.close()  # Close the plot
        with stylable_container(
            key="downloadWordCloud_Button",
            css_styles="""
                button {
                    background-color: #157739;
                    color: white;
                }
                """,
            ):
                exportWordCloud_Button = st.download_button("下載詞雲圖 \n\n Download the WordCloud", 
                                                            data = buf, file_name = 'WordCloud', mime = 'png')
           
    with tab_similarityMatrix:
        st.markdown("##### 相似矩陣 Similarity Matrix")
        top_n_topics = st.number_input('請輸入您希望呈現的主題數量 \n\n Choose the number of topics in the barchart', min_value = 1, max_value = len(st.session_state.topics) - 1, value = min(8, len(st.session_state.topics)), key = '44')
        fig = st.session_state.trained_model.visualize_heatmap(top_n_topics=top_n_topics)
        st.write(fig)
        buf = io.BytesIO()
        pio.write_image(fig, buf, format='png')
        buf.seek(0) 
        with stylable_container(
            key="downloadSimilarMatrix_Button",
            css_styles="""
                button {
                    background-color: #157739;
                    color: white;
                }
                """,
        ):
            exportSimilarMatrix_Button = st.download_button("下載相似矩陣 \n\n Download the Similarity Matrix", 
                                                            data = buf, file_name = 'Similarity Matrix', mime = 'png')
        

    st.markdown("### 主題 Topic")
    st.markdown('###   ')
    display_topic(st.session_state.topics, st.session_state.trained_model, st.session_state.filtered_topic)
        
else:
    st.markdown("### 可視化 Visualization")

    tab_Barchart, tab_intertopicDistMap, tab_wordCloud, tab_similarityMatrix, tab_doc_topic_2d, tab_doc_topic_datamap, tab_doc_topic_heatmap = st.tabs([
            "長條圖 \n\n Barchart",
            "主題間距圖 \n\n Intertopic Distance Map", 
            "詞雲庫 \n\n WordCloud", 
            "相似矩陣 \n\n Similarity Matrix",
            "文件-主題關係圖(2D) \n\n Document-Topic 2D",
            "文件-主題數據映射 \n\n Document-Topic Datamap",
            "文件-主題熱圖 \n\n Document-Topic Heatmap"])

    with tab_Barchart:
        st.markdown("##### 長條圖 Barchart")
        top_n_topics = st.number_input('請輸入您希望呈現的主題數量 \n\n Choose the number of topics in the barchart', min_value = 1, max_value = len(st.session_state.topics) - 1, value = min(8, len(st.session_state.topics)-1), key = '11')
        top_n_words = st.number_input('請輸入您希望每個主題呈現的關鍵詞數量 \n\n Choose the number of words per topic in the barchart', min_value = 3, max_value=st.session_state.top_n_words, value = 5)
        fig = st.session_state.trained_model.visualize_barchart(top_n_topics=top_n_topics, n_words=top_n_words, custom_labels=True)
        st.write(fig)
        buf = io.BytesIO()
        pio.write_image(fig, buf, format='png')
        buf.seek(0)  # Rewind the buffer
        with stylable_container(
        key="downloadBarchart_Button",
        css_styles="""
            button {
                background-color: #157739;
                color: white;
            }
            """,
        ):
            exportBarchart_Button = st.download_button("下載長條圖 \n\n Download the barchart", 
                                                       data = buf, file_name = 'Barchart', mime = 'png')
        
        
        

    with tab_intertopicDistMap:
        st.markdown("##### 主題間距圖 Intertopic Distance Map")
        if len(st.session_state.topics) < 5:
            st.warning("主題數量較少 主題間距圖無法適用 請調整訓練參數 以增加主題數量")
        else:
            top_n_topics = st.number_input('請輸入您希望呈現的主題數量 \n\n Choose the number of topics in the barchart', min_value = 1, max_value = len(st.session_state.topics) - 1, value = min(8, len(st.session_state.topics)-1), key = '22')
            fig = st.session_state.trained_model.visualize_topics(top_n_topics=top_n_topics, custom_labels=True)
            st.write(fig)
            buf = io.BytesIO()
            pio.write_image(fig, buf, format='png')
            buf.seek(0) 
            with stylable_container(
            key="downloadIntertopicDistMap_Button",
            css_styles="""
                button {
                    background-color: #157739;
                    color: white;
                }
                """,
            ):
                exportBarchart_Button = st.download_button("下載主題間距圖 \n\n Download the Intertopic Distance Map", 
                                                           data = buf, file_name = 'Intertopic Distance Map', mime = 'png')
            


    with tab_wordCloud:
        st.markdown("##### 詞雲庫 WordCloud")
        topic = st.number_input("請選擇您需要生成詞雲圖的主題編號 \n\n Choose your topic ID for the WordCloud", min_value = -1, max_value = len(st.session_state.topics) - 2, value = 0, key = '33')
        max_num_words = st.number_input('請選擇您需要呈現的最大詞匯數量 \n\n Choose you maximum number of words', min_value = min(3, st.session_state.top_n_words), max_value = st.session_state.top_n_words, value = st.session_state.top_n_words, key = '333')
        min_word_length = st.number_input('請選擇您需要呈現的最小詞匯長度 \n\n Choose you minimum length of words', min_value = 1, max_value = 10, value = 1, key = '3333')
        plt = visualize_wordcloud(st.session_state.trained_model, topic, max_num_words, min_word_length)
        st.pyplot(plt)
        buf = io.BytesIO()
        plt.savefig(buf, format = 'png')
        buf.seek(0)  # Rewind the buffer
        plt.close()  # Close the plot
        with stylable_container(
            key="downloadWordCloud_Button",
            css_styles="""
                button {
                    background-color: #157739;
                    color: white;
                }
                """,
            ):
                exportWordCloud_Button = st.download_button("下載詞雲圖 \n\n Download the WordCloud", 
                                                            data = buf, file_name = 'WordCloud', mime = 'png')
           

    with tab_similarityMatrix:
        st.markdown("##### 相似矩陣 Similarity Matrix")
        top_n_topics = st.number_input('請輸入您希望呈現的主題數量 \n\n Choose the number of topics in the barchart', min_value = 1, max_value = len(st.session_state.topics) - 1, value = min(8, len(st.session_state.topics) - 1), key = '44')
        fig = st.session_state.trained_model.visualize_heatmap(top_n_topics=top_n_topics)
        st.write(fig)
        buf = io.BytesIO()
        pio.write_image(fig, buf, format='png')
        buf.seek(0) 
        with stylable_container(
            key="downloadSimilarMatrix_Button",
            css_styles="""
                button {
                    background-color: #157739;
                    color: white;
                }
                """,
        ):
            exportSimilarMatrix_Button = st.download_button("下載相似矩陣 \n\n Download the Similarity Matrix", 
                                                            data = buf, file_name = 'Similarity Matrix', mime = 'png')
        

    with tab_doc_topic_2d:
        st.markdown("##### 文件-主題關係圖(2D) Document-Topic 2D")
        doc = []
        for value in st.session_state.trainingDoc.values():
            doc.append(value['content'])
            #doc = list(st.session_state.trainingDoc.values()['content'])
        fig = st.session_state.trained_model.visualize_documents(doc)
        st.write(fig)
        buf = io.BytesIO()
        pio.write_image(fig, buf, format='png')
        buf.seek(0) 
        with stylable_container(
            key="downloadDocTopic2D_Button",
            css_styles="""
                button {
                    background-color: #157739;
                    color: white;
                }
                """,
        ):
            exportDocTopic2D_Button = st.download_button("下載文件-主題關係圖(2D) \n\n Download the Document-Topic 2D", 
                                                         data = buf, file_name = 'Document-Topic 2D', mime = 'png')
        
        
    with tab_doc_topic_datamap:
        st.markdown("##### 文件-主題數據映射 Document-Topic Datamap")
        doc = []
        for value in st.session_state.trainingDoc.values():
            doc.append(value['content'])
        sentence_model = SentenceTransformer('uer/sbert-base-chinese-nli')
        embeddings = sentence_model.encode(doc, show_progress_bar=False)
        #top_n_topics = st.number_input('請輸入您希望呈現的主題數量 \n\n Choose the number of topics in the barchart', min_value = 1, max_value = len(st.session_state.topics) - 1, value = min(8, len(st.session_state.topics)))
        fig = visualize_document_datamap(st.session_state.trained_model, doc, embeddings = embeddings)
        st.write(fig)
        buf = io.BytesIO()
        fig.savefig(buf, format = 'png')
        buf.seek(0)  # Rewind the buffer
        plt.close()  # Close the plot
        with stylable_container(
            key="downloadDocTopicDatamap_Button",
            css_styles="""
                button {
                    background-color: #157739;
                    color: white;
                }
                """,
            ):
                exportDocTopicDatamap_Button = st.download_button("下載文件-主題數據映射 \n\n Download the Document-Topic Datamap", 
                                                                  data = buf, file_name = 'Document_Topic Datamap', mime = 'png')


    with tab_doc_topic_heatmap:
        st.markdown("##### 文件-主題熱圖 Document-Topic Heatmap")
        docs = []
        doc_idx = []
        for key, doc in st.session_state.trainingDoc.items():
            doc_idx.append(key)
            docs.append(doc['content'])
        topic_distr, topic_token_distr = st.session_state.trained_model.approximate_distribution(docs, calculate_tokens=True)
        docs_per_tab = 20
        num_tabs = (len(docs) + docs_per_tab - 1) // docs_per_tab
        tabs = st.tabs([str(i + 1) for i in range(num_tabs)])
        for i, tab in enumerate(tabs):
            with tab:
                start_idx = i * docs_per_tab
                end_idx = min((i + 1) * docs_per_tab, len(docs))
                tab_topic_distr = topic_distr[start_idx:end_idx,:]
                tab_doc_idx = doc_idx[start_idx:end_idx]
                plt.figure(figsize=(10, 8))
                seaborn.heatmap(tab_topic_distr, cmap = seaborn.cubehelix_palette(gamma = .5, as_cmap=True), 
                                annot = tab_topic_distr, yticklabels = tab_doc_idx)
                st.pyplot(plt)
                buf = io.BytesIO()
                plt.savefig(buf, format='png')  # Save the figure to the buffer
                buf.seek(0) 
                plt.close()
                with stylable_container(
                    key="downloadHeatMap_Button",
                    css_styles="""
                    button {
                    background-color: #157739;
                    color: white;
                }
                """,
                ):
                    exportHeatMap_Button = st.download_button("下載文件-主題熱圖 \n\n Download the Document-Topic Heatmap", 
                                                                data = buf, file_name = 'Document-Topic Heatmap', mime = 'png', key = f'heatmap{i}')     
        

    col_topic, col_doc = st.columns(2)

    with col_topic:
        st.markdown("### 主題 Topic")
        st.markdown('###   ')
        #st.write(st.session_state.topics)
        #st.session_state.trained_model.visualize_topics()
        display_topic(st.session_state.topics, st.session_state.trained_model, st.session_state.filtered_topic)

    with col_doc:
        st.markdown("### 文本 Documents")
        display_document(st.session_state.originalDoc, st.session_state.trained_model, st.session_state.filtered_topic, st.session_state.topics)


        
###########################
# Output training results #
###########################

st.markdown("---")
st.subheader("• " + "留存您的發現 Save your Findings")

if not st.session_state.Exist_training_model:
    st.warning("當前無已訓練模型，請在側邊欄-參數設定完成模型訓練或上載 \n\n No trained model, please train or upload your model in the sidebar.")
else:    
    tab_outputModel, tab_outputData = st.columns(2)

    with tab_outputModel:
        tempdir = tempfile.mkdtemp()
        st.session_state.trained_model.save(tempdir, 
                                            serialization = 'safetensors', 
                                            save_embedding_model = 'uer/sbert-base-chinese-nli',
                                            save_ctfidf = True)
        model_archive = zip_model(tempdir)
        shutil.rmtree(tempdir)
        #model_zip = download_model(model = st.session_state.trained_model, save_ctfidf=True, save_embedding_model='uer/sbert-base-chinese-nli')
        with stylable_container(
            key="DownloadModel_Button",
            css_styles="""
                button {
                    background-color: #157739;
                    color: white;
                }
                """,
        ):
            exportModel_Button = st.download_button(label = "下載訓練完成的模型 \n\n Download trained model", 
                                                    data = model_archive.getvalue(),
                                                    file_name = 'Chinese_topic_model.zip',
                                                    mime = 'application/zip')
        if exportModel_Button:
            st.success("下載成功 \n\n Model is successfully exported")

    with tab_outputData:
        df_t = pd.DataFrame(st.session_state.raw_topics)
        csv_t = df_t.to_csv().encode("utf-8")
        with stylable_container(
            key="DownloadData_Button",
            css_styles="""
                button {
                    background-color: #157739;
                    color: white;
                }
            """,
        ):
            exportTraining_Button = st.download_button(label="下載訓練數據 \n\n Download training data", data=csv_t, file_name="訓練數據.csv")
        if exportTraining_Button:
            st.success("下載成功 \n\n Training data is successfully exported")

##############
# Prediction #
##############

st.markdown("---")
st.subheader("• " + "應用您的發現 Apply Existing Findings to New Data")

if not st.session_state.Exist_training_model:
    st.warning("當前無已訓練模型，請在側邊欄-參數設定完成模型訓練或上載 \n\n No trained model, please train or upload your model in the sidebar.")
else:    
    uploaded_files_p = st.file_uploader("上載您用於主題模型預測的文件 \n\n Upload your file(s) for topic model prediction", accept_multiple_files=True, type=["txt", "csv"])

    if uploaded_files_p:
        st.session_state.predictingDoc = load_files(uploaded_files_p)

    #predictCol, exportCol = st.columns(2)

    #with predictCol:
    with stylable_container(
            key="predicttDoc_Button",
                css_styles="""
                    button {
                        background-color: #f5cc4a;
                        color: #013365;
                    }
                    """,
        ):
            predictDoc_Button = st.button("使用當前模型預測 \n\n Predict by the model")
    if predictDoc_Button and uploaded_files_p:
        sentence_model = SentenceTransformer('uer/sbert-base-chinese-nli')
        #st.write(st.session_state.predictingDoc)
        predict_doc = []
        for p_doc in st.session_state.predictingDoc.values():
            predict_doc.append(p_doc['content'])
        embeddings = sentence_model.encode(predict_doc, show_progress_bar=True)
        st.session_state.predicted_topics, st.session_state.predicted_probs = st.session_state.trained_model.transform(process_files(st.session_state.predictingDoc), embeddings=embeddings)
        #st.write(st.session_state.predicted_topics)
        st.session_state.predicted_df = display_prediction_df(st.session_state.predicted_topics, st.session_state.predictingDoc)
        st.dataframe(st.session_state.predicted_df)
    #with exportCol:
        #df_p = pd.DataFrame(st.session_state.predicted_topics, )
        df = st.session_state.predicted_df
        csv_p = df.to_csv().encode("utf-8")
        with stylable_container(
            key="exportPrediction_Button",
                css_styles="""
                    button {
                        background-color: #157739;
                        color: white;
                    }
                    """,
        ):
            exportPrediction_Button = st.download_button(label="輸出預測結果", data=csv_p, file_name="預測結果.csv")

        if exportPrediction_Button:
            st.success("下載成功 \n\n Prediction data is successfully exported")
            
#########################
# Topic Model Over Time #
#########################

if st.session_state.Topic_over_time:
    st.markdown("---")
    st.subheader("• " + "追蹤主題隨時間的變化 Track Topic Changes Over Time")

    if  st.session_state.Topic_over_time != "Timestamp":
        st.markdown("### 生成時間戳 Generate Timestamp")
        cluster_K = st.number_input("根據下方散點圖，為K-means設定K值 \n\n Set K value for K-mean clustering. Choose appropriate value of K according to the scatter plot below", min_value=0, max_value=100, value=5)
        with stylable_container(
                key="GenerateTimestamp_Button",
                css_styles="""
                    button {
                        background-color: #013365;
                        color: white;
                    }
                    """,
            ):
            GenerateTimestamp_Button = st.button("生成時間戳 \n\n Generate Timestamp")
        if GenerateTimestamp_Button:
            st.session_state.training_doc_time = GenerateTimestamp(st.session_state.trainingDoc, cluster_K)
            topics_over_time = None
            #ALL_RUNs[CUR_RUN]['TOPIC_TIME_RES'] = None
            Doc_timestamp = {}
            for docID in st.session_state.training_doc_time:
                Doc_timestamp[docID] = st.session_state.training_doc_time[docID]['timestamp']
            st.session_state.Doc_timestamp = Doc_timestamp
            st.session_state.Timestamp_text = TimestampText(st.session_state.training_doc_time)
            st.success("時間戳生成完成 Finished generating timestamp.")

            fig = Display_Time(st.session_state.training_doc_time, st.session_state.Topic_over_time, st.session_state.Timestamp_text)
            if fig:
                st.plotly_chart(fig)
                st.markdown("### 訓練隨時間變化的主題模型 \n\n Train Topic Model Over Time")

    col1, col2 = st.columns(2)
    with col1:
        with stylable_container(
                key="TrainModelOverTime_Button",
                css_styles="""
                    button {
                        background-color: #f5cc4a;
                        color: #013365;
                    }
                    """,
            ):
            TrainModelOverTime_Button = st.button("訓練隨時間變化的主題模型 \n\n Train Model Over Time")
        if TrainModelOverTime_Button:
            st.session_state.topic_time = TrainModelOverTime(st.session_state.training_doc_time, st.session_state.trained_model, st.session_state.Timestamp_text)
            
             

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
            ExportModelData_time_Button = st.button("下載隨時間變化的主題模型 \n\n Export Model Data Over Time")
        if ExportModelData_time_Button:
            ExportTopicOverTimeData(st.session_state.topic_time, st.session_state.training_doc_time, st.session_state.Timestamp_text)
    if st.session_state.topic_time != {}:
        st.write(st.session_state.topic_time['Figs']['Variation of Topics Over Time'][''])
