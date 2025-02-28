import streamlit as st
import pandas as pd
from utils.helper import BoldDoc
import matplotlib.pyplot as plt
import seaborn as sns
from functions.Sidebar_NewRun import DocTopic_heatmap_single, DocTopic_heatmap


def PredictDoc(RUN, PREDICT_DOCs):

    if not PREDICT_DOCs:
        st.warning("No documents to predict.")
        return PREDICT_DOCs, {}
    
    model = RUN["MODEL"]
    docs = [PREDICT_DOCs[key]['content'] for key in PREDICT_DOCs.keys()]
    approMatix = model.approximate_distribution(docs)[0].tolist()
    RES = {}
    RES['_ApproMatrix_'] = approMatix
    temp_run = {'TOPICs': RUN['TOPICs'], 'ApproDistribution': approMatix}
    RES['_ApproMatrix_all_'] = DocTopic_heatmap(temp_run, PREDICT_DOCs, title = "Predicted Document-Topic Heatmap")
    RES['_ApproMatrix_topic_'] = {}
    
    # process the prediction results
    topics, probs = model.transform(docs)
    topics = [f"Topic {int(i)}" for i in topics]
    all_topics = RUN['TOPICs'].keys()
    for TOP in all_topics:
        this_DOC = {}
        this_appro = []
        for i, docID in enumerate(PREDICT_DOCs.keys()):
            if (topics[i] == TOP):
                this_appro.append(approMatix[i])
                PREDICT_DOCs[docID]['TopicProb'] = (topics[i], probs[i])
                this_DOC[docID] = PREDICT_DOCs[docID]
        RES['_ApproMatrix_topic_'][TOP] = DocTopic_heatmap(temp_run, this_DOC, title = f"Predicted Document-Topic Heatmap for {TOP}")
    
    return PREDICT_DOCs, RES

def DisplayPredict_table(RUN, PREDICT_DOCs, CUR_TOPIC):
    if not CUR_TOPIC: # display all topics
        topics = RUN['TOPICs'].keys()   
    else: # display only the selected topic
        topics = [CUR_TOPIC]
    docs_per_tab = 5

    displayDict = {}
    for docID in PREDICT_DOCs.keys():
        content = PREDICT_DOCs[docID]['content']
        try:
            topic, prob = PREDICT_DOCs[docID]['TopicProb']
            if topic not in topics:
                continue
            if RUN['TOPICs'][topic]['LABEL']:
                displayDict[docID] = (f"{topic} | {RUN['TOPICs'][topic]['LABEL']}", prob, content)
            else:
                displayDict[docID] = (topic, prob, content)
        except:
            pass
    if displayDict:
        df = pd.DataFrame(displayDict.values(), index=displayDict.keys(), columns=['Topic', 'Probability', 'Content (double click to view full text)'])
        st.dataframe(df, height=400)
        
def DisplayPredict(RUN, CUR_TOPIC, PREDICT_DOCs, PREDICT_RES):

    tab_table, tab_heatmap = st.tabs(["Summary & Details", "Heatmap"])

    with tab_table:

        if not CUR_TOPIC: # display all topics
            topics = RUN['TOPICs'].keys()
        else: # display only the selected topic
            topics = [CUR_TOPIC]
        docs_per_tab = 5

        tempDict = {} # { Topic: [docIDs] }
        for docID in PREDICT_DOCs.keys():
            try:
                topic, _ = PREDICT_DOCs[docID]['TopicProb']
                if topic not in tempDict and topic in topics:
                    tempDict[topic] = []
                tempDict[topic].append(docID)
            except:
                pass

        DisStr = {}
        for topic in topics:
            label = ''
            if RUN['TOPICs'][topic]['LABEL']:
                label = f" | {RUN['TOPICs'][topic]['LABEL']}"
            Tcolor = RUN['TOPICs'][topic]['COLOR']
            predDocs = tempDict.get(topic, [])
            Twords = list(RUN['TOPICs'][topic]['WORDs'].keys())
            for docID in predDocs:
                DisStr[docID] = f"<div style='background-color: {Tcolor}; padding: 10px; border-radius: 5px; height: 8em; overflow: auto;'>"\
                                    f'<span style="font-weight: bold; font-size: 1.2em;">{docID}</span><br>'\
                                    f'<span style="font-weight: bold; font-size: 1.2em;">{topic+label}: {PREDICT_DOCs[docID]['TopicProb'][1]:.2f}</span><br>'\
                                    f"{BoldDoc(PREDICT_DOCs[docID]['content'], Twords)}</div>"

        sorted_docs = dict(sorted(DisStr.items()))
        tab_count = (len(sorted_docs) + docs_per_tab - 1) // docs_per_tab
        tabs = st.tabs(["Summary"] + [f"P.{i+1}" for i in range(tab_count)])
        
        with tabs[0]:
            DisplayPredict_table(RUN, PREDICT_DOCs, CUR_TOPIC)
                                
        for i, (docID, docStr) in enumerate(sorted_docs.items()):
            tab_index = i // docs_per_tab
            with tabs[tab_index + 1]:
                st.markdown(docStr, unsafe_allow_html=True)
                st.write("")

    with tab_heatmap:
        if not CUR_TOPIC:
            figs = PREDICT_RES['_ApproMatrix_all_']
            tabs = st.tabs([f"P.{i+1}" for i in range(len(figs))])
            for i, fig in enumerate(figs):
                with tabs[i]:
                    st.pyplot(fig)
        else:
            figs = PREDICT_RES['_ApproMatrix_topic_'][CUR_TOPIC]
            if not figs:
                st.warning(f"No documents are predicted to be in topic {CUR_TOPIC}.")
            else:
                tabs = st.tabs([f"P.{i+1}" for i in range(len(figs))])
                for i, fig in enumerate(figs):
                    with tabs[i]:
                        st.pyplot(fig)

