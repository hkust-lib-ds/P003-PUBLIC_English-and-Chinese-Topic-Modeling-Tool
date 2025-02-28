import streamlit as st
from utils.helper import BoldDoc
import pandas as pd


def DisplayTrain_table(RUN, TRAIN_DOCs, CUR_TOPIC):
    displayDict = {}
    if not CUR_TOPIC: # display all topics
        topics = RUN['TOPICs'].keys()
    else: # display only the selected topic
        topics = [CUR_TOPIC]
    
    for topic in topics:
        label = ''
        if RUN['TOPICs'][topic]['LABEL']:
            label = f" | {RUN['TOPICs'][topic]['LABEL']}"
        repDoc = RUN['TOPICs'][topic]['RepDocs']
        belDoc = RUN['TOPICs'][topic]['BelDocs']
        for docID in repDoc.keys():
            displayDict[docID] = (topic+label, repDoc[docID], True, TRAIN_DOCs[docID]['content'])
        for docID in belDoc.keys():
            displayDict[docID] = (topic+label, belDoc[docID], False, TRAIN_DOCs[docID]['content'])

    if displayDict:
        displayDict = dict(sorted(displayDict.items()))
        df = pd.DataFrame(displayDict.values(), index=displayDict.keys(), columns=['Topic', 'Probability', 'isRepresentative', 'Content (Double click to view full text)'])
        st.dataframe(df, height=400)


def DisplayDoc(RUN, CUR_TOPIC, TRAIN_DOCs):
    if not CUR_TOPIC: # display all topics
        topics = RUN['TOPICs'].keys()
        docs_per_tab = len(topics)
    else: # display only the selected topic
        topics = [CUR_TOPIC]
        docs_per_tab = 5
    
    rep_DisStr = {}
    bel_DisStr = {}
    for topic in topics:
        label = ''
        if RUN['TOPICs'][topic]['LABEL']:
            label = f" | {RUN['TOPICs'][topic]['LABEL']}"
        Tcolor = RUN['TOPICs'][topic]['COLOR']
        repDoc = RUN['TOPICs'][topic]['RepDocs']
        belDoc = RUN['TOPICs'][topic]['BelDocs']
        Twords = list(RUN['TOPICs'][topic]['WORDs'].keys())
        for docID in repDoc.keys():
            rep_DisStr[docID] = f"<div style='background-color: {Tcolor}; padding: 10px; border-radius: 5px; height: 8em; overflow: auto;'>"\
                                f'<span style="font-weight: bold; font-size: 1.2em;">{docID}</span><br>'\
                                f'<span style="font-weight: bold; font-size: 1.2em;">{topic+label}: {repDoc[docID]:.2f}</span><br>'\
                                f"{BoldDoc(TRAIN_DOCs[docID]['content'], Twords)}</div>"

        for docID in belDoc.keys():
            bel_DisStr[docID] = f"<div style='background-color: {Tcolor}; padding: 10px; border-radius: 5px; height: 8em; overflow: auto;'>"\
                                f'<span style="font-weight: bold; font-size: 1.2em;">{docID}</span><br>'\
                                f'<span style="font-weight: bold; font-size: 1.2em;">{topic+label}: {belDoc[docID]:.2f}</span><br>'\
                                f"{BoldDoc(TRAIN_DOCs[docID]['content'], Twords)}</div>"


    if not CUR_TOPIC: # all topics
        combined_dict = {**rep_DisStr, **bel_DisStr}
        sorted_docs = dict(sorted(combined_dict.items()))
        tab_count = (len(sorted_docs) + docs_per_tab - 1) // docs_per_tab
        tabs = st.tabs([f"P.{i+1}" for i in range(tab_count)])
        for i, (docID, docStr) in enumerate(sorted_docs.items()):
            tab_index = i // docs_per_tab
            with tabs[tab_index]:
                st.markdown(docStr, unsafe_allow_html=True)
                st.write("")

    else: # one topic
        sorted_docs = dict(sorted(rep_DisStr.items()))
        # Create tabs with "Representative" as the first tab
        tab_count = (len(bel_DisStr) + docs_per_tab - 1) // docs_per_tab 
        tabs = st.tabs(["Representative"] + [f"P{i+1}" for i in range(tab_count)])

        # Add representative documents to the first tab
        with tabs[0]:
            for docID, docStr in rep_DisStr.items():
                st.markdown(docStr, unsafe_allow_html=True)
                st.write("")

        # Add other documents to the remaining tabs
        for i, (docID, docStr) in enumerate(bel_DisStr.items()):
            tab_index = i // docs_per_tab
            with tabs[tab_index + 1]:
                st.markdown(docStr, unsafe_allow_html=True)
                st.write("")

def DisplayDoc_nodata(RUN, CUR_TOPIC, TRAIN_DOCs):
    if not CUR_TOPIC: # display all topics
        topic_tabs = st.tabs([topic for topic in RUN['TOPICs'].keys()])
        for i, tab in enumerate(RUN['TOPICs'].keys()):
            with topic_tabs[i]:
                rep_DisStr = {}
                label = ''
                if RUN['TOPICs'][tab]['LABEL']:
                    label = f" | {RUN['TOPICs'][tab]['LABEL']}"
                Tcolor = RUN['TOPICs'][tab]['COLOR']
                repDoc = RUN['TOPICs'][tab]['RepDocs']
                Twords = list(RUN['TOPICs'][tab]['WORDs'].keys())
                for docID in repDoc.keys():
                    rep_DisStr[docID] = f"<div style='background-color: {Tcolor}; padding: 10px; border-radius: 5px; height: 8em; overflow: auto;'>"\
                                        f'<span style="font-weight: bold; font-size: 1.2em;">{docID}</span><br>'\
                                        f"{BoldDoc(TRAIN_DOCs[docID]['content'], Twords)}</div>"
                if rep_DisStr:
                    for docID, docStr in rep_DisStr.items():
                        st.markdown(docStr, unsafe_allow_html=True)
                        st.write("")
    else: # display only the selected topic
        tab = st.tabs(["Representative"])
        with tab[0]:
            rep_DisStr = {}
            label = ''
            if RUN['TOPICs'][CUR_TOPIC]['LABEL']:
                label = f" | {RUN['TOPICs'][CUR_TOPIC]['LABEL']}"
            Tcolor = RUN['TOPICs'][CUR_TOPIC]['COLOR']
            repDoc = RUN['TOPICs'][CUR_TOPIC]['RepDocs']
            Twords = list(RUN['TOPICs'][CUR_TOPIC]['WORDs'].keys())
            for docID in repDoc.keys():
                rep_DisStr[docID] = f"<div style='background-color: {Tcolor}; padding: 10px; border-radius: 5px; height: 8em; overflow: auto;'>"\
                                    f'<span style="font-weight: bold; font-size: 1.2em;">{docID}</span><br>'\
                                    f"{BoldDoc(TRAIN_DOCs[docID]['content'], Twords)}</div>"
            if rep_DisStr:
                for docID, docStr in rep_DisStr.items():
                    st.markdown(docStr, unsafe_allow_html=True)
                    st.write("")
            
        