import streamlit as st
import pandas as pd
from utils.topic import *
import string

def display_topic(topics, trained_model, filtered_topic):
    #st.write(topics)
    if not filtered_topic:
        for topic in topics:
            label = topics[topic]['LABEL']
            words = topics[topic]['WORDs'].keys()
            color = topics[topic]['COLOR']
            if label:
                st.markdown(f"<div style='background-color: {color}; padding: 10px; border-radius: 5px; height: 8em; overflow: auto; color: white;'>"
                    f"<strong>{topic}: {label}</strong><br>"
                    f"{', '.join(words)}</div>", unsafe_allow_html=True)
                st.write("")
            else:
                st.markdown(f"<div style='background-color: {color}; padding: 10px; border-radius: 5px; height: 8em; overflow: auto; color: white;'>"
                    f"<strong>{topic}</strong><br>"
                    f"{', '.join(words)}</div>", unsafe_allow_html=True)
                st.write("")
    else:
        label = topics[filtered_topic]['LABEL']
        words = topics[filtered_topic]['WORDs'].keys()
        color = topics[filtered_topic]['COLOR']
        if label:
            st.markdown(f"<div style='background-color: {color}; padding: 10px; border-radius: 5px; height: 8em; overflow: auto; color: white;'>"
                    f"<strong>{filtered_topic}: {label}</strong><br>"
                    f"{', '.join(words)}</div>", unsafe_allow_html=True)
            st.write("")
        else:
            st.markdown(f"<div style='background-color: {color}; padding: 10px; border-radius: 5px; height: 8em; overflow: auto; color: white;'>"
                    f"<strong>{filtered_topic}</strong><br>"
                    f"{', '.join(words)}</div>", unsafe_allow_html=True)
            st.write("")    


def find_all_occurrences(sentence, word):
    start = 0
    positions = []
    while True:
        start = sentence.find(word, start)
        if start == -1:  # No more occurrences found
            break
        positions.append(start)
        start += len(word)  # Move past the last found word to search for the next occurrence
    return positions


def bold_doc(content, words):
    all_positions = {}
    for word in words:
        #st.write(word)
        positions = find_all_occurrences(content, word)
        if positions == []:
            continue
        else:
            for position in positions:
                all_positions[position] = word
    result = ""
    i = 0
    while i < len(content):
        if i in all_positions.keys():
            result += f'<span style="font-weight: bold; font-size: 1.2em;">{all_positions[i]}</span>'
            i += len(all_positions[i])
        else:
            result += content[i]
            i += 1
    return result
    

def display_document(original_document, trained_model, filtered_topic, topic_list):
    model = trained_model

    docs = [original_document[key]['content'] for key in original_document.keys()]

    Doc_topic = model.get_document_info(docs)
    #st.write(Doc_topic)
    docs_per_tab = len(topic_list)
    DisplayDoc_list = [ (key, original_document[key]['content']) for key in original_document.keys()]
    doc_DocID = {doc: DocID for DocID, doc in DisplayDoc_list}

    if not filtered_topic: # Display all topics
        num_tabs = (len(DisplayDoc_list) + docs_per_tab - 1) // docs_per_tab
        tabs = st.tabs([str(i + 1) for i in range(num_tabs)])
        for i, tab in enumerate(tabs):
            with tab:
                start_idx = i * docs_per_tab
                end_idx = min((i + 1) * docs_per_tab, len(DisplayDoc_list))
                for docID, doc in DisplayDoc_list[start_idx:end_idx]:
                    row = Doc_topic[Doc_topic['Document'] == doc]
                    asg_topic = row['Topic'].iloc[0]
                    color = topic_list[f'Topic {asg_topic}']['COLOR']
                    rep_words = row['Representation'].iloc[0]
                    doc = bold_doc(doc, rep_words)
                    if 'Probability' in Doc_topic:
                        prob = row['Probability'].iloc[0]
                        st.markdown(f"<div style='background-color: {color}; padding: 10px; border-radius: 5px; height: 8em; overflow: auto; color: white'>"
                                f"<strong>{docID}</strong><br>"
                                f"<strong>Topic {asg_topic}: {prob}</strong><br>"
                                f"{doc}</div>", unsafe_allow_html=True)
                    else:
                        st.markdown(f"<div style='background-color: {color}; padding: 10px; border-radius: 5px; height: 8em; overflow: auto; color: white'>"
                                f"<strong>{docID}</strong><br>"
                                f"<strong>Topic {asg_topic}</strong><br>"
                                f"{doc}</div>", unsafe_allow_html=True)
                    st.write("")
    
    else: # Display only the selected topic
        Doc = Doc_topic[Doc_topic['Topic'] == int(filtered_topic.split(' ')[1])]
        docDisplayList = Doc['Document']
        num_tabs = (len(Doc) + docs_per_tab - 1) // docs_per_tab
        tabs = st.tabs([str(i + 1) for i in range(num_tabs)])
        for i, tab in enumerate(tabs):
            with tab:
                start_idx = i * docs_per_tab
                end_idx = min((i + 1) * docs_per_tab, len(Doc))
                for doc in docDisplayList[start_idx: end_idx]:
                    DocID = doc_DocID[doc]
                    row = Doc_topic[Doc_topic['Document'] == doc]
                    asg_topic = row['Topic'].iloc[0]
                    color = topic_list[f'Topic {asg_topic}']['COLOR']
                    rep_words = row['Representation'].iloc[0]
                    doc = bold_doc(doc, rep_words)
                    if 'Probability' in Doc_topic:
                        prob = row['Probability'].iloc[0]
                        st.markdown(f"<div style='background-color: {color}; padding: 10px; border-radius: 5px; height: 8em; overflow: auto; color: white'>"
                                f"<strong>{DocID}</strong><br>"
                                f"<strong>{prob}</strong><br>"
                                f"{doc}</div>", unsafe_allow_html=True)
                    else:
                        st.markdown(f"<div style='background-color: {color}; padding: 10px; border-radius: 5px; height: 8em; overflow: auto; color: white'>"
                                f"<strong>{DocID}</strong><br>"
                                f"{doc}</div>", unsafe_allow_html=True)
                    st.write("")


def display_prediction_df(predicting_topics, predictingDoc):
    topic_doc = {}
    #st.write(predicting_topics)
    #st.write(predictingDoc)
    i = 0
    processed_predictingDoc = {}
    for key in predictingDoc.keys():
        processed_predictingDoc[i] = (key, predictingDoc[key]['content'])
        i+=1
    for i in range(len(predicting_topics)):
        #print(processed_predictingDoc[i])
        topic_doc[processed_predictingDoc[i][0]] = (processed_predictingDoc[i][1], predicting_topics[i])
    df = pd.DataFrame.from_dict(topic_doc, orient='index', columns=['predictingDoc','predicted_topic'])
    return df
