import streamlit as st
from functions.Sidebar_NewRun import ModelToRun, DocTopic_heatmap

def LabelTopics(RUN, TRAIN_DOCs, topic_label):
    model = RUN['MODEL']
    TOPICS = RUN['TOPICs']
    docs = [TRAIN_DOCs[key]['content'] for key in TRAIN_DOCs.keys()]
    topic_label = topic_label.split(',')
    # topic_label = [x.strip() for x in topic_label]
    failed = []
    labelToSet = {}
    for item in topic_label:
        try:
            topic, label = item.split(':')
            topic = topic.strip()
            label = label.strip()
            TOPICS[topic]['LABEL'] = label
            labelToSet[int(topic.split(" ")[1])] = f"{topic} | {label}"
        except:
            failed.append(item)
            continue
    RUN['TOPICs'] = TOPICS
    if labelToSet:
        all_topic = model.get_topics()
        model.set_topic_labels(labelToSet)
        RUN['FIGs'] = { "Document-Topic Heatmap":DocTopic_heatmap(RUN, TRAIN_DOCs),     # Gofigs: {fig_name: Gofig}
                        'Topic-Topic Heatmap':model.visualize_heatmap(custom_labels=True, title="Topic-Topic Similarity Matrix"),  # Gofigs: {fig_name: Gofig}
                        'Topic 2D Similarity':model.visualize_topics(custom_labels=True), 
                        'Topic Bar Chart': model.visualize_barchart(list(all_topic.keys()), top_n_topics=len(all_topic), n_words=len(all_topic[list(all_topic.keys())[0]]), custom_labels=True),
                        'Topic WordCloud': RUN['FIGs']['Topic WordCloud'],
                        'Document 2D Clusters':model.visualize_documents(docs, custom_labels=True, title="Document 2D Clusters"),
                        # 'Document-Datamap':model.visualize_document_datamap(docs, custom_labels=True),
                        }
        
        # update each topic-doc heatmap and barchart
        for key in all_topic.keys():
            topic_approMatix = []
            topic_TRAINING_DOCs = {}
            for docID, approdata in zip(list(TRAIN_DOCs.keys()), RUN['ApproDistribution']):
                if docID in RUN['TOPICs'][f'Topic {key}']['RepDocs'] or docID in RUN['TOPICs'][f'Topic {key}']['BelDocs']:
                    topic_approMatix.append(approdata)
                    topic_TRAINING_DOCs[docID] = TRAIN_DOCs[docID]
            temp_RUN = {
                        'TOPICs': RUN['TOPICs'],
                        'ApproDistribution': topic_approMatix,
                        }
            RUN['TOPICs'][f'Topic {key}']['Figs']['Document-Topic Heatmap'] = DocTopic_heatmap(temp_RUN, topic_TRAINING_DOCs, title = f"Document-Topic Heatmap (Topic {key})")
            TOPICS[f'Topic {key}']['Figs']['Bar Chart for Topic Word Score'] = model.visualize_barchart([key], top_n_topics=1, n_words=len(model.get_topics()[int(topic.split(" ")[1])]), custom_labels=True)
            

    if failed:
        st.warning(f"Failed to set label for {', '.join(failed)}")
        
    return RUN


def ReduceTopic(RUN, TRAIN_DOCs, ReduceToNum):
    model = RUN['MODEL']
    docs = [TRAIN_DOCs[key]['content'] for key in TRAIN_DOCs.keys()]
    if model:
        try: 
            model.reduce_topics(docs, nr_topics=ReduceToNum)
            RUN = ModelToRun(model, TRAIN_DOCs)
        except:
            st.warning("Failed to reduce topics")
            return RUN
        st.success(f"Successfully reduced to {ReduceToNum} topics")
    return RUN
    

def MergeTopic(RUN, TRAIN_DOCs, MergeTopics):
    model = RUN['MODEL']
    docs = [TRAIN_DOCs[key]['content'] for key in TRAIN_DOCs.keys()]
    if model:
        try: 
            model.merge_topics(docs, topics_to_merge=MergeTopics)
            RUN = ModelToRun(model, TRAIN_DOCs)
        except:
            st.warning("Failed to merge topics")
            return RUN
        st.success(f"Successfully merged topics")
    return RUN