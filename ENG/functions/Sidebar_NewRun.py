from bertopic import BERTopic
from sklearn.feature_extraction.text import CountVectorizer
import random
from utils.ConstValue import ALL_Topic_COLORS
import streamlit as st
from wordcloud import WordCloud
from utils.helper import GetRandomColor
import seaborn as sns
import matplotlib.pyplot as plt

########################################################################################
# FIG functions
def DocTopic_heatmap_single(RUN, TRAIN_DOCs, title = "Document-Topic Heatmap"):
    # Create labels for topics
    if TRAIN_DOCs == {}:
        return None
    no_outliers = {key: value for key, value in RUN['TOPICs'].items() if key != 'Topic -1'}
    topic_labels = [key if not RUN['TOPICs'][key]['LABEL'] else f"{key} | {RUN['TOPICs'][key]['LABEL']}" for key in no_outliers.keys()]
    # create labels fpr documents
    doc_labels = [key for key in TRAIN_DOCs.keys()]
    # Create distance matrix
    distance_matrix = RUN['ApproDistribution']
    # Create heatmap with sns
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(distance_matrix, ax=ax, cmap=sns.cubehelix_palette(gamma = .5, as_cmap=True), xticklabels=topic_labels, yticklabels=doc_labels, annot=distance_matrix)
    ax.set_title(title)
    
    return fig

def DocTopic_heatmap(RUN, TRAIN_DOCs, title = "Document-Topic Heatmap", doc_split_size = 20):
    # for each split size create a heatmap
    heatmaps = []
    split_batch_num = len(TRAIN_DOCs)//doc_split_size + 1
    for i in range(split_batch_num):
        this_RUN = {'TOPICs': RUN['TOPICs'],
                    'ApproDistribution': RUN['ApproDistribution'][i*doc_split_size:min((i+1)*doc_split_size, len(TRAIN_DOCs))]
                    }
        keys = list(TRAIN_DOCs.keys())[i*doc_split_size:min((i+1)*doc_split_size, len(TRAIN_DOCs))]
        this_TRAIN_DOCs = {key: value for key, value in TRAIN_DOCs.items() if key in keys}
        res = DocTopic_heatmap_single(this_RUN, this_TRAIN_DOCs, title = title)
        if res:
            heatmaps.append(res)
    return heatmaps

def draw_wordCloud(TOPIC):
    wordcloud = WordCloud(width=400, height=400, 
                          background_color='white', 
                          stopwords=None, 
                          min_font_size=10).generate_from_frequencies(TOPIC['WORDs'])
    return wordcloud


def draw_wordCloud_all(all_topic, RUN_TOPICs):
    # normalizedfactors = freq of topic
    norm_factors = {}
    for topic in RUN_TOPICs:
        norm_factors[topic] = len(RUN_TOPICs[topic]['RepDocs'])+len(RUN_TOPICs[topic]['BelDocs'])

    sum_norm = sum(norm_factors.values())
    norm_factors = {key: value/sum_norm for key, value in norm_factors.items()}

    all_score = {}
    for topic in all_topic:
        for word, weight in all_topic[topic]:
            if word in all_score:
                all_score[word] += weight*norm_factors[f"Topic {topic}"]
            else:
                all_score[word] = weight*norm_factors[f"Topic {topic}"]
    wordcloud = WordCloud(width=400, height=400, 
                          background_color='white', 
                          stopwords=None, 
                          min_font_size=10).generate_from_frequencies(all_score)
    return wordcloud

########################################################################################
# RUN management functions

def UploadStopWords(NewStopWords):
    NewStopWords = NewStopWords.split(',')
    NewStopWords = [x.strip() for x in NewStopWords]
    NewStopWords = list(set(NewStopWords))
    return NewStopWords

def TrainNewModel(TRAIN_DOCs, top_n_words = 10, n_gram_range = (1, 1), min_topic_size = 10, nr_topics = 'No Reduction', embedding_model = 'DEFAULT: all-MiniLM-L6-v2', stop_words = None):
    docs = [TRAIN_DOCs[key]['content'] for key in TRAIN_DOCs.keys()]
    if embedding_model == 'DEFAULT: all-MiniLM-L6-v2':
        embedding_model = 'all-MiniLM-L6-v2'
    if nr_topics == 'No Reduction':
        nr_topics = None
    elif nr_topics == 'Automatic Reduction':
        nr_topics = 'auto'
    vectorizer_model = CountVectorizer(ngram_range=n_gram_range, stop_words="english")
    model = BERTopic(language="english", vectorizer_model=vectorizer_model, embedding_model=embedding_model, top_n_words=top_n_words, min_topic_size=min_topic_size, nr_topics=nr_topics)
    model.fit(docs)
    return model

"""
TO TRAIN : fit_transform
TO PREDICT : transform


top_n_words:    10
                number of words per topic to extract, usually <= 10
n_gram_range:   1 - [1,3]
                n-gram range for the CountVectorizer
min_topic_size: 10
nr_topics:     None or 'auto' or int == reduce_topics
embedding_model: list avaliabel: https://www.sbert.net/docs/pretrained_models.html

###
stop_words: None or 'english' or list of stop words
stop might not be necessary if no (too few documents or too short documents)
default does not have parameter for stop words
need:
from bertopic import BERTopic
from sklearn.feature_extraction.text import CountVectorizer
vectorizer_model = CountVectorizer(ngram_range=(1, 2), stop_words="english")
topic_model = BERTopic(vectorizer_model=vectorizer_model)


Advanced Parameters:
seed_topic_list
zeroshot_topic_list
zeroshot_min_similarity

Parameters not sutable for customization:
umap_model: dimenstionality reduction algorithm with .fit and .transform functions
hdbscan_model: clustering algorithm with .fit and .predict functions and .labels_ variable
vectorizer_model: custom CountVectorizer
ctfidf_model: custom ClassTfidfTransformer
representation_model: fine-tunes the topic representations calculated through c-TF-IDF
"""

def ModelToRun(model, TRAIN_DOCs):
    docs = [TRAIN_DOCs[key]['content'] for key in TRAIN_DOCs.keys()]
    all_topic = model.get_topics()

    default_label = [f'Topic {i}' if i != -1 else 'Topic -1 | OUTLIER' for i in all_topic.keys()]
    model.set_topic_labels(default_label)

    if len(all_topic) <= 5:
        RUN = {'MODEL': model,  # BERTopic model
            'TOPICs': {},  # {Topic: {LABEL, COLOR, WORDs, RepDocs, BelDocs, Figs}}
            'FIGs': {},   # no fig if too small number of topics
            'ApproDistribution': model.approximate_distribution(docs)[0], # {DocID: {Topic: Probability}}
            'Doc_timestamp': None
            }
        st.warning("Too few topics or documnets, may encounter error when displaying.")
    
    else:
        RUN = {'MODEL': model,  # BERTopic model
            'TOPICs': {}, # {Topic: {LABEL, COLOR, WORDs, RepDocs, BelDocs, Figs}}
            'FIGs': {"Document-Topic Heatmap":None, # LATER: DocTopic_heatmap(RUN),     # Gofigs: {fig_name: [Gofig]}
                    'Topic-Topic Heatmap':model.visualize_heatmap(custom_labels=True, title="Topic-Topic Similarity Matrix"),  # Gofigs: {fig_name: Gofig}
                    'Topic 2D Similarity':model.visualize_topics(custom_labels=True), 
                    'Topic Word Score': model.visualize_barchart(list(all_topic.keys()), top_n_topics=len(all_topic), n_words=len(all_topic[list(all_topic.keys())[0]]), custom_labels=True),
                    'Topic WordCloud': None, # LATER: draw_wordCloud_all(all_topic),
                    'Document 2D Clusters':model.visualize_documents(docs, custom_labels=True, title="Document 2D Clusters"),
                    'Table Summary for Training Documents': True,
                    # 'Document-Datamap':model.visualize_document_datamap(docs, custom_labels=True),
                    }, 
            'ApproDistribution':model.approximate_distribution(docs)[0].tolist(), # {DocID: {Topic: Probability}}
            'Doc_timestamp': None,
            'TOPIC_TIME_RES': None                          
            }
    
    Doc_stamp = {}
    for docID in TRAIN_DOCs:
        try:
            Doc_stamp[docID] = TRAIN_DOCs[docID]['timestamp']
        except:
            pass # no topic over time
    if Doc_stamp != {}:
        RUN['Doc_timestamp'] = Doc_stamp
    
    used_colors = []
    document_info = model.get_document_info(docs)

    # original result start from -1 , -1 means outlier
    for key in all_topic.keys():
        RUN['TOPICs'][f'Topic {key}'] = {'LABEL': 'OUTLIER' if key == -1 else None,    # str                        # str
                                         'COLOR': GetRandomColor(used_colors, ALL_Topic_COLORS), # HTML color code
                                         'WORDs': {}, # {word: weight}
                                         'RepDocs':{}, # {DocID: Probability}
                                         'BelDocs':{}, # {DocID: Probability}
                                         'Figs':{} # {fig_name: Gofig}
                                         }
        
        words = all_topic[key] 
        RUN['TOPICs'][f'Topic {key}']['WORDs'] = {word: weight for word, weight in words}
        RUN['TOPICs'][f'Topic {key}']['Figs'] = {
                                                'Document-Topic Heatmap': None, # LATER: DocTopic_heatmap(RUN, TRAIN_DOCs, title = f"Document-Topic Heatmap (Topic {key})"), # [Gofigs]
                                                'WordCloud':draw_wordCloud(RUN['TOPICs'][f'Topic {key}']),
                                                'Bar Chart for Topic Word Score': model.visualize_barchart([key], top_n_topics=1, n_words=len(all_topic[list(all_topic.keys())[0]]), custom_labels=True),
                                                'Table Summary for Words and Scores': True,
                                                'Table Summary for Training Documents': True,
                                                 }

        used_colors.append(RUN['TOPICs'][f'Topic {key}']['COLOR'])
        
        for row in document_info.iterrows():
            doc = row[1]['Document']
            DocID = [key for key, value in TRAIN_DOCs.items() if value['content'] == doc][0]
            if row[1]['Topic'] == key:
                if row[1]['Representative_document']:
                    RUN['TOPICs'][f'Topic {key}']['RepDocs'][DocID] = row[1]['Probability']
                else:
                    RUN['TOPICs'][f'Topic {key}']['BelDocs'][DocID] = row[1]['Probability']

    # heatmap for each topic
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

    RUN['FIGs']['Topic WordCloud'] = draw_wordCloud_all(all_topic, RUN['TOPICs'])
    RUN['FIGs']['Document-Topic Heatmap'] = DocTopic_heatmap(RUN, TRAIN_DOCs, title = "Document-Topic Heatmap All")
    return RUN


def LabelTopics_nodata(RUN, topic_label):
    model = RUN['MODEL']
    TOPICS = RUN['TOPICs']
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
            # update figs
            labelToSet[int(topic.split(" ")[1])] = f"{topic} | {label}"
        except:
            failed.append(item)
            continue
    RUN['TOPICs'] = TOPICS
    if labelToSet:
        all_topic = model.get_topics()
        model.set_topic_labels(labelToSet)
        RUN['FIGs'] = {
                        'Topic-Topic Heatmap':model.visualize_heatmap(custom_labels=True, title="Topic-Topic Similarity Matrix"),  # Gofigs: {fig_name: Gofig}
                        'Topic 2D Similarity':model.visualize_topics(custom_labels=True), 
                        'Topic Word Score': model.visualize_barchart(list(all_topic.keys()), top_n_topics=len(all_topic), n_words=len(all_topic[list(all_topic.keys())[0]]), custom_labels=True),
                        }
        
        # update each barchart
        for key in all_topic.keys():
            TOPICS[f'Topic {key}']['Figs']['Bar Chart for Topic Word Score'] = model.visualize_barchart([key], top_n_topics=1, n_words=len(all_topic[list(all_topic.keys())[0]]), custom_labels=True)

    failed = [ x for x in failed if x]
    if failed:
        st.warning(f"Failed to set label for {', '.join(failed)}")
        
    return RUN

def ModelToRun_nodata(model):
    TRAINING_DOCs = {}
    all_topic = model.get_topics()
    default_label = [f'Topic {i}' if i != -1 else 'Topic -1 | OUTLIER' for i in all_topic.keys()]
    # get back topic_label 
    original_labels = {topic: label for topic, label in zip(sorted(set(model.topics_)), model.custom_labels_)}
    original_labels = {value.split(" | ")[0]: value.split(" | ")[1] if len(value.split(" | ")) == 2 else None for key, value in original_labels.items()}
    original_topic_label_text = ''
    for key, value in original_labels.items():
        if value:
            original_topic_label_text += f"{key}: {value},"
    if original_topic_label_text.endswith(','):
        original_topic_label_text = original_topic_label_text[:-1]
    model.set_topic_labels(default_label)
    RUN = {'MODEL': model,  # BERTopic model
            'TOPICs': {}, # {Topic: {LABEL, COLOR, WORDs, RepDocs, BelDocs, Figs}}
            'FIGs': {   
                    'Topic-Topic Heatmap':model.visualize_heatmap(custom_labels=True, title="Topic-Topic Similarity Matrix"),  # Gofigs: {fig_name: Gofig}
                    'Topic 2D Similarity':model.visualize_topics(custom_labels=True), 
                    'Topic Word Score': model.visualize_barchart(list(all_topic.keys()), top_n_topics=len(all_topic), n_words=len(all_topic[list(all_topic.keys())[0]]), custom_labels=True),
                    }, 
            'ApproDistribution':None, # {DocID: {Topic: Probability}}
            'Doc_timestamp': None                            
            }
    used_colors = []

    for key in all_topic.keys():
        RUN['TOPICs'][f'Topic {key}'] = {'LABEL': 'OUTLIER' if key == -1 else None,    # str                        # str
                                         'COLOR': GetRandomColor(used_colors, ALL_Topic_COLORS), # HTML color code
                                         'WORDs': {}, # {word: weight}
                                         'RepDocs':{}, # {DocID: Probability}
                                         'BelDocs':{}, # {DocID: Probability}
                                         'Figs':{} # {fig_name: Gofig}
                                         }
        
        words = all_topic[key] 
        RUN['TOPICs'][f'Topic {key}']['WORDs'] = {word: weight for word, weight in words}
        RUN['TOPICs'][f'Topic {key}']['Figs'] = {
                                                 'WordCloud':draw_wordCloud(RUN['TOPICs'][f'Topic {key}']),
                                                 'Bar Chart for Topic Word Score': model.visualize_barchart([key], top_n_topics=1, n_words=len(all_topic[list(all_topic.keys())[0]]), custom_labels=True),
                                                 'Table Summary for Words and Scores': True,
                                                 }

        used_colors.append(RUN['TOPICs'][f'Topic {key}']['COLOR'])

        repre_doc = model.get_representative_docs(key)
        digit = len(str(len(repre_doc)))
        for i, doc in enumerate(repre_doc):
            docID = f"Topic{key}_Repre{i+1:0{digit}d}"
            TRAINING_DOCs[docID] = {'content': doc}
            RUN['TOPICs'][f'Topic {key}']['RepDocs'][docID] = None

    # reset original topic label
    RUN = LabelTopics_nodata(RUN, original_topic_label_text)

    return RUN, TRAINING_DOCs


"""
Arguments:
    language: The main language used in your documents. The default sentence-transformers
                model for "english" is `all-MiniLM-L6-v2`. For a full overview of
                supported languages see bertopic.backend.languages. Select
                "multilingual" to load in the `paraphrase-multilingual-MiniLM-L12-v2`
                sentence-transformers model that supports 50+ languages.
                NOTE: This is not used if `embedding_model` is used.
    top_n_words: The number of words per topic to extract. Setting this
                    too high can negatively impact topic embeddings as topics
                    are typically best represented by at most 10 words.
    n_gram_range: The n-gram range for the CountVectorizer.
                    Advised to keep high values between 1 and 3.
                    More would likely lead to memory issues.
                    NOTE: This param will not be used if you pass in your own
                    CountVectorizer.
    min_topic_size: The minimum size of the topic. Increasing this value will lead
                    to a lower number of clusters/topics and vice versa.
                    It is the same parameter as `min_cluster_size` in HDBSCAN.
                    NOTE: This param will not be used if you are using `hdbscan_model`.
    nr_topics: Specifying the number of topics will reduce the initial
                number of topics to the value specified. This reduction can take
                a while as each reduction in topics (-1) activates a c-TF-IDF
                calculation. If this is set to None, no reduction is applied. Use
                "auto" to automatically reduce topics using HDBSCAN.
                NOTE: Controlling the number of topics is best done by adjusting
                `min_topic_size` first before adjusting this parameter.
    low_memory: Sets UMAP low memory to True to make sure less memory is used.
                NOTE: This is only used in UMAP. For example, if you use PCA instead of UMAP
                this parameter will not be used.
    calculate_probabilities: Calculate the probabilities of all topics
                                per document instead of the probability of the assigned
                                topic per document. This could slow down the extraction
                                of topics if you have many documents (> 100_000).
                                NOTE: If false you cannot use the corresponding
                                visualization method `visualize_probabilities`.
                                NOTE: This is an approximation of topic probabilities
                                as used in HDBSCAN and not an exact representation.
    seed_topic_list: A list of seed words per topic to converge around
    zeroshot_topic_list: A list of topic names to use for zero-shot classification
    zeroshot_min_similarity: The minimum similarity between a zero-shot topic and
                                a document for assignment. The higher this value, the more
                                confident the model needs to be to assign a zero-shot topic to a document.
    verbose: Changes the verbosity of the model, Set to True if you want
                to track the stages of the model.
    embedding_model: Use a custom embedding model.
                        The following backends are currently supported
                        * SentenceTransformers
                        * Flair
                        * Spacy
                        * Gensim
                        * USE (TF-Hub)
                        You can also pass in a string that points to one of the following
                        sentence-transformers models:
                        * https://www.sbert.net/docs/pretrained_models.html
    umap_model: Pass in a UMAP model to be used instead of the default.
                NOTE: You can also pass in any dimensionality reduction algorithm as long
                as it has `.fit` and `.transform` functions.
    hdbscan_model: Pass in a hdbscan.HDBSCAN model to be used instead of the default
                    NOTE: You can also pass in any clustering algorithm as long as it has
                    `.fit` and `.predict` functions along with the `.labels_` variable.
    vectorizer_model: Pass in a custom `CountVectorizer` instead of the default model.
    ctfidf_model: Pass in a custom ClassTfidfTransformer instead of the default model.
    representation_model: Pass in a model that fine-tunes the topic representations
                            calculated through c-TF-IDF. Models from `bertopic.representation`
                            are supported.
"""
