from bertopic import BERTopic
from sklearn.feature_extraction.text import CountVectorizer
import jieba

def tokenize_zh(text):
    words = jieba.lcut(text)
    return words

def training_model(trainingDoc, stop_words, top_n_words, min_topic_size, max_topic_num):
    data = list(trainingDoc.values())
    vectorizer = CountVectorizer(stop_words = stop_words, tokenizer=tokenize_zh)
    model = BERTopic(embedding_model='uer/sbert-base-chinese-nli', verbose=True, vectorizer_model=vectorizer, top_n_words=top_n_words, min_topic_size=min_topic_size, nr_topics = max_topic_num)
    model.fit(data)
    return model

