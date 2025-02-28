import opencc
import hanzidentifier

def Simplified2Traditional(str):
    converter = opencc.OpenCC('s2t.json')
    return converter.convert(str)

def Traditional2Simplified(str):
    converter = opencc.OpenCC('t2s.json')
    return converter.convert(str)

def ExpandStopwordList(stopwords):
    new_stopwords = []
    for stopword in stopwords:
        new_stopwords.append(stopword)
        if hanzidentifier.is_simplified(stopword):
            new_stopword = Simplified2Traditional(stopword)
            if new_stopword != stopword:
                new_stopwords.append(new_stopword)
        else:
            new_stopword = Traditional2Simplified(stopword)
            if new_stopword != stopword:
                new_stopwords.append(new_stopword)
    return new_stopwords     

def AddStopwords(stopwords, new_words):
    for word in new_words:
        stopwords.append(word)
        if hanzidentifier.is_simplified(word):
            new_word = Simplified2Traditional(word)
            if new_word != word:
                stopwords.append(new_word)
        else:
            new_word = Traditional2Simplified(word)
            if new_word != word:
                stopwords.append(new_word)
    return stopwords

def DeleteStopwords(stopwords, useless_words):
    for word in useless_words:
        stopwords.remove(word)
        if hanzidentifier.is_simplified(word):
            another_word = Simplified2Traditional(word)
            if another_word != word:
                stopwords.remove(another_word)
        else:
            another_word = Traditional2Simplified(word)
            if another_word != word:
                stopwords.remove(another_word)
    return stopwords
