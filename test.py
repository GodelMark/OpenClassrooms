from flask import Flask, request, render_template
import sklearn
import nltk
import re
import joblib
import numpy as np
from nltk.tokenize import sent_tokenize, word_tokenize
import time
import tensorflow_hub as hub
from nltk.stem import WordNetLemmatizer
#nltk.download('punkt_tab')

def bracket_exterminator(sentence):
    ret = ''
    skip1c = 0
    skip2c = 0
    for i in sentence:
        if i == '<':
            skip1c += 1
        elif i == '>' and skip1c > 0:
            skip1c -= 1
        elif skip1c == 0 and skip2c == 0:
            ret += i
    return ret

def code_exterminator(sentence):
    idx1 = sentence.find("<code>")
    idx2 = sentence.find("<\code>")
    res = sentence[:idx1] +  sentence[:idx2 + len("<\code>") + 1]
    return res

def tokenizer_fct(sentence) :
    # print(sentence)
    sentence_clean = sentence.replace('-', ' ').replace('+', ' ').replace('/', ' ').replace('#', ' ').replace('. ', ' ').replace('=', ' ').replace('(', ' ').replace(')', ' ').replace('{', ' ').replace('}', ' ').replace(',', ' ').replace(';', ' ').replace('.', ' ').replace('<', ' ')
    word_tokens = word_tokenize(sentence_clean)
    return word_tokens

# Stop words
#from nltk.corpus import stopwords
#stop_w = list(set(stopwords.words('english'))) + ['[', ']', ',', ':', '?', '(', ')','{','}']

def stop_word_filter_fct(list_words) :
#    filtered_w = [w for w in list_words if not w in stop_w]
    filtered_w2 = [w for w in list_words if len(w) > 2]
    return filtered_w2

# lower case et alpha
def lower_start_fct(list_words) :
    lw = [w.lower() for w in list_words if (not w.startswith("@")) 
    #                                   and (not w.startswith("#"))
                                       and (not w.startswith("http"))]
    return lw

# Lemmatizer (base d'un mot)
from nltk.stem import WordNetLemmatizer

def lemma_fct(list_words) :
    lemmatizer = WordNetLemmatizer()
    lem_w = [lemmatizer.lemmatize(w) for w in list_words]
    return lem_w

# Fonction de préparation du texte pour le bag of words (Countvectorizer et Tf_idf, Word2Vec)
def transform_bow_fct(desc_text) :
    word_tokens = tokenizer_fct(desc_text)
    sw = stop_word_filter_fct(word_tokens)
    lw = lower_start_fct(sw)
    # lem_w = lemma_fct(lw)    
    transf_desc_text = ' '.join(lw)
    return transf_desc_text

# Fonction de préparation du texte pour le bag of words avec lemmatization
def transform_bow_lem_fct(desc_text) :
    word_tokens = tokenizer_fct(desc_text)
    sw = stop_word_filter_fct(word_tokens)
    lw = lower_start_fct(sw)
    lem_w = lemma_fct(lw)    
    transf_desc_text = ' '.join(lem_w)
    return transf_desc_text

# Fonction de préparation du texte pour le Deep learning (USE et BERT)
def transform_dl_fct(desc_text) :
    word_tokens = tokenizer_fct(desc_text)
#    sw = stop_word_filter_fct(word_tokens)
    lw = lower_start_fct(word_tokens)
    # lem_w = lemma_fct(lw)    
    transf_desc_text = ' '.join(lw)
    return transf_desc_text

def transform_dl_fct2(desc_text) :
#    word_tokens = tokenizer_fct(desc_text)
#    sw = stop_word_filter_fct(word_tokens)
#    lw = lower_start_fct(desc_text)
    # lem_w = lemma_fct(lw)    
#    transf_desc_text = ' '.join(word_tokens)
    return desc_text

def bert_inp_fct(sentences, bert_tokenizer, max_length) :
    input_ids=[]
    token_type_ids = []
    attention_mask=[]
    bert_inp_tot = []

    for sent in sentences:
        bert_inp = bert_tokenizer.encode_plus(sent,
                                              add_special_tokens = True,
                                              max_length = max_length,
                                              padding='max_length',
                                              return_attention_mask = True, 
                                              return_token_type_ids=True,
                                              truncation=True,
                                              return_tensors="tf")
    
        input_ids.append(bert_inp['input_ids'][0])
        token_type_ids.append(bert_inp['token_type_ids'][0])
        attention_mask.append(bert_inp['attention_mask'][0])
        bert_inp_tot.append((bert_inp['input_ids'][0], 
                             bert_inp['token_type_ids'][0], 
                             bert_inp['attention_mask'][0]))

    input_ids = np.asarray(input_ids)
    token_type_ids = np.asarray(token_type_ids)
    attention_mask = np.array(attention_mask)
    
    return input_ids, token_type_ids, attention_mask, bert_inp_tot

embed = hub.load("https://tfhub.dev/google/universal-sentence-encoder/4")


def feature_USE_fct(sentences, b_size) :
    batch_size = b_size
    time1 = time.time()

    for step in range(len(sentences)//batch_size) :
        idx = step*batch_size
        #with tf.device('/CPU:0'):
        #    feat = embed(sentences[idx:idx+batch_size])
        feat = embed(sentences[idx:idx+batch_size])

        if step ==0 :
            features = feat
        else :
            features = np.concatenate((features,feat))

    time2 = np.round(time.time() - time1,0)
    return features

