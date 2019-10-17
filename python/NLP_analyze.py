#!/home/llu/anaconda3/bin/python

import os, sys
import nltk
from gensim.models import Word2Vec    
from nltk.tokenize import sent_tokenize
from collections import Counter
import re

def read_file(filename = 'alltranscript.txt'):
    
    ff = open('../data/HouseMD_data/alltranscript.txt', 'r')
    content = ff.read().split('\n')
    
    return(content)
    
def Word_Vector():
    
    content = read_file()
    sentences = [one for one in content if one.startswith('House:')]
    sentences = [one.replace('House:', '') for one in sentences ]
    
    ##most common sentences
    sen_nltk = ' '.join(sentences)
    sent_tokenize_list = sent_tokenize(sen_nltk)
    print('Total sentences House said are %d' % len(sent_tokenize_list))
    print('Top 10 responses: %s' % Counter(sent_tokenize_list).most_common(10))
    
    #word similarity
    wv_text = [one.split(' ') for one in sentences]
    model = Word2Vec(wv_text, min_count=1)
    words = list(model.wv.vocab)
    print('The similarity between moron and idiot is %.3f' % model.wv.similarity('moron', 'idiot'))
    
    model.wv.most_similar(['idiot', 'moron'])
    
    text = ' '.join(sentences).split(' ')
    text = nltk.Text(text)
    text.similar('idiot')
    text.similar('moron')
    text.count('idiot'); text.count('moron');

    #add tagger and find verbs
    tagger = nltk.word_tokenize(' '.join(sentences))
    tagger = nltk.pos_tag(tagger)

    verbs = [one for one in tagger if 'VB' in one[1]]
    verbs = [one[0] for one in verbs if len(one[0]) > 2]

    Counter(verbs).most_common(10)
    
    
    

