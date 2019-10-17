#!/home/llu/anaconda3/bin/python
#coding=utf-8

##data source https://github.com/suriyadeepan/practical_seq2seq/tree/master/datasets 
####part I clean data and split (Twitter and movie conversations -- cornell_corpus)
##input and outut are both sentences

from nltk.wsd import lesk
from nltk.corpus import wordnet as wn

    
    
def WSD_generation(input_sentence = "I went to the bank to play Frisbee.", input_word = "bank"):
    
    sense = lesk(input_sentence.split(), input_word, 'n')
    #tt = 'bank.n.07'
    tt = str(sense).replace("Synset('", "").replace("')", "")
    
    similar_words = wn.synset(tt).lemma_names()
    definition = wn.synset(tt).definition()
    
    out1 = input_word + " meaning : " + input_sentence
    out2 = input_word + " meaning : " + definition + "<br>" + "Similar words: " + "; ".join(similar_words)
    out3 = ""
    
    for ss in wn.synsets(input_word):
        #print(ss, ss.definition())
        out3 = out3 + str(ss) + " : " + ss.definition() + "<br>"
        
            
    #return(input_text, translation, true_seq)
    return(out1, out2, out3)







