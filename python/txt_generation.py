#!/home/llu/anaconda3/bin/python
#coding=utf-8

##tutorial from https://machinelearningmastery.com/how-to-develop-a-word-level-neural-language-model-in-keras/
####part I clean data (The Republic by Plato)
import os, sys
import string

sys.path.append("./")

def clean_text_complete():
    
    # load doc into memory
    def load_doc(filename):
      # open the file as read only
      file = open(filename, 'r', encoding="utf-8")
      # read all text
      text = file.read()
      # close the file
      file.close()
      return text
     
    # turn a doc into clean tokens
    def clean_doc(doc):
      # replace '--' with a space ' '
      doc = doc.replace('--', ' ')
      # split into tokens by white space
      tokens = doc.split()
      # remove punctuation from each token
      #table = str.maketrans('', '', string.punctuation)
      #tokens = [w.translate(table) for w in tokens]
      # remove remaining tokens that are not alphabetic
      #tokens = [word for word in tokens if word.isalpha()]
      # make lower case
      #tokens = [word.lower() for word in tokens]
      return tokens
     
    # save tokens to file, one dialog per line
    def save_doc(lines, filename):
      data = '\n'.join(lines)
      file = open(filename, 'w')
      file.write(data)
      file.close()
     
    # load document
    in_filename = '../data/republic.txt'
    doc = load_doc(in_filename)
    print(doc[:200])
     
    # clean document
    tokens = clean_doc(doc)
    print(tokens[:200])
    print('Total Tokens: %d' % len(tokens))
    print('Unique Tokens: %d' % len(set(tokens)))
     
    # organize into sequences of tokens
    length = 50 + 1
    sequences = list()
    for i in range(length, len(tokens)):
      # select sequence of tokens
      seq = tokens[i-length:i]
      # convert into a line
      line = ' '.join(seq)
      # store
      sequences.append(line)
    print('Total Sequences: %d' % len(sequences))
     
    # save sequences to file
    out_filename = '../data/republic_sequences.txt'
    save_doc(sequences, out_filename)    
    
    print("clean text finished!!!")
    
clean_text_complete()    
    
# 
# from nltk.tokenize import sent_tokenize
# 
# def clean_text_complete(filename = '../data/republic.txt', out_filename = '../data/republic_sequences2.txt'):
#     
#     file = open(filename, 'r', encoding="utf-8")
#     text = file.read()
#     file.close()
#     
#     doc = text.replace("\n", " ").lower() 
#     tokens = doc.split()
#     print(tokens[:200])
#     print('Total Tokens: %d' % len(tokens))
#     print('Unique Tokens: %d' % len(set(tokens)))    
#     
#     sent_tokenize_list = sent_tokenize(doc)
#     print(sent_tokenize_list[:5])
#     print('Total Sentences: %d' % len(sent_tokenize_list))    
#     
#     out_filename = '../data/republic_sequences2.txt'
#     data = '\n'.join(sent_tokenize_list)
#     out = open(out_filename, 'w')
#     out.write(data)
#     out.close()    
#     
# clean_text_complete()    
    

############part II language model training
from numpy import array
from pickle import dump
from keras.preprocessing.text import Tokenizer
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Embedding

def language_model_training():
    
    # load doc into memory
    def load_doc(filename):
      # open the file as read only
      file = open(filename, 'r', encoding="utf-8")
      # read all text
      text = file.read()
      # close the file
      file.close()
      return text
     
    # load
    in_filename = '../data/republic_sequences.txt'
    doc = load_doc(in_filename)
    lines = doc.split('\n')
     
    # integer encode sequences of words
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(lines)
    sequences = tokenizer.texts_to_sequences(lines)
    # vocabulary size
    vocab_size = len(tokenizer.word_index) + 1
     
    # separate into input and output
    sequences = array(sequences)
    X, y = sequences[:,:-1], sequences[:,-1]
    y = to_categorical(y, num_classes=vocab_size)
    seq_length = X.shape[1]
     
    # define model
    model = Sequential()
    model.add(Embedding(vocab_size, 50, input_length=seq_length))
    model.add(LSTM(100, return_sequences=True))
    model.add(LSTM(100))
    model.add(Dense(100, activation='relu'))
    model.add(Dense(vocab_size, activation='softmax'))
    print(model.summary())
    # compile model
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    # fit model
    model.fit(X, y, batch_size=128, epochs=100)
     
    # save the model to file
    model.save('./model/txt_generation_model.h5')
    # save the tokenizer
    dump(tokenizer, open('txt_generation_tokenizer.pkl', 'wb'))

language_model_training()

#####Part III language model to generate text
from random import randint
from pickle import load
from keras.models import load_model
from keras.preprocessing.sequence import pad_sequences

def txt_generation():
    
    # load doc into memory
    def load_doc(filename):
      # open the file as read only
      file = open(filename, 'r')
      # read all text
      text = file.read()
      # close the file
      file.close()
      return text
     
    # generate a sequence from a language model
    def generate_seq(model, tokenizer, seq_length, seed_text, n_words):
      result = list()
      in_text = seed_text
      # generate a fixed number of words
      for _ in range(n_words):
        # encode the text as integer
        encoded = tokenizer.texts_to_sequences([in_text])[0]
        print(encoded)
        # truncate sequences to a fixed length
        encoded = pad_sequences([encoded], maxlen=seq_length, truncating='pre')
        # predict probabilities for each word
        yhat = model.predict_classes(encoded, verbose=0)
        # map predicted word index to word
        out_word = ''
        for word, index in tokenizer.word_index.items():
          if index == yhat:
            out_word = word
            print(word)
            break
        # append to input
        in_text += ' ' + out_word
        result.append(out_word)
      return ' '.join(result)
     
    # load cleaned text sequences
    in_filename = sys.path[-1] + '../data/republic_sequences.txt'
    doc = load_doc(in_filename)
    lines = doc.split('\n')
    seq_length = len(lines[0].split()) - 1
     
    # load the model
    model = load_model(sys.path[-1] + 'model/txt_generation_model.h5')
     
    # load the tokenizer
    tokenizer = load(open(sys.path[-1] + 'model/txt_generation_tokenizer.pkl', 'rb'))
     
    # select a seed text
    seed_text = lines[randint(0,len(lines))]
    print(seed_text + '\n')
     
    # generate new text
    generated = generate_seq(model, tokenizer, seq_length, seed_text, 50)
    print(generated)
    
    return(seed_text, generated)

seed_text, output = txt_generation()

print(seed_text, output)


