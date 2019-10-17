#!/home/llu/anaconda3/bin/python
#coding=utf-8

##data source https://github.com/suriyadeepan/practical_seq2seq/tree/master/datasets 
####part I clean data and split (Twitter and movie conversations -- cornell_corpus)
##input and outut are both sentences

import os, sys
import string
import re
from pickle import dump
from unicodedata import normalize
from numpy import array
from pickle import load
from pickle import dump
from numpy.random import rand
from numpy.random import shuffle

def clean_text_complete(datasource = "DS_twitter"):
    
    def twitter_to_pairs(filename = "twitter_en.txt"):
        #read file
        file = open('../data/DS_data/' + filename, mode='rt', encoding='utf-8')
        text = file.read()
        file.close()
        lines = text.strip().split('\n')
        count = len(lines) //2
        pairs = list()
        for i in range(count - 1):
            pairs.append( lines[i:(i+2)] )
        return pairs        

    def movie_to_pairs(movie_lines = "movie_lines.txt", movie_conv = "movie_conversations.txt"):
        
        movie_lines = open('../data/DS_data/' + movie_lines, mode='rt', encoding='utf-8')
        text1 = movie_lines.read()
        movie_lines.close()

        conv = open('DS_data/' + movie_conv, mode='rt', encoding='utf-8')
        text2 = conv.read()
        conv.close()
        
        text1 = text1.strip().split('\n')
        text2 = text2.strip().split('\n')
        
        lines = text2.strip().split('\n')
        
        count = len(lines) //2
        for i in range(count - 1):
            pairs = "\t".join(lines[i:(i+2)])
        return pairs
        
    # split a loaded document into sentences
    def to_pairs(doc):
        lines = doc.strip().split('\n')
        pairs = [line.split('\t') for line in  lines]
        return pairs
     
    # clean a list of lines
    def clean_pairs(lines):
        cleaned = list()
        # prepare regex for char filtering
        re_print = re.compile('[^%s]' % re.escape(string.printable))
        # prepare translation table for removing punctuation
        table = str.maketrans('', '', string.punctuation)
        for pair in lines:
            clean_pair = list()
            for line in pair:
                # normalize unicode characters
                line = normalize('NFD', line).encode('ascii', 'ignore')
                line = line.decode('UTF-8')
                # tokenize on white space
                line = line.split()
                # convert to lowercase
                line = [word.lower() for word in line]
                # remove punctuation from each token
                line = [word.translate(table) for word in line]
                # remove non-printable chars form each token
                line = [re_print.sub('', w) for w in line]
                # remove tokens with numbers in them
                line = [word for word in line if word.isalpha()]
                # store as string
                clean_pair.append(' '.join(line))
            cleaned.append(clean_pair)
        return array(cleaned)
     
    # save a list of clean sentences to file
    def save_clean_data(sentences, filename):
        dump(sentences, open(filename, 'wb'))
        print('Saved: %s' % filename)
     
    # load dataset
    datasource = "DS_twitter"
    pairs = twitter_to_pairs()
    # clean sentences
    #clean_pairs = clean_pairs(pairs)
    clean_pairs = array(pairs)
    #clean_pairs[:, 1] = [ " ".join(list(one)) for one in clean_pairs[:, 1]]
    # save clean pairs to file
    #save_clean_data(clean_pairs, 'model/english-chinese.pkl')
    save_clean_data(clean_pairs, 'model/' + datasource + '.pkl')
    # spot check
    for i in range(100):
        print('[%s] => [%s]' % (clean_pairs[i,0], clean_pairs[i,1]))
        
    # load a clean dataset
    def load_clean_sentences(filename):
        return load(open(filename, 'rb'))
     
    # save a list of clean sentences to file
    def save_clean_data(sentences, filename):
        dump(sentences, open(filename, 'wb'))
        print('Saved: %s' % filename)
     
    # load dataset
    raw_dataset = load_clean_sentences('model/' + datasource + '.pkl') #377264
     
    # reduce dataset size
    n_sentences = 1000
    dataset = raw_dataset[:n_sentences, :]
    # random shuffle
    shuffle(dataset)
    # split into train/test
    test, train = dataset[:int(n_sentences * 0.9)], dataset[int(n_sentences * 0.9):]
    # save
    #save_clean_data(dataset, 'model/english-chinese-both.pkl')
    #save_clean_data(train, 'model/english-chinese-train.pkl')
    #save_clean_data(test, 'model/english-chinese-test.pkl')

    save_clean_data(dataset, 'model/' + datasource + '-both' + '.pkl')
    save_clean_data(train, 'model/' + datasource + '-train.pkl')
    save_clean_data(test, 'model/' + datasource + '-test.pkl')

#clean_text_complete()

############part II language model training
from pickle import load
from numpy import array
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from keras.utils.vis_utils import plot_model
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense
from keras.layers import Embedding
from keras.layers import RepeatVector
from keras.layers import TimeDistributed
from keras.callbacks import ModelCheckpoint
from keras.callbacks import TensorBoard
#tensorboard --logdir=./TFboardlog/E_C_translation_model

def language_model_training():
    
    # load a clean dataset
    def load_clean_sentences(filename):
        return load(open(filename, 'rb'))
     
    # fit a tokenizer
    def create_tokenizer(lines):
        tokenizer = Tokenizer()
        tokenizer.fit_on_texts(lines)
        return tokenizer
     
    # max sentence length
    def max_length(lines):
        return max(len(line.split()) for line in lines)
     
    # encode and pad sequences
    def encode_sequences(tokenizer, length, lines):
        # integer encode sequences
        X = tokenizer.texts_to_sequences(lines)
        # pad sequences with 0 values
        X = pad_sequences(X, maxlen=length, padding='post')
        return X
     
    # one hot encode target sequence
    def encode_output(sequences, vocab_size):
        ylist = list()
        for sequence in sequences:
            encoded = to_categorical(sequence, num_classes=vocab_size)
            ylist.append(encoded)
        y = array(ylist)
        y = y.reshape(sequences.shape[0], sequences.shape[1], vocab_size)
        return y
     
    # define NMT model
    def define_model(src_vocab, tar_vocab, src_timesteps, tar_timesteps, n_units):
        model = Sequential()
        model.add(Embedding(src_vocab, n_units, input_length=src_timesteps, mask_zero=True))
        model.add(LSTM(n_units))
        model.add(RepeatVector(tar_timesteps))
        model.add(LSTM(n_units, return_sequences=True))
        model.add(TimeDistributed(Dense(tar_vocab, activation='softmax')))
        return model
     
    # load datasets
    datasource = "DS_twitter"
    dataset = load_clean_sentences(sys.path[-1] + 'model/' + datasource + '-both' + '.pkl')
    test = load_clean_sentences(sys.path[-1] + 'model/' + datasource + '-train' + '.pkl')
    train = load_clean_sentences(sys.path[-1] + 'model/' + datasource + '-test' + '.pkl')    
    
    # prepare english tokenizer
    eng_tokenizer = create_tokenizer(dataset[:, 0])
    eng_vocab_size = len(eng_tokenizer.word_index) + 1
    eng_length = max_length(dataset[:, 0])
    print('English Vocabulary Size: %d' % eng_vocab_size)
    print('English Max Length: %d' % (eng_length))
    # prepare german tokenizer
    #dataset[:, 1] = [ " ".join(list(one)) for one in dataset[:, 1]]
    
    ger_tokenizer = create_tokenizer(dataset[:, 1])
    ger_vocab_size = len(ger_tokenizer.word_index) + 1
    ger_length = max_length(dataset[:, 1])
    print('Chinese Vocabulary Size: %d' % ger_vocab_size)
    print('Chinese Max Length: %d' % (ger_length))
     
    # prepare training data
    trainX = encode_sequences(eng_tokenizer, eng_length, train[:, 0])
    trainY = encode_sequences(ger_tokenizer, ger_length, train[:, 1])
    trainY = encode_output(trainY, ger_vocab_size)
    
    # prepare validation data
    testX = encode_sequences(eng_tokenizer, eng_length, test[:, 0])
    testY = encode_sequences(ger_tokenizer, ger_length, test[:, 1])
    testY = encode_output(testY, ger_vocab_size)
     
    # define model
    model = define_model(eng_vocab_size, ger_vocab_size, eng_length, ger_length, 256)
    model.compile(optimizer='adam', loss='categorical_crossentropy')
    # summarize defined model
    print(model.summary())
    #plot_model(model, to_file='model/E_C_translation_model.png', show_shapes=True)
    # fit model
    filename = datasource + '_model.h5'
    checkpoint = ModelCheckpoint(filename, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
    print(trainX.shape, trainY.shape, testX.shape, testY.shape)
    model.fit(trainX, trainY, epochs=100, batch_size=64, validation_data=(testX, testY), callbacks=[TensorBoard(log_dir='./TFboardlog/E_C_translation_model')])
    #model.predict_classes(trainX, verbose=0)
    
    # save the model to file
    model.save('./model/' + datasource + '_model.h5')

#language_model_training()

#####Part III language model to generate text
from pickle import load
from numpy import array
from numpy import argmax
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import load_model
from nltk.translate.bleu_score import corpus_bleu
from random import randint
from keras.backend import clear_session


def DS_generation():
    
    # load a clean dataset
    def load_clean_sentences(filename):
        return load(open(filename, 'rb'))
     
    # fit a tokenizer
    def create_tokenizer(lines):
        tokenizer = Tokenizer()
        tokenizer.fit_on_texts(lines)
        return tokenizer
     
    # max sentence length
    def max_length(lines):
        return max(len(line.split()) for line in lines)
     
    # encode and pad sequences
    def encode_sequences(tokenizer, length, lines):
        # integer encode sequences
        X = tokenizer.texts_to_sequences(lines)
        # pad sequences with 0 values
        X = pad_sequences(X, maxlen=length, padding='post')
        return X
     
    # map an integer to a word
    def word_for_id(integer, tokenizer):
        for word, index in tokenizer.word_index.items():
            if index == integer:
                return word
        return None
     
    # generate target given source sequence
    def predict_sequence(model, tokenizer, source):
        prediction = model.predict(source, verbose=0)[0]
        integers = [argmax(vector) for vector in prediction]
        target = list()
        for i in integers:
            word = word_for_id(i, tokenizer)
            if word is None:
                break
            target.append(word)
        return ' '.join(target)
     
    # evaluate the skill of the model
    def evaluate_model(model, tokenizer, sources, raw_dataset):
        actual, predicted = list(), list()
        for i, source in enumerate(sources):
            # translate encoded source text
            source = source.reshape((1, source.shape[0]))
            translation = predict_sequence(model, tokenizer, source)
            raw_src, raw_target = raw_dataset[i]
            if i < 10:
                print('src=[%s], target=[%s], predicted=[%s]' % (raw_src, raw_target, translation))
            actual.append(raw_target.split())
            predicted.append(translation.split())
        # calculate BLEU score
        print('BLEU-1: %f' % corpus_bleu(actual, predicted, weights=(1.0, 0, 0, 0)) )
        print('BLEU-2: %f' % corpus_bleu(actual, predicted, weights=(0.5, 0.5, 0, 0)) )
        print('BLEU-3: %f' % corpus_bleu(actual, predicted, weights=(0.3, 0.3, 0.3, 0)) )
        print('BLEU-4: %f' % corpus_bleu(actual, predicted, weights=(0.25, 0.25, 0.25, 0.25)) )
     
    # load datasets
    datasource = "DS_twitter"
    dataset = load_clean_sentences(sys.path[-1] + './model/' + datasource + '-both.pkl')
    train = load_clean_sentences(sys.path[-1] + './model/' + datasource + '-train.pkl')
    test = load_clean_sentences(sys.path[-1] + './model/' + datasource + '-test.pkl')
    # prepare english tokenizer
    eng_tokenizer = create_tokenizer(dataset[:, 0])
    eng_vocab_size = len(eng_tokenizer.word_index) + 1
    eng_length = max_length(dataset[:, 0])
    # prepare chinese tokenizer
    #dataset[:, 1] = [ " ".join(list(one)) for one in dataset[:, 1]]
    
    ger_tokenizer = create_tokenizer(dataset[:, 1])
    ger_vocab_size = len(ger_tokenizer.word_index) + 1
    ger_length = max_length(dataset[:, 1])
    # prepare data
    trainX = encode_sequences(eng_tokenizer, eng_length, train[:, 0])
    testX = encode_sequences(eng_tokenizer, eng_length, test[:, 0])
     
    # load model
    model = load_model(sys.path[-1] + './model/' + datasource + '_model.h5')
    # test on some training sequences
    print('train')
    #evaluate_model(model, ger_tokenizer, trainX, train)
    # test on some test sequences
    print('test')
    #evaluate_model(model, ger_tokenizer, testX, test)
    
    ## translate sentences
    index = randint(0, 100)
    X = train[:, 0][index]
    true_seq = train[:, 1][index]
    
    trainXX = encode_sequences(eng_tokenizer, eng_length, train[:, 0])[index]
    source = trainXX.reshape((1, trainXX.shape[0]))
    translation = predict_sequence(model, ger_tokenizer, source)
    
    true_seq = train[:, 1][index]
    
    print("Input English: ", train[:, 0][index])
    print("Translated Chinese: ", translation)
    print("True Chinese: ", true_seq)
    
    clear_session()
    
    input_text = train[:, 0][index]
    return(input_text, translation, true_seq)

#DS_generation()

#seed_text, output, true_seq = DS_generation()

#print(seed_text, output)






