#!/home/llu/anaconda3/bin/python
#coding=utf-8

##data source https://github.com/suriyadeepan/practical_seq2seq/tree/master/datasets 
####part I clean data and split (Twitter and movie conversations -- cornell_corpus)
##input and outut are both sentences


from keras.models import Model
from keras.layers import Input, LSTM, Dense, GRU
import numpy as np
np.random.seed(1234)  # for reproducibility
import pandas as pd
import numpy as np    
from keras.preprocessing import sequence
from scipy import sparse, io
from numpy.random import permutation
import re
from keras.preprocessing.sequence import pad_sequences
from keras.layers.embeddings import Embedding
import numpy as np
from numpy import argmax
from numpy import array_equal
from keras.utils import to_categorical
from string import punctuation
from os import listdir
from collections import Counter
from nltk.corpus import stopwords
from gensim.models import Word2Vec
from keras.preprocessing.text import Tokenizer    
from keras.callbacks import ModelCheckpoint
from keras.models import model_from_json
from keras.callbacks import TensorBoard, ReduceLROnPlateau
import time
from keras.layers import Bidirectional
import string
from string import digits
from contextlib import redirect_stdout
from nltk.tokenize.stanford import StanfordTokenizer
from nltk.tokenize.toktok import ToktokTokenizer
from nltk.tokenize import word_tokenize
import enchant
from keras.utils.vis_utils import plot_model
from keras import regularizers
from rouge import Rouge
from random import randint
from keras.backend import clear_session

import scipy.spatial as sp
from nltk.translate.bleu_score import sentence_bleu

import pydot
#except ImportError:
    #pydot = None

#####load model plot function
    
def _check_pydot():
    """Raise errors if `pydot` or GraphViz unavailable."""
    if pydot is None:
        raise ImportError(
            'Failed to import `pydot`. '
            'Please install `pydot`. '
            'For example with `pip install pydot`.')
    try:
        # Attempt to create an image of a blank graph
        # to check the pydot/graphviz installation.
        pydot.Dot.create(pydot.Dot())
    except OSError:
        raise OSError(
            '`pydot` failed to call GraphViz.'
            'Please install GraphViz (https://www.graphviz.org/) '
            'and ensure that its executables are in the $PATH.')


def model_to_dot(model, show_shapes=False, show_layer_names=True, rankdir='TB', expand_nested=False, dpi=96, subgraph=False):
    
    from keras.layers.wrappers import Wrapper
    #from keras.layers.wrappers import Wrapper
    #from ..layers.wrappers import Wrapper
    #from ..models import Model
    #from ..models import Sequential
    from keras.models import Sequential
    from keras.models import Model
    
    #_check_pydot()
    if subgraph:
        dot = pydot.Cluster(style='dashed')
        dot.set('label', model.name)
        dot.set('labeljust', 'l')
    else:
        dot = pydot.Dot()
        dot.set('rankdir', rankdir)
        dot.set('concentrate', True)
        dot.set('dpi', dpi)
        dot.set_node_defaults(shape='record')

    if isinstance(model, Sequential):
        if not model.built:
            model.build()
            
    layers = model._layers

    # Create graph nodes.
    for i, layer in enumerate(layers):
        layer_id = str(id(layer))

        # Append a wrapped layer's label to node's label, if it exists.
        layer_name = layer.name
        class_name = layer.__class__.__name__
        if isinstance(layer, Wrapper):
            if expand_nested and isinstance(layer.layer, Model):
                submodel = model_to_dot(layer.layer, show_shapes,
                                        show_layer_names, rankdir, expand_nested,
                                        subgraph=True)
                model_nodes = submodel.get_nodes()
                dot.add_edge(pydot.Edge(layer_id, model_nodes[0].get_name()))
                if len(layers) > i + 1:
                    next_layer_id = str(id(layers[i + 1]))
                    dot.add_edge(pydot.Edge(
                        model_nodes[len(model_nodes) - 1].get_name(),
                        next_layer_id))
                dot.add_subgraph(submodel)
            else:
                layer_name = '{}({})'.format(layer_name, layer.layer.name)
                child_class_name = layer.layer.__class__.__name__
                class_name = '{}({})'.format(class_name, child_class_name)

        # Create node's label.
        if show_layer_names:
            label = '{}: {}'.format(layer_name, class_name)
        else:
            label = class_name

        # Rebuild the label as a table including input/output shapes.
        if show_shapes:
            try:
                outputlabels = str(layer.output_shape)
            except AttributeError:
                outputlabels = 'multiple'
            if hasattr(layer, 'input_shape'):
                inputlabels = str(layer.input_shape)
            elif hasattr(layer, 'input_shapes'):
                inputlabels = ', '.join(
                    [str(ishape) for ishape in layer.input_shapes])
            else:
                inputlabels = 'multiple'
            label = '%s\n|{input:|output:}|{{%s}|{%s}}' % (label,
                                                           inputlabels,
                                                           outputlabels)
        node = pydot.Node(layer_id, label=label)
        dot.add_node(node)

    # Connect nodes with edges.
    for layer in layers:
        layer_id = str(id(layer))
        for i, node in enumerate(layer._inbound_nodes):
            node_key = layer.name + '_ib-' + str(i)
            if node_key in model._network_nodes:
                for inbound_layer in node.inbound_layers:
                    if not expand_nested or not (
                            isinstance(inbound_layer, Wrapper) and
                            isinstance(inbound_layer.layer, Model)):
                        inbound_layer_id = str(id(inbound_layer))
                        # Make sure that both nodes exist before connecting them with
                        # an edge, as add_edge would otherwise
                        # create any missing node.
                        assert dot.get_node(inbound_layer_id)
                        assert dot.get_node(layer_id)
                        dot.add_edge(pydot.Edge(inbound_layer_id, layer_id))
    return dot


def plot_model(model,
               to_file='model.png',
               show_shapes=False,
               show_layer_names=True,
               rankdir='TB',
               expand_nested=False,
               dpi=96):
    """
    Converts a Keras model to dot format and save to a file.
    Arguments
        model: A Keras model instance
        to_file: File name of the plot image.
        show_shapes: whether to display shape information.
        show_layer_names: whether to display layer names.
        rankdir: `rankdir` argument passed to PyDot,
            a string specifying the format of the plot:
            'TB' creates a vertical plot;
            'LR' creates a horizontal plot.
        expand_nested: whether to expand nested models into clusters.
        dpi: dot DPI.
    
        A Jupyter notebook Image object if Jupyter is installed.
        This enables in-line display of the model plots in notebooks.
    """
    dot = model_to_dot(model, show_shapes, show_layer_names, rankdir,
                       expand_nested, dpi)
    _, extension = os.path.splitext(to_file)
    if not extension:
        extension = 'png'
    else:
        extension = extension[1:]
    dot.write(to_file, format=extension)
    # Return the image as a Jupyter Image object, to be displayed in-line.
    try:
        from IPython import display
        return display.Image(filename=to_file)
    except ImportError:
        pass
  
    
def is_ascii(s):
    return all(ord(c) < 128 for c in s) and len(s) < 50

def read_reddit():
    
    dir_data = "/home/llu/HardDisk/LiangqunLuGitHub/DLForChatbot/NMT_sentx/"
    train_from = dir_data + "train.from"
    train_to = dir_data + "train.to"

    N = n_samples
    with open(train_from, 'r', encoding='utf-8') as f:
        #input_lines = f.read().split('\n')
        input_lines = [next(f).strip() for x in range(N)]
    
    with open(train_to, 'r', encoding='utf-8') as f:
        #target_lines = f.read().split('\n')  
        target_lines = [next(f).strip() for x in range(N)]        

    print("Input and output lines: %d and %d"%(len(input_lines), len(target_lines)) ) 
    # Vectorize the data.
    input_texts = []
    target_texts = []  
    #check english words
    #d = enchant.Dict("en_US")
    for ll in range(len(input_lines)):
        
        input_txt = ToktokTokenizer().tokenize(input_lines[ll], return_str=True)
        output_txt = ToktokTokenizer().tokenize(target_lines[ll], return_str=True)
        
        #if min(len(input_txt.split()), len(output_txt.split())) > 5:
            #continue
        #else:
        if max(len(input_txt.split()), len(output_txt.split())) <= 5 and max(len(input_txt.split()), len(output_txt.split())) > 2 and is_ascii(input_txt) and is_ascii(output_txt):
        
            #input_text, target_text = line.split('\t')
            #input_text = input_lines[ll].lower()
            #target_text = target_lines[ll].lower()
            input_text = input_txt.lower()
            target_text = output_txt.lower()            
            
            # We use "tab" as the "start sequence" character
            # for the targets, and "\n" as "end sequence" character.
            input_text = input_text.replace('\n', '')            
            target_text = target_text.replace('\n', '')
            input_texts.append(input_text)
            target_texts.append(target_text)
                                  
    # NMT concepts and parameters
    # Keras NMT tutorial https://github.com/keras-team/keras/blob/master/examples/lstm_seq2seq.py           
    #clean input texts and target texts
    
    
    #input_texts, target_texts = clean_input_pairs(input_texts, target_texts)
    lines = pd.DataFrame({'eng': input_texts,'fr': target_texts} ) 

    lines.eng=lines.eng.apply(lambda x: x.lower())
    lines.fr=lines.fr.apply(lambda x: x.lower())
    lines.eng=lines.eng.apply(lambda x: re.sub("'", '', x)).apply(lambda x: re.sub(",", ' ', x))
    lines.fr=lines.fr.apply(lambda x: re.sub("'", '', x)).apply(lambda x: re.sub(",", ' ', x))
    
    exclude = set(string.punctuation)
    lines.eng=lines.eng.apply(lambda x: ''.join(ch for ch in x if ch not in exclude))
    lines.fr=lines.fr.apply(lambda x: ''.join(ch for ch in x if ch not in exclude))
    print(lines.head(n = 10))
    
    lines.head(n = 20).to_csv(datatype + "_first_pairs.csv")
    
    lines.fr = lines.fr.apply(lambda x : '\t'+ x + '\n')
    
    input_texts = lines['eng'].tolist()
    target_texts = lines['fr'].tolist()
    
    max_encoder_seq_length = max([len(txt.split()) for txt in input_texts])
    max_decoder_seq_length = max([len(txt.split()) for txt in target_texts])
    
    print('Number of sample input:', len(input_texts) )    
    print('Number of sample target:', len(target_texts))
    print('Max sequence length for inputs:', max_encoder_seq_length)
    print('Max sequence length for outputs:', max_decoder_seq_length)
    
    #num_encoder_tokens = len(input_texts)
    #num_decoder_tokens = len(target_texts)
    
    return input_texts, target_texts  

def read_movie():
    
    dir_data = "/home/llu/HardDisk/LiangqunLuGitHub/DLForChatbot/MS_DL/data/"
    mydata = dir_data + "movie_dialogue.txt"
    with open(mydata, 'r', encoding="ISO-8859-1") as f:
        lines = f.readlines()

    input_lines, target_lines = list(), list()    
    for one in lines:
        input_lines.append(one.split("\t")[0])
        target_lines.append(one.split("\t")[1])   
    print("Input and output lines: %d and %d"%(len(input_lines), len(target_lines)) ) 
    
    # Vectorize the data.
    input_texts = []
    target_texts = []
        
    for ll in range(len(input_lines)):
        
        input_txt = ToktokTokenizer().tokenize(input_lines[ll], return_str=True)
        output_txt = ToktokTokenizer().tokenize(target_lines[ll], return_str=True)
        
        #if min(len(input_txt.split()), len(output_txt.split())) > 5:
            #continue
        #else:
        if max(len(input_txt.split()), len(output_txt.split())) <= 5 and max(len(input_txt.split()), len(output_txt.split())) > 2:
        
            #input_text, target_text = line.split('\t')
            #input_text = input_lines[ll].lower()
            #target_text = target_lines[ll].lower()
            input_text = input_txt.lower()
            target_text = output_txt.lower()            
            
            # We use "tab" as the "start sequence" character
            # for the targets, and "\n" as "end sequence" character.
            input_text = input_text.replace('\n', '')            
            target_text = target_text.replace('\n', '')
            input_texts.append(input_text)
            target_texts.append(target_text)
                                  
    # NMT concepts and parameters
    # Keras NMT tutorial https://github.com/keras-team/keras/blob/master/examples/lstm_seq2seq.py           
    #clean input texts and target texts
    #input_texts, target_texts = clean_input_pairs(input_texts, target_texts)
    lines = pd.DataFrame({'eng': input_texts,'fr': target_texts} ) 
    
    lines.eng=lines.eng.apply(lambda x: x.lower())
    lines.fr=lines.fr.apply(lambda x: x.lower())
    #lines.eng=lines.eng.apply(lambda x: re.sub("'", '', x)).apply(lambda x: re.sub(",", ' COMMA', x))
    #lines.fr=lines.fr.apply(lambda x: re.sub("'", '', x)).apply(lambda x: re.sub(",", ' COMMA', x))
    lines.eng=lines.eng.apply(lambda x: re.sub("'", '', x)).apply(lambda x: re.sub(",", ' ', x))
    lines.fr=lines.fr.apply(lambda x: re.sub("'", '', x)).apply(lambda x: re.sub(",", ' ', x))
    
    exclude = set(string.punctuation)
    lines.eng=lines.eng.apply(lambda x: ''.join(ch for ch in x if ch not in exclude))
    lines.fr=lines.fr.apply(lambda x: ''.join(ch for ch in x if ch not in exclude))
    print(lines.head(n = 10))
    
    #lines.head(n = 10).to_csv(datatype + "_first_pairs.csv")
    
    #lines.fr = lines.fr.apply(lambda x : 'START_ '+ x + ' _END')
    lines.fr = lines.fr.apply(lambda x : '\t'+ x + '\n')
    
    input_texts = lines['eng'].tolist()
    target_texts = lines['fr'].tolist()
    
    max_encoder_seq_length = max([len(txt.split()) for txt in input_texts])
    max_decoder_seq_length = max([len(txt.split()) for txt in target_texts])
    
    print('Number of sample input:', len(input_texts) )    
    print('Number of sample target:', len(target_texts))
    print('Max sequence length for inputs:', max_encoder_seq_length)
    print('Max sequence length for outputs:', max_decoder_seq_length)
    
    #num_encoder_tokens = len(input_texts)
    #num_decoder_tokens = len(target_texts)
    
    return input_texts, target_texts



def tokenize_char(input_texts, target_texts, datatype = "movie"):
    
    #input_texts, target_texts = read_movie()
    # Vectorize the data.
    input_characters = set()
    target_characters = set()
        
    for ll in range(len(input_texts)):
            input_text = input_texts[ll];
            target_text = target_texts[ll];
            for char in input_text:
                if char not in input_characters:
                    input_characters.add(char)
            for char in target_text:
                if char not in target_characters:
                    target_characters.add(char)                                    
    # NMT concepts and parameters
    # Keras NMT tutorial https://github.com/keras-team/keras/blob/master/examples/lstm_seq2seq.py        
    input_characters = sorted(list(input_characters))
    target_characters = sorted(list(target_characters))
    num_encoder_tokens = len(input_characters)
    num_decoder_tokens = len(target_characters)
    max_encoder_seq_length = max([len(txt) for txt in input_texts])
    max_decoder_seq_length = max([len(txt) for txt in target_texts])
    
    print('Number of samples:', len(input_texts))
    print('Number of unique input tokens:', num_encoder_tokens) # input
    print('Number of unique output tokens:', num_decoder_tokens) #input
    print('Max sequence length for inputs:', max_encoder_seq_length)
    print('Max sequence length for outputs:', max_decoder_seq_length)      

    input_token_index = dict([(char, i) for i, char in enumerate(input_characters)])
    target_token_index = dict([(char, i) for i, char in enumerate(target_characters)])
    
    encoder_input_data = np.zeros((len(input_texts), max_encoder_seq_length, num_encoder_tokens),
        dtype='float32')
    decoder_input_data = np.zeros((len(input_texts), max_decoder_seq_length, num_decoder_tokens),
        dtype='float32')
    decoder_target_data = np.zeros((len(input_texts), max_decoder_seq_length, num_decoder_tokens),
        dtype='float32')
    
    for i, (input_text, target_text) in enumerate(zip(input_texts, target_texts)):
        for t, char in enumerate(input_text):
            encoder_input_data[i, t, input_token_index[char]] = 1.
        for t, char in enumerate(target_text):
            # decoder_target_data is ahead of decoder_input_data by one timestep
            decoder_input_data[i, t, target_token_index[char]] = 1.
            if t > 0:
                # decoder_target_data will be ahead by one timestep
                # and will not include the start character.
                decoder_target_data[i, t - 1, target_token_index[char]] = 1.
    
    print("encoder_input_data shape %s, decoder_input_data shape %s, decoder_target_data shape %s" %(encoder_input_data.shape, decoder_input_data.shape, decoder_target_data.shape))
        
    return encoder_input_data, decoder_input_data, decoder_target_data, input_token_index, target_token_index



def seq2seq_models_LSTM(encoder_input_data, decoder_input_data, decoder_target_data, embedding = False, datatype = "Movie", name_func = "_seq2seq_models_LSTM"):
    
    num_encoder_tokens = encoder_input_data.shape[2]
    num_decoder_tokens = decoder_input_data.shape[2]
    # Define an input sequence and process it.
    encoder_inputs = Input(shape=(None, num_encoder_tokens))
    
    if embedding:
        #num_encoder_tokens = len(input_characters)
        x = Embedding(num_encoder_tokens, latent_dim)(encoder_inputs)
        x, state_h, state_c = LSTM(latent_dim, return_state=True)(x)
    else:
        encoder = LSTM(latent_dim, return_state=True)
        encoder_outputs, state_h, state_c = encoder(encoder_inputs)
        
    encoder_states = [state_h, state_c]
    
    # Set up the decoder, using `encoder_states` as initial state.
    decoder_inputs = Input(shape=(None, num_decoder_tokens))
    if embedding:
        x = Embedding(num_decoder_tokens, latent_dim)(decoder_inputs)
        x = LSTM(latent_dim, return_sequences=True)(x, initial_state=encoder_states)
        decoder_outputs = Dense(num_decoder_tokens, activation='softmax')(x)
    else:
        decoder_lstm = LSTM(latent_dim, return_sequences=True, return_state=True)
        decoder_outputs, _, _ = decoder_lstm(decoder_inputs,initial_state=encoder_states)
        decoder_dense = Dense(num_decoder_tokens, activation='softmax')
        decoder_outputs = decoder_dense(decoder_outputs)
    # Define the model that will turn
    # `encoder_input_data` & `decoder_input_data` into `decoder_target_data`
    model = Model([encoder_inputs, decoder_inputs], decoder_outputs)
    
    # Run training
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['acc'])
    
    #model.fit([encoder_input_data, decoder_input_data], decoder_target_data, batch_size=128, epochs=20, validation_split=0.05)
    
    checkpointer = ModelCheckpoint(filepath='./tmp/' + datatype + name_func + '_weights.hdf5', verbose=1, save_best_only=True)
    #os.makedirs(path, exist_ok=True)
    #embeddings_layer_names = None
    #tensorboard = TensorBoard(log_dir= './'+ datatype + '_train_logs/Model' + name_func, write_graph=True, write_grads=True, write_images=True, embeddings_freq=100, embeddings_layer_names=embeddings_layer_names )
    
    #log_dir= './'+ datatype + '_train_logs/Model' + name_func
    
    #with open('./metadata.tsv', 'w') as f:
        #np.savetxt(f, embedding)
    
    #tensorboard = TensorBoard(log_dir= './'+ datatype + '_train_logs/Model' + name_func, batch_size=batch_size, embeddings_freq=1, embeddings_layer_names=['features'], embeddings_metadata='metadata.tsv', embeddings_data = embedding )
    
    tensorboard = TensorBoard(log_dir= './'+ datatype + '_train_logs/Model' + name_func )
    
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=0.001)
    #model.fit(X_train, Y_train, callbacks=[reduce_lr])

    model.fit([encoder_input_data, decoder_input_data], decoder_target_data,
              batch_size=batch_size, epochs=epochs, validation_split=0.2, callbacks=[tensorboard])
    # Save model
    model.save(datatype + name_func + '.h5')
    
    model.summary()
    with open(datatype + name_func + '_modelsummary.txt', 'w') as f:
        with redirect_stdout(f):
            model.summary()
            
    #plot_model(model, to_file='model.png', show_shapes=True, show_layer_names=True)
    plot_model(model, to_file=datatype + name_func + '_model.png', show_shapes=True, show_layer_names=True)
    
    # Define sampling models
    encoder_model = Model(encoder_inputs, encoder_states)
    
    decoder_state_input_h = Input(shape=(latent_dim,))
    decoder_state_input_c = Input(shape=(latent_dim,))
    decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]
    decoder_outputs, state_h, state_c = decoder_lstm(
        decoder_inputs, initial_state=decoder_states_inputs)
    decoder_states = [state_h, state_c]
    decoder_outputs = decoder_dense(decoder_outputs)
    decoder_model = Model( [decoder_inputs] + decoder_states_inputs, [decoder_outputs] + decoder_states)

    # summarize model
    plot_model(encoder_model, to_file=datatype + name_func + '_encoder_model.png', show_shapes=True, show_layer_names=True)
    plot_model(decoder_model, to_file=datatype + name_func + '_decoder_model.png', show_shapes=True, show_layer_names=True)
    
    with open(datatype + '_encoder' + name_func + '.json', 'w', encoding='utf8') as f:
        f.write(encoder_model.to_json())
    encoder_model.save_weights(datatype + '_encoder' + name_func + '.h5')

    with open(datatype + '_decoder' + name_func + '.json', 'w', encoding='utf8') as f:
        f.write(decoder_model.to_json())
    decoder_model.save_weights(datatype + '_decoder' + name_func + '.h5')
    
    return model, encoder_model, decoder_model


def DS_generation():
    
    def decode_sequence(input_seq):
        # Encode the input as state vectors.
        states_value = encoder_model.predict(input_seq)

        # Generate empty target sequence of length 1.
        target_seq = np.zeros((1, 1, num_decoder_tokens))
        # Populate the first character of target sequence with the start character.
        target_seq[0, 0, target_token_index['\t'] ] = 1.

        # Sampling loop for a batch of sequences
        # (to simplify, here we assume a batch of size 1).
        stop_condition = False
        decoded_sentence = ''
        while not stop_condition:
            output_tokens, h, c = decoder_model.predict(
                [target_seq] + states_value)

            # Sample a token
            sampled_token_index = np.argmax(output_tokens[0, -1, :])
            sampled_char = reverse_target_char_index[sampled_token_index]

            #print(sampled_token_index, sampled_char)

            decoded_sentence += sampled_char

            # Exit condition: either hit max length
            # or find stop character.
            if (sampled_char == '\n' or
               len(decoded_sentence) > max_decoder_seq_length):
                stop_condition = True

            # Update the target sequence (of length 1).
            target_seq = np.zeros((1, 1, num_decoder_tokens))
            target_seq[0, 0, sampled_token_index] = 1.

            # Update states
            states_value = [h, c]

        return decoded_sentence
         
    # load datasets
    n_word_sentence = 30
    n_features = 100

    batch_size = 64  # Batch size for training.
    epochs = 1000  # Number of epochs to train for.
    latent_dim = 128  # Latent dimensionality of the encoding space.
    n_samples = 400000  # Number of samples to train on.
    #obtain data
    #input_texts, target_texts = read_nmt()
    
    datatype = "Movie"
    input_texts, target_texts = read_movie()
    #input_texts = input_texts[:1000]
    #target_texts = target_texts[:1000]
    
    '''
    datatype = "Reddit"
    input_texts, target_texts = read_reddit()
    '''
    
    max_encoder_seq_length = max([len(txt.split()) for txt in input_texts])
    max_decoder_seq_length = max([len(txt.split()) for txt in target_texts])

    encoder_input_data, decoder_input_data, decoder_target_data, input_token_index,target_token_index  = tokenize_char(input_texts, target_texts)

    name_func = "_seq2seq_models_LSTM" + str(epochs)
    #model, encoder_model, decoder_model = seq2seq_models_LSTM(encoder_input_data, decoder_input_data, decoder_target_data, embedding = False, name_func = name_func)


    #load trained model
    name_func = "_seq2seq_models_LSTM" + str(epochs)
    model_dir = '/home/llu/HardDisk/LiangqunLuGitHub/DLForChatbot/MS_DL/results/seq2seq_char/'
    
    def load_model(model_filename, model_weights_filename):
        with open(model_dir + model_filename, 'r', encoding='utf8') as f:
            model = model_from_json(f.read())
        model.load_weights(model_dir + model_weights_filename)
        return model

    encoder_model = load_model(datatype + '_encoder' + name_func + '.json', datatype + '_encoder' + name_func + '.h5' )
    decoder_model = load_model(datatype + '_decoder' + name_func + '.json', datatype + '_decoder' + name_func + '.h5')

    # Reverse-lookup token index to decode sequences back to
    # something readable.

    reverse_input_char_index = dict((i, char) for char, i in input_token_index.items())
    reverse_target_char_index = dict((i, char) for char, i in target_token_index.items())
    num_encoder_tokens = encoder_input_data.shape[2]
    num_decoder_tokens = decoder_input_data.shape[2]
    
    index = randint(0, 100)
    #for seq_index in [index]:
    # Take one sequence (part of the training set)
    # for trying out decoding.
    seq_index = index
    input_seq = encoder_input_data[seq_index: seq_index + 1]
    target_seq = decoder_input_data[seq_index: seq_index + 1]
    #target_seq = decoder_target_data[seq_index: seq_index + 1] 
    #predict_sentence, decoded_tokens = decode_sequence(input_seq)
    predict_sentence = decode_sequence(input_seq)
    
    print('###############################')
    print('Input sentence:', input_texts[seq_index])
    print('Target sentence:', target_texts[seq_index]) 
    print('Predicted sentence:', predict_sentence)
    #print('Input tokens:', input_seq)        
    #print('Target tokens:', target_seq)
    #print('Predicted tokens:', decoded_tokens)
    
    print(seq_index, input_seq.shape, target_seq.shape)
    
    #target_seq = np.squeeze(target_seq, axis=0)
    #print(input_seq.shape, output_array.shape, target_seq.shape )    
    # target_seq.shape Out[107]: (1, 1, 2336)
    #cosine_smi = 1 - sp.distance.cdist(target_seq, decoded_tokens, 'cosine')[0][0]
    cosine_smi = 0
    
    #dist = (len(np.unique(decoded_tokens)) - 1)/(len(np.unique(target_seq)) - 1)    # unique token percentage
    
    input_sentence11 = input_texts[seq_index]
    #input_sentence11 = input_sentence11.replace(' COMMA ', ', ')    
    
    target_seq11 = target_texts[seq_index]
    predict_sentence11 = predict_sentence  
    
    target_seq11 = target_seq11.replace('\t', '')
    #target_seq11 = target_seq11.replace(' _END', '')
    target_seq11 = target_seq11.replace('\n', ', ')
    
    predict_sentence11 = predict_sentence11.replace('\t', '')   
    predict_sentence11 = predict_sentence11.replace('\n', '')    
    #predict_sentence11 = predict_sentence11.replace(' COMMA ', ', ')
    
    sentence1 = target_seq11.split()
    sentence2 = predict_sentence11.split()

    #Individual 1-gram
    #bleu_score = sentence_bleu(sentence1, sentence2, weights=(1, 0, 0, 0))
    
    #Individual ROUGE 
    #rouge = Rouge()
    #scores = rouge.get_scores(hypothesis, reference)
    #rouge = rouge.get_scores(predict_sentence11, target_seq11, avg=True)
    #rouge = rouge['rouge-1']['f']
    
    #dist = len(sentence2)/len(sentence1)
    
    ## translate sentences
    #index = randint(0, 100)
    
    #X = train[:, 0][index]
    #true_seq = train[:, 1][index]
    
    #trainXX = encode_sequences(eng_tokenizer, eng_length, train[:, 0])[index]
    #source = trainXX.reshape((1, trainXX.shape[0]))
    #translation = predict_sequence(model, ger_tokenizer, source)
    
    #true_seq = train[:, 1][index]
    
    #print("Input English: ", train[:, 0][index])
    #print("Translated Chinese: ", translation)
    #print("True Chinese: ", true_seq)
    
    clear_session()
    
    #input_text = train[:, 0][index]
    
    #return(input_text, translation, true_seq)
    return(input_sentence11, predict_sentence11, target_seq11)


#DS_generation()

#seed_text, output, true_seq = DS_generation()

#print(seed_text, output)






