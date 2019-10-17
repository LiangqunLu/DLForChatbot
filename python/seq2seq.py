#!

'''
make use of three python models: tensorflow, keras and skopt
train a model in seq2seq in Keras
'''
from __future__ import print_function

from keras.models import Model
from keras.layers import Input, LSTM, Dense
from keras.optimizers import Adam
import numpy as np
from keras.callbacks import TensorBoard
from keras import backend as K
from keras.models import load_model

#hyperparameters
import skopt
from skopt import gp_minimize, forest_minimize
from skopt.space import Real, Categorical, Integer
from skopt.plots import plot_convergence
from skopt.plots import plot_objective, plot_evaluations
from skopt.plots import plot_histogram, plot_objective_2D
from skopt.utils import use_named_args

batch_size = 64  # Batch size for training.
epochs = 10  # Number of epochs to train for.
latent_dim = 256  # Latent dimensionality of the encoding space.
num_samples = 10000  # Number of samples to train on.

dim_learning_rate = Real(low=1e-6, high=1e-2, prior='log-uniform', name='learning_rate')
#dim_num_dense_layers = Integer(low=1, high=5, name='num_dense_layers')
#dim_num_dense_nodes = Integer(low=5, high=512, name='num_dense_nodes')
latent_dim = Integer(low=16, high=128, name='latent_dim')
dim_activation = Categorical(categories=['relu', 'sigmoid', 'softmax'], name='activation')
                             
dimensions = [dim_learning_rate,
                latent_dim,
              dim_activation]
              
default_parameters = [1e-5, 32, 'softmax']              



##load data
def input_training_file(data_path = '../data/cmn.txt'):
    '''
    input file is seq2seq
    output is the files for encoder_input_data and decoder_target_data
    '''
    #data_path = '../data/cmn.txt'
    global encoder_input_data, decoder_input_data, decoder_target_data
    global num_encoder_tokens, num_decoder_tokens

    input_texts = []
    target_texts = []
    input_characters = set()
    target_characters = set()
    with open(data_path, 'r', encoding='utf-8') as f:
        lines = f.read().split('\n')
    for line in lines[: min(num_samples, len(lines) - 1)]:
        input_text, target_text = line.split('\t')
        # We use "tab" as the "start sequence" character
        # for the targets, and "\n" as "end sequence" character.
        target_text = '\t' + target_text + '\n'
        input_texts.append(input_text)
        target_texts.append(target_text)
        for char in input_text:
            if char not in input_characters:
                input_characters.add(char)
        for char in target_text:
            if char not in target_characters:
                target_characters.add(char)
    
    input_characters = sorted(list(input_characters))
    target_characters = sorted(list(target_characters))
    num_encoder_tokens = len(input_characters)
    num_decoder_tokens = len(target_characters)
    max_encoder_seq_length = max([len(txt) for txt in input_texts])
    max_decoder_seq_length = max([len(txt) for txt in target_texts])
    
    print('Number of samples:', len(input_texts))
    print('Number of unique input tokens:', num_encoder_tokens)
    print('Number of unique output tokens:', num_decoder_tokens)
    print('Max sequence length for inputs:', max_encoder_seq_length)
    print('Max sequence length for outputs:', max_decoder_seq_length)
    
    input_token_index = dict(
        [(char, i) for i, char in enumerate(input_characters)])
    target_token_index = dict(
        [(char, i) for i, char in enumerate(target_characters)])
    
    encoder_input_data = np.zeros(
        (len(input_texts), max_encoder_seq_length, num_encoder_tokens),
        dtype='float32')
    decoder_input_data = np.zeros(
        (len(input_texts), max_decoder_seq_length, num_decoder_tokens),
        dtype='float32')
    decoder_target_data = np.zeros(
        (len(input_texts), max_decoder_seq_length, num_decoder_tokens),
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
    
    print("encoder input data dim: ", encoder_input_data.shape)
    print("decoder input data dim: ", decoder_input_data.shape)
    print("output data dim: ", decoder_target_data.shape)

##create model
def create_model(learning_rate, latent_dim, activation):
    
    global num_encoder_tokens, num_decoder_tokens
    #latent_dim = 32
        
    encoder_inputs = Input(shape=(None, num_encoder_tokens))
    encoder = LSTM(latent_dim, return_state=True)
    encoder_outputs, state_h, state_c = encoder(encoder_inputs)
    encoder_states = [state_h, state_c]
    decoder_inputs = Input(shape=(None, num_decoder_tokens))
    decoder_lstm = LSTM(latent_dim, return_sequences=True, return_state=True)
    decoder_outputs, _, _ = decoder_lstm(decoder_inputs, initial_state=encoder_states)
    decoder_dense = Dense(num_decoder_tokens, activation = activation)
    decoder_outputs = decoder_dense(decoder_outputs)
    
    model = Model([encoder_inputs, decoder_inputs], decoder_outputs)
    
    model.compile(optimizer=Adam(lr=learning_rate), loss='categorical_crossentropy')

    return(model)
    
def log_dir_name(learning_rate, latent_dim, activation):

    # The dir-name for the TensorBoard log-dir.
    s = "./seq2seq_logs/lr_{0:.0e}_nodes_{1}_activation_{2}/"

    # Insert all the hyper-parameters in the dir-name.
    log_dir = s.format(learning_rate,
                       latent_dim,
                       activation)
    return log_dir

path_best_model = 'best_model_nmt.keras'
best_loss = 1000

##fitness
@use_named_args(dimensions=dimensions)
def fitness(learning_rate, latent_dim, activation):
    
    # Print the hyper-parameters.
    print('learning rate: {0:.1e}'.format(learning_rate))
    #print('num_dense_layers:', num_dense_layers)
    #print('num_dense_nodes:', num_dense_nodes)
    print('neurons in LSTM:', latent_dim)
    print('activation:', activation)
    print()
    
    # Create the neural network with these hyper-parameters.
    model = create_model(learning_rate=learning_rate,
                         latent_dim=latent_dim,
                         activation=activation)
    
    # Dir-name for the TensorBoard log-files.
    log_dir = log_dir_name(learning_rate, latent_dim, activation)

    callback_log = TensorBoard(
        log_dir=log_dir,
        histogram_freq=0,
        batch_size=32,
        write_graph=True,
        write_grads=False,
        write_images=False)
   
    # Use Keras to train the model.
    history = model.fit([encoder_input_data, decoder_input_data], decoder_target_data,
            batch_size=128,
            epochs=3,
            validation_split=0.2,
            #validation_data=validation_data,
            callbacks=[callback_log])

    # Get the classification accuracy on the validation-set
    # after the last training-epoch.
    #accuracy = history.history['val_acc'][-1]
    loss = history.history['loss'][-1]

    # Print the classification accuracy.
    print()
    print("Loss: {0:.3}".format(loss))
    print()

    # Save the model if it improves on the best-found performance.
    # We use the global keyword so we update the variable outside
    # of this function.
    global best_loss

    # If the classification accuracy of the saved model is improved ...
    if loss < best_loss:
        # Save the new model to harddisk.
        model.save("model/" + path_best_model + ".h5")
        # Update the classification accuracy.
        best_loss = loss

    # Delete the Keras model with these hyper-parameters from memory.
    del model
    
    # Clear the Keras session, otherwise it will keep adding new
    # models to the same TensorFlow graph each time we create
    # a model with a different set of hyper-parameters.
    K.clear_session()
    
    return(loss)

input_training_file(data_path = '../data/cmn.txt')
learning_rate, latent_dim, activation = default_parameters

fitness(x = default_parameters)

search_result = gp_minimize(func=fitness,
                            dimensions=dimensions,
                            acq_func='EI', # Expected Improvement.
                            n_calls=40,
                            x0=default_parameters)

#search_result.x
#sorted(zip(search_result.func_vals, search_result.x_iters))

# Define sampling models
#model = Model([encoder_inputs, decoder_inputs], decoder_outputs)

def seq2seq_model():

    encoder_inputs = Input(shape=(None, num_encoder_tokens))
    encoder = LSTM(latent_dim, return_state=True)
    encoder_outputs, state_h, state_c = encoder(encoder_inputs)
    encoder_states = [state_h, state_c]
    encoder_model = Model(encoder_inputs, encoder_states)

    decoder_state_input_h = Input(shape=(latent_dim,))
    decoder_state_input_c = Input(shape=(latent_dim,))
    decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]
    decoder_lstm = LSTM(latent_dim, return_sequences=True, return_state=True)
    decoder_outputs, state_h, state_c = decoder_lstm(decoder_inputs, initial_state=decoder_states_inputs)
    decoder_states = [state_h, state_c]
    decoder_dense = Dense(num_decoder_tokens, activation='softmax')

    decoder_outputs = decoder_dense(decoder_outputs)
    decoder_model = Model(
        [decoder_inputs] + decoder_states_inputs,
        [decoder_outputs] + decoder_states)
        
    return(encoder_model, decoder_model)    


reverse_input_char_index = dict(
    (i, char) for char, i in input_token_index.items())
reverse_target_char_index = dict(
    (i, char) for char, i in target_token_index.items())

def decode_sequence(input_seq):

    encoder_model = seq2seq_model[0]
    decoder_model = seq2seq_model[1]

    states_value = encoder_model.predict(input_seq)
    # Generate empty target sequence of length 1.
    target_seq = np.zeros((1, 1, num_decoder_tokens))
    # Populate the first character of target sequence with the start character.
    target_seq[0, 0, target_token_index['\t']] = 1.

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

    return(decoded_sentence)


#ouput = decode_sequence(input_seq)



def generate_seq():
    
    #model = load_model("model/" + path_best_model + ".h5")
    input_test = input_texts[100]
    response_test = target_texts[100]
    
    seq_index = 100
    
    input_seq = encoder_input_data[seq_index: seq_index + 1]
    decoded_sentence = decode_sequence(input_seq)
    print('-')
    print('Input sentence:', input_texts[seq_index])
    print('True response sentence:', target_texts[seq_index])
    print('Decoded sentence:', decoded_sentence)    



