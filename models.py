import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, RepeatVector, TimeDistributed, Dropout
from tensorflow.keras.activations import elu, relu
from attention_decoder import AttentionDecoder
from tensorflow import keras
# from bert import BertModelLayer
# import bert


def get_model(algo, INPUT_LEN, TARGET_LEN, dim):
    if algo == 'lstm':
        model = get_lstm(INPUT_LEN, dim)
    elif algo == 'seq':
        model = get_seq2seq(INPUT_LEN, TARGET_LEN, dim)
    elif algo == 'seq_at':
        # print("need to be added")
        #model = get_seq2seq_attention(INPUT_LEN, dim) # does not work with tf 2+
    # elif algo == 'albert':
    #     model = get_albert(INPUT_LEN, dim)

    return model


def get_lstm(INPUT_LEN, dim):
    ############ LSTM model #######################
    model = Sequential()
    model.add(LSTM(128, input_shape=(INPUT_LEN, dim), return_sequences=True))
    model.add(LSTM(128, input_shape=(INPUT_LEN, dim), return_sequences=True))
    model.add(LSTM(128, input_shape=(INPUT_LEN, dim), return_sequences=True))
    # model.add(Flatten())
    model.add(Dense(dim))
    model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])
    return model


def get_seq2seq(INPUT_LEN, TARGET_LEN, dim):
    hidden_size = 128
    model = Sequential()
    model.add(LSTM(hidden_size, input_shape=(INPUT_LEN, dim)))
    model.add(Dropout(0.2))
    model.add(RepeatVector(TARGET_LEN))
    model.add(LSTM(hidden_size, return_sequences=True))
    model.add(Dropout(0.2))
    model.add(Dense(hidden_size, activation=relu))
    # model.add(Dropout(0.2))
    model.add(TimeDistributed(Dense(dim, activation=relu)))

    model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])
    return model


# def get_seq2seq_attention(INPUT_LEN, dim):
#     hidden_size = 128
#     model = Sequential()
#     model.add(LSTM(hidden_size, input_shape=(INPUT_LEN, dim), return_sequences=True))
#     model.add(LSTM(hidden_size, input_shape=(INPUT_LEN, dim), return_sequences=True))
#     model.add(LSTM(hidden_size, input_shape=(INPUT_LEN, dim), return_sequences=True))
#     model.add(AttentionDecoder(hidden_size, dim))

#     model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])
#     return model


# def get_albert(INPUT_LEN, dim):
#     model_name = "albert_base"
#     model_dir = bert.fetch_tfhub_albert_model(model_name, ".models")
#     model_params = bert.albert_params(model_name)
#     l_bert = bert.BertModelLayer.from_params(model_params, name="albert")
#
#     max_seq_len = 128 # INPUT_LEN*dim
#     l_input_ids = keras.layers.Input(shape=(max_seq_len,), dtype='int32')
#     l_token_type_ids = keras.layers.Input(shape=(max_seq_len,), dtype='int32')
#
#     # using the default token_type/segment id 0
#     output = l_bert(l_input_ids)  # output: [batch_size, max_seq_len, hidden_size]
#     model = keras.Model(inputs=l_input_ids, outputs=output)
#     model.build(input_shape=(None, max_seq_len))

    # # # provide a custom token_type/segment id as a layer input
    # # output = l_bert([l_input_ids, l_token_type_ids])  # [batch_size, max_seq_len, hidden_size]
    # # model = keras.Model(inputs=[l_input_ids, l_token_type_ids], outputs=output)
    # # model.build(input_shape=[(None, max_seq_len), (None, max_seq_len)])
    #
    # model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])
    # return model
