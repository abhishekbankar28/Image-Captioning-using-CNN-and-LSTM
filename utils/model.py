import numpy as np
from tensorflow.keras.applications.inception_v3 import InceptionV3, preprocess_input
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Dropout, LSTM, Embedding, concatenate, RepeatVector, TimeDistributed, Bidirectional
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import GlobalAveragePooling2D
from nltk.translate.bleu_score import SmoothingFunction
from tensorflow.keras.utils import register_keras_serializable
import tensorflow as tf
from tqdm import tqdm
from nltk.translate.bleu_score import corpus_bleu

# Custom loss function for sequence prediction - FIXED
def masked_loss(y_true, y_pred):
    # y_true shape: (batch, time_steps)
    # y_pred shape: (batch, time_steps, vocab_size)
    y_true_flat = tf.reshape(y_true, [-1])
    y_pred_flat = tf.reshape(y_pred, [-1, tf.shape(y_pred)[-1]])
    loss = tf.keras.losses.sparse_categorical_crossentropy(y_true_flat, y_pred_flat)
    mask = tf.cast(tf.not_equal(y_true_flat, 0), dtype=tf.float32)
    loss *= mask
    return tf.reduce_sum(loss) / tf.reduce_sum(mask)




# CNN Model (unchanged)
def CNNModel(model_type):
    if model_type == 'inceptionv3':
        base_model = InceptionV3(weights='imagenet', include_top=True)
        model = Model(inputs=base_model.input, outputs=base_model.layers[-2].output)
    elif model_type == 'vgg16':
        base_model = VGG16(weights='imagenet')
        base_model.layers.pop()
        model = Model(inputs=base_model.inputs, outputs=base_model.layers[-1].output)
    return model

# RNN Model (Original) - Updated for sequence prediction
# In model.py, update your RNNModel and AlternativeRNNModel as follows:
def RNNModel(vocab_size, max_len, rnnConfig, model_type):
    embedding_size = rnnConfig['embedding_size']
    
    # Image branch
    if model_type == 'inceptionv3':
        image_input = Input(shape=(2048,))
    elif model_type == 'vgg16':
        image_input = Input(shape=(4096,))
    image_features = Dense(embedding_size, activation='relu')(image_input)
    image_features = Dropout(rnnConfig['dropout'])(image_features)
    # Repeat the image features across all time steps
    image_features = RepeatVector(max_len)(image_features)
    
    # Caption branch
    caption_input = Input(shape=(max_len,))
    # Use mask_zero=False since your loss function handles masking
    caption_features = Embedding(vocab_size, embedding_size, mask_zero=False)(caption_input)
    caption_features = LSTM(rnnConfig['LSTM_units'], return_sequences=True)(caption_features)
    caption_features = TimeDistributed(Dense(embedding_size))(caption_features)
    
    # Merge branches
    merged = concatenate([image_features, caption_features])
    merged = LSTM(rnnConfig['LSTM_units'], return_sequences=True)(merged)
    output = TimeDistributed(Dense(vocab_size, activation='softmax'))(merged)
    
    model = Model(inputs=[image_input, caption_input], outputs=output)
    model.compile(loss=masked_loss, optimizer='adam')
    return model




def AlternativeRNNModel(vocab_size, max_len, rnnConfig, model_type):
    embedding_size = rnnConfig['embedding_size']
    if model_type == 'inceptionv3':
        image_input = Input(shape=(2048,))
    elif model_type == 'vgg16':
        image_input = Input(shape=(4096,))
    
    image_model = Dense(embedding_size, activation='relu')(image_input)
    image_model = RepeatVector(max_len)(image_model)

    caption_input = Input(shape=(max_len,))
    # Disable automatic masking here as well
    caption_model = Embedding(vocab_size, embedding_size, mask_zero=False)(caption_input)
    caption_model = LSTM(rnnConfig['LSTM_units'], return_sequences=True)(caption_model)
    caption_model = TimeDistributed(Dense(embedding_size))(caption_model)

    merged = concatenate([image_model, caption_model])
    merged = Bidirectional(LSTM(rnnConfig['LSTM_units'], return_sequences=True))(merged)
    output = TimeDistributed(Dense(vocab_size, activation='softmax'))(merged)

    model = Model(inputs=[image_input, caption_input], outputs=output)
    model.compile(loss=masked_loss, optimizer='adam')
    return model


# Utility functions (updated for sequence prediction)
def int_to_word(integer, tokenizer):
    return tokenizer.index_word.get(integer, None)  # Handle unknown tokens

def generate_caption(model, tokenizer, image, max_length):
    in_text = 'startseq'
    for _ in range(max_length):
        sequence = tokenizer.texts_to_sequences([in_text])[0]
        sequence = pad_sequences([sequence], maxlen=max_length, padding='post')
        yhat = model.predict([image, sequence], verbose=0)
        yhat = np.argmax(yhat[0, -1, :])  # Get last timestep prediction
        word = int_to_word(yhat, tokenizer)
        if word is None or word == 'endseq':
            break
        in_text += ' ' + word
    return in_text.replace('startseq ', '').replace(' endseq', '').strip()

def generate_caption_beam_search(model, tokenizer, image, max_length, beam_index=3):
    in_text = [[tokenizer.texts_to_sequences(['startseq'])[0], 0.0]]
    while len(in_text[0][0]) < max_length:
        tempList = []
        for seq in in_text:
            padded_seq = pad_sequences([seq[0]], maxlen=max_length, padding='post')
            preds = model.predict([image, padded_seq], verbose=0)[0]  # Shape: (max_len, vocab_size)
            top_preds = np.argsort(preds[-1, :])[-beam_index:]  # Last timestep
            for word in top_preds:
                next_seq, prob = seq[0][:], seq[1]
                next_seq.append(word)
                prob += np.log(preds[-1, word])  # Use log probability
                tempList.append([next_seq, prob])
        in_text = sorted(tempList, key=lambda l: l[1])[-beam_index:]
    best_seq = in_text[-1][0]
    caption = [int_to_word(idx, tokenizer) for idx in best_seq if idx != 0]
    return ' '.join([w for w in caption if w not in ['startseq', 'endseq']])

def evaluate_model(model, images, captions, tokenizer, max_length):
    actual, predicted = list(), list()
    for image_id, caption_list in tqdm(captions.items()):
        yhat = generate_caption(model, tokenizer, images[image_id], max_length)
        ground_truth = [caption.split() for caption in caption_list]
        actual.append(ground_truth)
        predicted.append(yhat.split())
    print('BLEU Scores:')
    print('BLEU-1: %f' % corpus_bleu(actual, predicted, weights=(1.0, 0, 0, 0)))
    print('BLEU-2: %f' % corpus_bleu(actual, predicted, weights=(0.5, 0.5, 0, 0)))
    print('BLEU-3: %f' % corpus_bleu(actual, predicted, weights=(0.3, 0.3, 0.3, 0)))
    print('BLEU-4: %f' % corpus_bleu(actual, predicted, weights=(0.25, 0.25, 0.25, 0.25)))

def evaluate_model_beam_search(model, images, captions, tokenizer, max_length, beam_index=3):
    actual, predicted = list(), list()
    smoothing_function = SmoothingFunction().method1  # Use smoothing to avoid zero counts
    for image_id, caption_list in tqdm(captions.items()):
        yhat = generate_caption_beam_search(model, tokenizer, images[image_id], max_length, beam_index)
        ground_truth = [caption.split() for caption in caption_list]
        actual.append(ground_truth)
        predicted.append(yhat.split())
    print('BLEU Scores:')
    print('BLEU-1: %f' % corpus_bleu(actual, predicted, weights=(1.0, 0, 0, 0), smoothing_function=smoothing_function))
    print('BLEU-2: %f' % corpus_bleu(actual, predicted, weights=(0.5, 0.5, 0, 0), smoothing_function=smoothing_function))
    print('BLEU-3: %f' % corpus_bleu(actual, predicted, weights=(0.3, 0.3, 0.3, 0), smoothing_function=smoothing_function))
    print('BLEU-4: %f' % corpus_bleu(actual, predicted, weights=(0.25, 0.25, 0.25, 0.25), smoothing_function=smoothing_function))
