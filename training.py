import numpy as np
import random
import os
from pickle import load
from utils.model import *  
from utils.load_data import loadTrainData, loadValData, data_generator
from tensorflow.keras.callbacks import ModelCheckpoint
from config import config, rnnConfig

required_keys = [
    'images_path',
    'train_data_path',
    'val_data_path',
    'captions_path',
    'tokenizer_path',
    'model_data_path',
    'model_load_path',
    'num_of_epochs',
    'max_length',
    'batch_size',
    'beam_search_k',
    'test_data_path',
    'model_type',
    'random_seed'
]
for key in required_keys:
    if key not in config:
        raise ValueError(f"Missing required config key: '{key}'")

assert isinstance(config['num_of_epochs'], int), 'num_of_epochs must be an integer'
assert isinstance(config['max_length'], int), 'max_length must be an integer'
assert isinstance(config['batch_size'], int), 'batch_size must be an integer'
assert isinstance(config['beam_search_k'], int), 'beam_search_k must be an integer'
assert isinstance(config['random_seed'], int), 'random_seed must be an integer'
assert isinstance(rnnConfig['embedding_size'], int), 'embedding_size must be an integer'
assert isinstance(rnnConfig['LSTM_units'], int), 'LSTM_units must be an integer'
assert isinstance(rnnConfig['dense_units'], int), 'dense_units must be an integer'
assert isinstance(rnnConfig['dropout'], float), 'dropout must be a float'

random.seed(config['random_seed'])

if not os.path.isdir(config['model_data_path']):
    print(f"Directory '{config['model_data_path']}' does not exist. Creating it...")
    os.makedirs(config['model_data_path'])

X1train, X2train, max_length = loadTrainData(config)
X1val, X2val = loadValData(config)

tokenizer_path = config['tokenizer_path']
with open(tokenizer_path, 'rb') as f:
    tokenizer = load(f)

vocab_size = len(tokenizer.word_index) + 1

model = AlternativeRNNModel(vocab_size, max_length, rnnConfig, config['model_type'])
print('RNN Model (Decoder) Summary:')
print(model.summary())

num_of_epochs = config['num_of_epochs']
batch_size = config['batch_size']
steps_train = len(X2train) // batch_size
if len(X2train) % batch_size != 0:
    steps_train += 1
steps_val = len(X2val) // batch_size
if len(X2val) % batch_size != 0:
    steps_val += 1

model_save_path = os.path.join(
    config['model_data_path'],
    f"model_{config['model_type']}.keras"
)

if os.path.exists(model_save_path):
    print(f"WARNING: The model file '{model_save_path}' already exists and will be overwritten.")

checkpoint = ModelCheckpoint(
    filepath=model_save_path,
    monitor='val_loss',
    verbose=1,
    save_best_only=False,
    mode='min'
)
callbacks = [checkpoint]

print('steps_train: {}, steps_val: {}'.format(steps_train, steps_val))
print('Batch Size:', batch_size)
print('Total Number of Epochs =', num_of_epochs)

ids_train = list(X2train.keys())
random.shuffle(ids_train)
X2train = {_id: X2train[_id] for _id in ids_train}

generator_train = data_generator(X1train, X2train, tokenizer, max_length, batch_size, config['random_seed'])
generator_val = data_generator(X1val, X2val, tokenizer, max_length, batch_size, config['random_seed'])

model.fit(
    generator_train,
    epochs=num_of_epochs,
    steps_per_epoch=steps_train,
    validation_data=generator_val,
    validation_steps=steps_val,
    callbacks=callbacks,
    verbose=1
)

print(
    f"Model trained successfully. Running model on validation set for calculating BLEU score "
    f"using BEAM search with k={config['beam_search_k']}"
)
evaluate_model_beam_search(model, X1val, X2val, tokenizer, max_length, beam_index=config['beam_search_k'])
