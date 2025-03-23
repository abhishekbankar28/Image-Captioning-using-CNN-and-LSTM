import numpy as np
from utils.preprocessing import *
from pickle import load, dump
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
import random

def load_set(filename):
    file = open(filename, 'r')
    doc = file.read()
    file.close()
    ids = list()
    # Process line by line
    for line in doc.split('\n'):
        # Skip empty lines
        if len(line) < 1:
            continue
        # Get the image identifier(id)
        _id = line.split('.')[0]
        ids.append(_id)
    return set(ids)

def load_cleaned_captions(filename, ids):
    file = open(filename, 'r')
    doc = file.read()
    file.close()
    captions = dict()
    _count = 0
    # Process line by line
    for line in doc.split('\n'):
        # Split line on white space
        tokens = line.split()
        # Skip empty lines
        if len(tokens) < 2:
            continue
        # Split id from caption
        image_id, image_caption = tokens[0], tokens[1:]
        # Skip images not in the ids set
        if image_id in ids:
            # Create list
            if image_id not in captions:
                captions[image_id] = list()
            # Wrap caption in start & end tokens
            caption = 'startseq ' + ' '.join(image_caption) + ' endseq'
            # Store
            captions[image_id].append(caption)
            _count = _count+1
    return captions, _count

# Load image features
def load_image_features(filename, ids):
    # load all features
    all_features = load(open(filename, 'rb'))
    # filter features
    features = {_id: all_features[_id] for _id in ids}
    return features

# Convert a dictionary to a list
def to_lines(captions):
    all_captions = list()
    for image_id in captions.keys():
        [all_captions.append(caption) for caption in captions[image_id]]
    return all_captions

def create_tokenizer(captions):
    lines = to_lines(captions)
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(lines)
    return tokenizer

# Calculate the length of the captions with the most words
def calc_max_length(captions):
    lines = to_lines(captions)
    return max(len(line.split()) for line in lines)

# Modified create_sequences function to match the model's expected input/output shape
def create_sequences(tokenizer, max_length, captions_list, image):
    X1, X2, y = list(), list(), list()
    # For each caption, generate one training sample (full sequence)
    for caption in captions_list:
        seq = tokenizer.texts_to_sequences([caption])[0]
        # Use teacher forcing: Input is caption tokens except the last, target is caption tokens except the first.
        # This assumes your captions always start with a special token (e.g., "startseq") and end with "endseq".
        in_seq = pad_sequences([seq[:-1]], maxlen=max_length, padding='post')[0]
        out_seq = pad_sequences([seq[1:]], maxlen=max_length, padding='post')[0]
        X1.append(image)
        X2.append(in_seq)
        y.append(out_seq)
    return X1, X2, y



# Fixed data generator that returns targets in the correct shape
def data_generator(images, captions, tokenizer, max_length, batch_size, random_seed):
    random.seed(random_seed)
    image_ids = list(captions.keys())
    _count = 0
    
    assert batch_size <= len(image_ids), 'Batch size must be less than or equal to {}'.format(len(image_ids))
    
    while True:
        if _count >= len(image_ids):
            _count = 0
            
        input_img_batch, input_sequence_batch, output_sequence_batch = list(), list(), list()
        
        for i in range(_count, min(len(image_ids), _count + batch_size)):
            image_id = image_ids[i]
            image = images[image_id][0]
            captions_list = captions[image_id]
            random.shuffle(captions_list)
            # Generate one training sample per caption (you could also average over multiple if desired)
            input_img, input_seq, output_seq = create_sequences(tokenizer, max_length, captions_list, image)
            input_img_batch.extend(input_img)
            input_sequence_batch.extend(input_seq)
            output_sequence_batch.extend(output_seq)
        
        _count += batch_size
        
        yield ((np.array(input_img_batch), np.array(input_sequence_batch)),
               np.array(output_sequence_batch))


def loadTrainData(config):
    train_image_ids = load_set(config['train_data_path'])
    preprocessData(config)
    train_captions, _count = load_cleaned_captions(config['model_data_path']+'captions.txt', train_image_ids)
    train_image_features = load_image_features(config['model_data_path']+'features_'+str(config['model_type'])+'.pkl', train_image_ids)
    print('{}: Available images for training: {}'.format(mytime(), len(train_image_features)))
    print('{}: Available captions for training: {}'.format(mytime(), _count))
    if not os.path.exists(config['model_data_path']+'tokenizer.pkl'):
        tokenizer = create_tokenizer(train_captions)
        dump(tokenizer, open(config['model_data_path']+'tokenizer.pkl', 'wb'))
    max_length = calc_max_length(train_captions)
    # Cap the maximum caption length to avoid large tensors
    MAX_CAPTION_LENGTH = 30  # adjust this value as needed
    max_length = min(max_length, MAX_CAPTION_LENGTH)
    return train_image_features, train_captions, max_length


def loadValData(config):
    val_image_ids = load_set(config['val_data_path'])
    # Load captions
    val_captions, _count = load_cleaned_captions(config['model_data_path']+'captions.txt', val_image_ids)
    # Load image features
    val_features = load_image_features(config['model_data_path']+'features_'+str(config['model_type'])+'.pkl', val_image_ids)
    print('{}: Available images for validation: {}'.format(mytime(),len(val_features)))
    print('{}: Available captions for validation: {}'.format(mytime(),_count))
    return val_features, val_captions