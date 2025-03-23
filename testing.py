import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import os
import pickle

from keras.preprocessing.image import load_img, img_to_array
from keras.models import load_model
from utils.model import masked_loss, CNNModel, generate_caption_beam_search
from config import config

# Simple configuration checks
assert isinstance(config['max_length'], int), 'Please provide an integer value for `max_length` parameter in config.py file'
assert isinstance(config['beam_search_k'], int), 'Please provide an integer value for `beam_search_k` parameter in config.py file'

# Function to extract features from an image using the CNN model
def extract_features(filename, model, model_type):
    if model_type == 'inceptionv3':
        from keras.applications.inception_v3 import preprocess_input
        target_size = (299, 299)
    elif model_type == 'vgg16':
        from keras.applications.vgg16 import preprocess_input
        target_size = (224, 224)
    # Load and resize image
    image = load_img(filename, target_size=target_size)
    # Convert image pixels to numpy array
    image = img_to_array(image)
    # Reshape for the model
    image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
    # Preprocess for the CNN model
    image = preprocess_input(image)
    # Extract features using the CNN model
    features = model.predict(image, verbose=0)
    return features

# Load the tokenizer
with open(config['tokenizer_path'], 'rb') as f:
    tokenizer = pickle.load(f)

# Use max_length from config (ensure config['max_length'] is updated to 30)
max_length = config['max_length']

# Load the captioning model with the custom loss function
caption_model = load_model(config['model_load_path'], custom_objects={'masked_loss': masked_loss})

# Load the CNN model for image feature extraction
image_model = CNNModel(config['model_type'])

# Loop over images in the test directory and generate captions
for image_file in os.listdir(config['test_data_path']):
    # Skip files already generated (output files)
    if image_file.startswith('output'):
        continue
    # Process only jpg/jpeg files
    if image_file.lower().endswith(('.jpg', '.jpeg')):
        print('Generating caption for {}'.format(image_file))
        # Extract image features
        image_path = os.path.join(config['test_data_path'], image_file)
        image = extract_features(image_path, image_model, config['model_type'])
        # Generate caption using beam search (beam index from config)
        generated_caption = generate_caption_beam_search(
            caption_model, tokenizer, image, max_length, beam_index=config['beam_search_k']
        )
        # Remove startseq and endseq tokens, then format caption text
        words = generated_caption.split()
        if len(words) > 2:
            caption = 'Caption: ' + words[1].capitalize() + ' ' + ' '.join(words[2:-1]) + '.'
        else:
            caption = generated_caption
        # Open and display the image with caption
        pil_im = Image.open(image_path, 'r')
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.axis('off')
        ax.imshow(np.asarray(pil_im), interpolation='nearest')
        ax.set_title("BEAM Search with k={}\n{}".format(config['beam_search_k'], caption),
                     fontdict={'fontsize': '20', 'fontweight': '40'})
        # Save the output image with the generated caption
        output_path = os.path.join(config['test_data_path'], 'output--' + image_file)
        plt.savefig(output_path)
        plt.close(fig)
