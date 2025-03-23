Image Captioning using CNN & LSTM
This repository implements an image captioning system that uses a pre-trained Convolutional Neural Network (CNN) for feature extraction and a Long Short-Term Memory network (LSTM) for sequence generation. The model extracts rich visual features from images (using InceptionV3 or VGG16) and then generates natural language captions that describe the image.

Features
CNN Feature Extraction:
Leverages pre-trained models (e.g., InceptionV3 or VGG16) to extract deep image features.

LSTM Decoder:
Uses an LSTM network to generate captions word-by-word, with support for teacher forcing during training.

Beam Search Inference:
Implements beam search for improved caption generation during inference.

Custom Loss & Evaluation:
Utilizes a masked loss function to handle padded tokens and computes BLEU scores (with smoothing) to evaluate caption quality.

Getting Started
Prerequisites
Python 3.7+

TensorFlow 2.x and Keras

Numpy

Pillow

NLTK

tqdm

Install the dependencies using pip:

bash
Copy
Edit
pip install tensorflow keras numpy pillow nltk tqdm
Data Preparation
Images:
Place your image dataset (e.g., Flickr8k) in the directory specified in config.py.

Captions:
Ensure that your caption file (e.g., Flickr8k.token.txt) is available and update its path in config.py.

Preprocessing:
Run the provided preprocessing scripts to generate cleaned captions and extracted image features.

Training
Run the training script:

bash
Copy
Edit
python train_val.py
This script:

Loads and preprocesses the image and caption data.

Defines and trains the CNN+LSTM model.

Saves a single final model in the Keras format.

Evaluates the model using BLEU scores.

Inference
After training, generate captions for new images by running:

bash
Copy
Edit
python test.py
This script loads the trained model, extracts image features, and generates captions using beam search.
