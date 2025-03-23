# Image Captioning using CNN & LSTM

This repository implements an image captioning system that uses a pre-trained Convolutional Neural Network (CNN) for feature extraction and a Long Short-Term Memory network (LSTM) for sequence generation. The model extracts deep visual features from images (using models such as InceptionV3 or VGG16) and then generates descriptive natural language captions.

## Features

- **CNN Feature Extraction:**  
  Uses pre-trained CNNs (e.g., InceptionV3 or VGG16) to extract rich visual features from images.
  
- **LSTM Decoder:**  
  An LSTM-based decoder generates captions word-by-word, using teacher forcing during training.
  
- **Beam Search Inference:**  
  Implements beam search to improve caption quality during inference.
  
- **Custom Loss & Evaluation:**  
  Employs a masked loss function to ignore padded tokens and computes BLEU scores (with smoothing) for evaluation.

## Getting Started

### Prerequisites

- Python 3.7+
- TensorFlow 2.x and Keras
- NumPy
- Pillow
- NLTK
- tqdm

Install the dependencies via pip:

```bash
pip install tensorflow keras numpy pillow nltk tqdm
