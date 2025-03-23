# config.py

config = {
    # Images directory
    'images_path': "D:\\CV_projects\\Image-Caption-Generator-Using-CNN-main\\train_val_data\\Flicker8k_Dataset\\",
    
    # Train and validation image lists
    'train_data_path': "D:\\CV_projects\\Image-Caption-Generator-Using-CNN-main\\train_val_data\\Flickr_8k.trainImages.txt",
    'val_data_path': "D:\\CV_projects\\Image-Caption-Generator-Using-CNN-main\\train_val_data\\Flickr_8k.devImages.txt",
    
    # Captions file
    'captions_path': "D:\\CV_projects\\Image-Caption-Generator-Using-CNN-main\\train_val_data\\Flickr8k.token.txt",
    
    # Tokenizer path
    'tokenizer_path': "D:\\CV_projects\\Image-Caption-Generator-Using-CNN-main\\model_data\\tokenizer.pkl",
    
    # Directory for model data (weights, checkpoints, etc.)
    'model_data_path': "D:\\CV_projects\\Image-Caption-Generator-Using-CNN-main\\model_data\\",
    
    # Path to a pre-trained model checkpoint (if you have one)
    'model_load_path': "D:\\CV_projects\\Image-Caption-Generator-Using-CNN-main\\model_data\\model_inceptionv3.keras",
    
    # Training hyperparameters
    'num_of_epochs': 20,
    'max_length': 30,  # Updated max_length to match the model's expected input length
    'batch_size': 64,
    
    # Beam search parameter
    'beam_search_k': 3,
    
    # Test data directory
    'test_data_path': "D:\\CV_projects\\Image-Caption-Generator-Using-CNN-main\\test_data\\",
    
    # Model type (e.g., inceptionv3 or vgg16)
    'model_type': 'inceptionv3',
    
    # Random seed
    'random_seed': 1035
}

# Define rnnConfig here as well.
rnnConfig = {
    'embedding_size': 300,
    'LSTM_units': 256,
    'dense_units': 256,
    'dropout': 0.3
}
