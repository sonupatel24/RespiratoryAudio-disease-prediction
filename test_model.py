#!/usr/bin/env python3
"""
Test script for the trained Respiratory Disease LSTM Model
"""

import os
import numpy as np
import librosa
import tensorflow as tf
import joblib
from lstm_model import RespiratoryAudioLSTM

def test_model():
    """Test the trained model with sample predictions"""
    
    # Check if model files exist
    model_path = 'respiratory_lstm_model.h5'
    encoder_path = 'label_encoder.pkl'
    
    if not os.path.exists(model_path):
        print(f"Error: Model file not found: {model_path}")
        print("Please train the model first using: python train_model.py")
        return
    
    if not os.path.exists(encoder_path):
        print(f"Error: Label encoder file not found: {encoder_path}")
        print("Please train the model first using: python train_model.py")
        return
    
    print("Loading trained model...")
    
    # Initialize model
    audio_dir = "Respiratory_Sound_Database/Respiratory_Sound_Database/audio_and_txt_files"
    annotations_file = "data/annotations.csv"
    patient_diagnosis_file = "Respiratory_Sound_Database/Respiratory_Sound_Database/patient_diagnosis.csv"
    
    lstm_model = RespiratoryAudioLSTM(audio_dir, annotations_file, patient_diagnosis_file)
    
    # Load the trained model
    lstm_model.load_model()
    
    print("Model loaded successfully!")
    print(f"Available classes: {lstm_model.label_encoder.classes_}")
    
    # Test with a sample audio file
    test_files = [
        "101_1b1_Al_sc_Meditron.wav",
        "102_1b1_Ar_sc_Meditron.wav",
        "121_1b1_Tc_sc_Meditron.wav"
    ]
    
    print("\nTesting with sample audio files...")
    
    for filename in test_files:
        audio_path = os.path.join(audio_dir, filename)
        if os.path.exists(audio_path):
            print(f"\nTesting: {filename}")
            predicted_class, confidence = lstm_model.predict(audio_path)
            
            if predicted_class is not None:
                print(f"Predicted Disease: {predicted_class}")
                print(f"Confidence: {confidence:.2%}")
            else:
                print("Error: Could not process audio file")
        else:
            print(f"File not found: {filename}")
    
    print("\nModel test completed!")

if __name__ == "__main__":
    test_model()
