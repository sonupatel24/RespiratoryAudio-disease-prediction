#!/usr/bin/env python3
"""
Training script for Respiratory Disease LSTM Model
Run this script to train the model on the respiratory audio dataset
"""

import os
import sys
import argparse
from lstm_model import RespiratoryAudioLSTM

def main():
    parser = argparse.ArgumentParser(description='Train Respiratory Disease LSTM Model')
    parser.add_argument('--audio_dir', type=str, 
                       default='Respiratory_Sound_Database/Respiratory_Sound_Database/audio_and_txt_files',
                       help='Path to audio files directory')
    parser.add_argument('--annotations', type=str, 
                       default='data/annotations.csv',
                       help='Path to annotations CSV file')
    parser.add_argument('--diagnosis', type=str, 
                       default='Respiratory_Sound_Database/Respiratory_Sound_Database/patient_diagnosis.csv',
                       help='Path to patient diagnosis CSV file')
    parser.add_argument('--test_size', type=float, default=0.2,
                       help='Test set size (default: 0.2)')
    parser.add_argument('--random_state', type=int, default=42,
                       help='Random state for reproducibility (default: 42)')
    
    args = parser.parse_args()
    
    # Check if files exist
    if not os.path.exists(args.audio_dir):
        print(f"Error: Audio directory not found: {args.audio_dir}")
        sys.exit(1)
    
    if not os.path.exists(args.annotations):
        print(f"Error: Annotations file not found: {args.annotations}")
        sys.exit(1)
    
    if not os.path.exists(args.diagnosis):
        print(f"Error: Diagnosis file not found: {args.diagnosis}")
        sys.exit(1)
    
    print("=" * 60)
    print("RESPIRATORY DISEASE LSTM MODEL TRAINING")
    print("=" * 60)
    print(f"Audio directory: {args.audio_dir}")
    print(f"Annotations file: {args.annotations}")
    print(f"Diagnosis file: {args.diagnosis}")
    print(f"Test size: {args.test_size}")
    print(f"Random state: {args.random_state}")
    print("=" * 60)
    
    # Initialize model
    lstm_model = RespiratoryAudioLSTM(
        audio_dir=args.audio_dir,
        annotations_file=args.annotations,
        patient_diagnosis_file=args.diagnosis
    )
    
    try:
        # Train the model
        print("Starting training...")
        history = lstm_model.train(
            test_size=args.test_size,
            random_state=args.random_state
        )
        
        # Save the model
        print("\nSaving model...")
        lstm_model.save_model()
        
        print("\n" + "=" * 60)
        print("TRAINING COMPLETED SUCCESSFULLY!")
        print("=" * 60)
        print("Model saved as: respiratory_lstm_model.h5")
        print("Label encoder saved as: label_encoder.pkl")
        print("Training plots saved as: training_history.png, confusion_matrix.png")
        print("\nYou can now run the Streamlit app with: streamlit run streamlit_app.py")
        
    except Exception as e:
        print(f"\nError during training: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
