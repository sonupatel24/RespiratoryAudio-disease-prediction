#!/usr/bin/env python3
"""
Check and Train Script for Respiratory Disease Prediction System
This script checks if the model exists and trains it if needed
"""

import os
import sys
import subprocess

def print_header():
    print("=" * 70)
    print("🫁 RESPIRATORY DISEASE PREDICTION SYSTEM")
    print("🔍 MODEL CHECK AND TRAINING")
    print("=" * 70)

def check_model_files():
    """Check if model files exist"""
    print("\n🔍 Checking for trained model files...")
    
    model_files = {
        'respiratory_lstm_model.h5': 'LSTM Model',
        'label_encoder.pkl': 'Label Encoder'
    }
    
    missing_files = []
    for filename, description in model_files.items():
        if os.path.exists(filename):
            print(f"✅ {description}: {filename}")
        else:
            print(f"❌ {description}: {filename} (MISSING)")
            missing_files.append(filename)
    
    return len(missing_files) == 0, missing_files

def check_dataset():
    """Check if dataset files exist"""
    print("\n🔍 Checking dataset files...")
    
    required_files = [
        "data/annotations.csv",
        "Respiratory_Sound_Database/Respiratory_Sound_Database/patient_diagnosis.csv",
        "Respiratory_Sound_Database/Respiratory_Sound_Database/audio_and_txt_files"
    ]
    
    missing_files = []
    for file_path in required_files:
        if os.path.exists(file_path):
            print(f"✅ {file_path}")
        else:
            print(f"❌ {file_path} (MISSING)")
            missing_files.append(file_path)
    
    return len(missing_files) == 0, missing_files

def train_model():
    """Train the LSTM model"""
    print("\n🤖 Starting model training...")
    print("   This will take 10-30 minutes depending on your computer...")
    print("   Please be patient and don't close this window...")
    
    try:
        # Run the training script
        result = subprocess.run([sys.executable, "train_model.py"], 
                              capture_output=True, text=True, timeout=3600)  # 1 hour timeout
        
        if result.returncode == 0:
            print("✅ Model training completed successfully!")
            print("\nTraining output:")
            print(result.stdout)
            return True
        else:
            print("❌ Model training failed!")
            print("\nError output:")
            print(result.stderr)
            return False
            
    except subprocess.TimeoutExpired:
        print("❌ Training timed out (took longer than 1 hour)")
        return False
    except Exception as e:
        print(f"❌ Error during training: {e}")
        return False

def main():
    print_header()
    
    # Check if model files exist
    model_exists, missing_model_files = check_model_files()
    
    if model_exists:
        print("\n🎉 Model files found! You can now run the web application.")
        print("\nTo start the web app, run:")
        print("   streamlit run streamlit_app.py")
        print("\nOr run the complete system:")
        print("   python run_system.py")
        return
    
    # Check dataset
    dataset_ok, missing_dataset = check_dataset()
    
    if not dataset_ok:
        print("\n❌ Dataset files are missing!")
        print("Missing files:")
        for file in missing_dataset:
            print(f"   - {file}")
        print("\nPlease ensure the dataset is properly organized as described in README.md")
        return
    
    # Ask user if they want to train
    print(f"\n⚠️  Model files missing: {', '.join(missing_model_files)}")
    print("\nWould you like to train the model now? (y/n): ", end="")
    
    try:
        response = input().lower().strip()
    except KeyboardInterrupt:
        print("\n\n👋 Training cancelled by user")
        return
    
    if response.startswith('y'):
        print("\n🚀 Starting model training...")
        if train_model():
            print("\n" + "=" * 70)
            print("🎉 TRAINING COMPLETED SUCCESSFULLY!")
            print("=" * 70)
            print("\nYou can now:")
            print("1. Run the web app: streamlit run streamlit_app.py")
            print("2. Run the complete system: python run_system.py")
            print("3. Test the model: python test_model.py")
        else:
            print("\n❌ Training failed. Please check the error messages above.")
            print("\nTroubleshooting:")
            print("1. Make sure all dependencies are installed")
            print("2. Check that audio files exist in the dataset")
            print("3. Ensure you have enough disk space and RAM")
    else:
        print("\n❌ Cannot run the application without a trained model.")
        print("Please train the model first by running:")
        print("   python train_model.py")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n👋 Goodbye!")
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        sys.exit(1)
