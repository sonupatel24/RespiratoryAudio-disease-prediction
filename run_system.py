#!/usr/bin/env python3
"""
Main runner script for the Respiratory Disease Prediction System
This script checks prerequisites and guides you through running the system
"""

import os
import sys
import subprocess
import platform

def print_header():
    print("=" * 70)
    print("🫁 RESPIRATORY DISEASE PREDICTION SYSTEM")
    print("=" * 70)
    print("Deep Learning LSTM Model for Audio-based Disease Classification")
    print("=" * 70)

def check_python_version():
    """Check if Python version is compatible"""
    print("\n🔍 Checking Python version...")
    if sys.version_info < (3, 8):
        print("❌ Error: Python 3.8 or higher is required")
        print(f"   Current version: {sys.version}")
        return False
    print(f"✅ Python version: {sys.version.split()[0]}")
    return True

def check_dependencies():
    """Check if required packages are installed"""
    print("\n🔍 Checking dependencies...")
    required_packages = [
        'tensorflow', 'librosa', 'numpy', 'pandas', 
        'scikit-learn', 'matplotlib', 'seaborn', 'streamlit', 'plotly'
    ]
    
    missing_packages = []
    for package in required_packages:
        try:
            __import__(package)
            print(f"✅ {package}")
        except ImportError:
            missing_packages.append(package)
            print(f"❌ {package}")
    
    if missing_packages:
        print(f"\n⚠️  Missing packages: {', '.join(missing_packages)}")
        print("   Run: pip install -r requirements.txt")
        return False
    
    return True

def check_dataset():
    """Check if dataset files exist"""
    print("\n🔍 Checking dataset...")
    required_files = [
        "data/annotations.csv",
        "Respiratory_Sound_Database/Respiratory_Sound_Database/patient_diagnosis.csv",
        "Respiratory_Sound_Database/Respiratory_Sound_Database/audio_and_txt_files"
    ]
    
    all_exist = True
    for file_path in required_files:
        if os.path.exists(file_path):
            print(f"✅ {file_path}")
        else:
            print(f"❌ {file_path}")
            all_exist = False
    
    if not all_exist:
        print("\n⚠️  Dataset files missing. Please ensure the dataset is properly organized.")
        return False
    
    return True

def check_model():
    """Check if model is trained"""
    print("\n🔍 Checking trained model...")
    model_files = ["respiratory_lstm_model.h5", "label_encoder.pkl"]
    
    model_exists = all(os.path.exists(f) for f in model_files)
    
    if model_exists:
        print("✅ Trained model found")
        return True
    else:
        print("❌ Trained model not found")
        print("   You need to train the model first")
        return False

def install_dependencies():
    """Install required dependencies"""
    print("\n📦 Installing dependencies...")
    try:
        # First try the stable requirements
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements_stable.txt"])
        print("✅ Dependencies installed successfully")
        return True
    except subprocess.CalledProcessError:
        print("⚠️  Stable requirements failed, trying flexible requirements...")
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
            print("✅ Dependencies installed successfully")
            return True
        except subprocess.CalledProcessError:
            print("⚠️  Requirements file failed, trying individual installation...")
            try:
                subprocess.check_call([sys.executable, "install_dependencies.py"])
                return True
            except subprocess.CalledProcessError as e:
                print(f"❌ Error installing dependencies: {e}")
                return False

def train_model():
    """Train the LSTM model"""
    print("\n🤖 Training LSTM model...")
    print("   This may take 10-30 minutes depending on your computer...")
    try:
        subprocess.check_call([sys.executable, "train_model.py"])
        print("✅ Model training completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Error training model: {e}")
        return False

def run_streamlit_app():
    """Run the Streamlit web application"""
    print("\n🌐 Starting Streamlit web application...")
    print("   The app will open in your browser at http://localhost:8501")
    print("   Press Ctrl+C to stop the application")
    try:
        subprocess.run([sys.executable, "-m", "streamlit", "run", "streamlit_app.py"])
    except KeyboardInterrupt:
        print("\n👋 Application stopped by user")
    except Exception as e:
        print(f"❌ Error running Streamlit app: {e}")

def main():
    print_header()
    
    # Check Python version
    if not check_python_version():
        sys.exit(1)
    
    # Check dependencies
    if not check_dependencies():
        print("\n🔧 Would you like to install missing dependencies? (y/n): ", end="")
        if input().lower().startswith('y'):
            if not install_dependencies():
                sys.exit(1)
        else:
            sys.exit(1)
    
    # Check dataset
    if not check_dataset():
        print("\n📁 Please ensure the dataset is properly organized as described in README.md")
        sys.exit(1)
    
    # Check if model is trained
    if not check_model():
        print("\n🤖 Would you like to train the model now? (y/n): ", end="")
        if input().lower().startswith('y'):
            if not train_model():
                sys.exit(1)
        else:
            print("❌ Cannot run the application without a trained model")
            sys.exit(1)
    
    # All checks passed, run the application
    print("\n🎉 All checks passed! Starting the application...")
    run_streamlit_app()

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n👋 Goodbye!")
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        sys.exit(1)
