import os
import numpy as np
import pandas as pd
import librosa
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend to avoid Tcl error
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

class RespiratoryAudioLSTM:
    def __init__(self, audio_dir, annotations_file, patient_diagnosis_file):
        self.audio_dir = audio_dir
        self.annotations_file = annotations_file
        self.patient_diagnosis_file = patient_diagnosis_file
        self.model = None
        self.label_encoder = LabelEncoder()
        self.scaler = None
        
    def extract_features(self, audio_path, sr=22050, n_mfcc=13, n_fft=2048, hop_length=512):
        """Extract MFCC features from audio file"""
        try:
            # Load audio file
            y, sr = librosa.load(audio_path, sr=sr)
            
            # Extract MFCC features
            mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc, n_fft=n_fft, hop_length=hop_length)
            
            # Transpose to get (time, features) format for LSTM
            mfccs = mfccs.T
            
            # Pad or truncate to fixed length
            max_length = 100  # Adjust based on your data
            if mfccs.shape[0] > max_length:
                mfccs = mfccs[:max_length]
            else:
                padding = max_length - mfccs.shape[0]
                mfccs = np.pad(mfccs, ((0, padding), (0, 0)), mode='constant')
            
            return mfccs
        except Exception as e:
            print(f"Error processing {audio_path}: {e}")
            return None
    
    def load_data(self):
        """Load and preprocess the dataset"""
        print("Loading annotations...")
        annotations_df = pd.read_csv(self.annotations_file)
        
        print("Loading patient diagnoses...")
        diagnosis_df = pd.read_csv(self.patient_diagnosis_file, names=['patient_id', 'diagnosis'])
        
        # Merge annotations with diagnoses
        annotations_df['patient_id'] = annotations_df['filename'].str.extract('(\d+)').astype(int)
        merged_df = annotations_df.merge(diagnosis_df, on='patient_id', how='left')
        
        print(f"Found {len(merged_df)} audio files")
        print(f"Diagnosis distribution:")
        print(merged_df['diagnosis'].value_counts())
        
        # Filter out files with missing diagnoses
        merged_df = merged_df.dropna(subset=['diagnosis'])
        
        print(f"After filtering: {len(merged_df)} files")
        
        # Extract features
        features = []
        labels = []
        valid_files = []
        
        print("Extracting features...")
        for idx, row in merged_df.iterrows():
            audio_path = os.path.join(self.audio_dir, row['filename'])
            if os.path.exists(audio_path):
                feature = self.extract_features(audio_path)
                if feature is not None:
                    features.append(feature)
                    labels.append(row['diagnosis'])
                    valid_files.append(row['filename'])
                else:
                    print(f"Skipping {row['filename']} due to processing error")
            else:
                print(f"File not found: {audio_path}")
        
        features = np.array(features)
        labels = np.array(labels)
        
        print(f"Final dataset shape: {features.shape}")
        print(f"Labels shape: {labels.shape}")
        
        return features, labels, valid_files
    
    def build_model(self, input_shape, num_classes):
        """Build LSTM model"""
        model = Sequential([
            LSTM(128, return_sequences=True, input_shape=input_shape),
            Dropout(0.3),
            BatchNormalization(),
            
            LSTM(64, return_sequences=True),
            Dropout(0.3),
            BatchNormalization(),
            
            LSTM(32, return_sequences=False),
            Dropout(0.3),
            BatchNormalization(),
            
            Dense(64, activation='relu'),
            Dropout(0.3),
            Dense(32, activation='relu'),
            Dropout(0.2),
            Dense(num_classes, activation='softmax')
        ])
        
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    def train(self, test_size=0.2, random_state=42):
        """Train the LSTM model"""
        # Load data
        X, y, filenames = self.load_data()
        
        # Check class distribution
        from collections import Counter
        class_counts = Counter(y)
        print(f"Class distribution: {dict(class_counts)}")
        
        # Filter out classes with less than 2 samples
        min_samples = 2
        valid_classes = [cls for cls, count in class_counts.items() if count >= min_samples]
        
        if len(valid_classes) < 2:
            print("❌ Error: Not enough classes with sufficient samples for training")
            print("   Need at least 2 classes with 2+ samples each")
            return None
        
        # Filter data to only include valid classes
        valid_indices = [i for i, label in enumerate(y) if label in valid_classes]
        X_filtered = X[valid_indices]
        y_filtered = y[valid_indices]
        
        print(f"Using {len(valid_classes)} classes: {valid_classes}")
        print(f"Filtered dataset size: {len(X_filtered)} samples")
        
        # Encode labels for filtered data
        y_encoded = self.label_encoder.fit_transform(y_filtered)
        num_classes = len(self.label_encoder.classes_)
        
        print(f"Number of classes: {num_classes}")
        print(f"Classes: {self.label_encoder.classes_}")
        
        # Split data (without stratification to avoid the error)
        try:
            X_train, X_test, y_train, y_test = train_test_split(
                X_filtered, y_encoded, test_size=test_size, random_state=random_state, stratify=y_encoded
            )
        except ValueError:
            # If stratification still fails, split without stratification
            print("⚠️  Using random split without stratification due to class imbalance")
            X_train, X_test, y_train, y_test = train_test_split(
                X_filtered, y_encoded, test_size=test_size, random_state=random_state
            )
        
        print(f"Training set shape: {X_train.shape}")
        print(f"Test set shape: {X_test.shape}")
        
        # Build model
        input_shape = (X_train.shape[1], X_train.shape[2])
        self.model = self.build_model(input_shape, num_classes)
        
        print("Model architecture:")
        self.model.summary()
        
        # Callbacks
        callbacks = [
            EarlyStopping(patience=10, restore_best_weights=True),
            ReduceLROnPlateau(factor=0.5, patience=5, min_lr=1e-7)
        ]
        
        # Train model
        print("Training model...")
        history = self.model.fit(
            X_train, y_train,
            validation_data=(X_test, y_test),
            epochs=100,
            batch_size=32,
            callbacks=callbacks,
            verbose=1
        )
        
        # Evaluate model
        print("\nEvaluating model...")
        test_loss, test_accuracy = self.model.evaluate(X_test, y_test, verbose=0)
        print(f"Test Accuracy: {test_accuracy:.4f}")
        
        # Predictions
        y_pred = self.model.predict(X_test)
        y_pred_classes = np.argmax(y_pred, axis=1)
        
        # Classification report - FIXED: Use only the classes that are actually present
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred_classes, labels=range(num_classes), target_names=self.label_encoder.classes_))
        
        # Plot training history
        self.plot_training_history(history)
        
        # Plot confusion matrix
        self.plot_confusion_matrix(y_test, y_pred_classes)
        
        return history
    
    def plot_training_history(self, history):
        """Plot training history"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # Plot accuracy
        ax1.plot(history.history['accuracy'], label='Training Accuracy')
        ax1.plot(history.history['val_accuracy'], label='Validation Accuracy')
        ax1.set_title('Model Accuracy')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Accuracy')
        ax1.legend()
        
        # Plot loss
        ax2.plot(history.history['loss'], label='Training Loss')
        ax2.plot(history.history['val_loss'], label='Validation Loss')
        ax2.set_title('Model Loss')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Loss')
        ax2.legend()
        
        plt.tight_layout()
        plt.savefig('training_history.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_confusion_matrix(self, y_true, y_pred):
        """Plot confusion matrix"""
        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=self.label_encoder.classes_,
                   yticklabels=self.label_encoder.classes_)
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.tight_layout()
        plt.savefig('confusion_matrix.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def predict(self, audio_path):
        """Predict diagnosis for a single audio file"""
        if self.model is None:
            raise ValueError("Model not trained yet. Call train() first.")
        
        # Extract features
        features = self.extract_features(audio_path)
        if features is None:
            return None, None
        
        # Reshape for prediction
        features = features.reshape(1, features.shape[0], features.shape[1])
        
        # Predict
        prediction = self.model.predict(features)
        predicted_class_idx = np.argmax(prediction[0])
        predicted_class = self.label_encoder.inverse_transform([predicted_class_idx])[0]
        confidence = prediction[0][predicted_class_idx]
        
        return predicted_class, confidence
    
    def save_model(self, model_path='respiratory_lstm_model.h5'):
        """Save the trained model"""
        if self.model is None:
            raise ValueError("Model not trained yet. Call train() first.")
        
        self.model.save(model_path)
        
        # Save label encoder
        import joblib
        joblib.dump(self.label_encoder, 'label_encoder.pkl')
        
        print(f"Model saved to {model_path}")
        print("Label encoder saved to label_encoder.pkl")
    
    def load_model(self, model_path='respiratory_lstm_model.h5'):
        """Load a trained model"""
        self.model = tf.keras.models.load_model(model_path)
        
        # Load label encoder
        import joblib
        self.label_encoder = joblib.load('label_encoder.pkl')
        
        print(f"Model loaded from {model_path}")

def main():
    # Initialize the model
    audio_dir = "Respiratory_Sound_Database/Respiratory_Sound_Database/audio_and_txt_files"
    annotations_file = "data/annotations.csv"
    patient_diagnosis_file = "Respiratory_Sound_Database/Respiratory_Sound_Database/patient_diagnosis.csv"
    
    lstm_model = RespiratoryAudioLSTM(audio_dir, annotations_file, patient_diagnosis_file)
    
    # Train the model
    print("Starting training...")
    history = lstm_model.train()
    
    # Save the model
    lstm_model.save_model()
    
    print("Training completed!")

if __name__ == "__main__":
    main()
