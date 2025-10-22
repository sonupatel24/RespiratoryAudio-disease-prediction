import streamlit as st
import numpy as np
import librosa
import tensorflow as tf
import joblib
import tempfile
import os
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd

# Set page config
st.set_page_config(
    page_title="Respiratory Disease Prediction",
    page_icon="🫁",
    layout="wide",
    initial_sidebar_state="expanded"
)

class RespiratoryDiseasePredictor:
    def __init__(self):
        self.model = None
        self.label_encoder = None
        self.load_model()
    
    def load_model(self):
        """Load the trained LSTM model and label encoder"""
        try:
            self.model = tf.keras.models.load_model('respiratory_lstm_model.h5')
            self.label_encoder = joblib.load('label_encoder.pkl')
            st.success("Model loaded successfully!")
        except Exception as e:
            st.error(f"Error loading model: {e}")
            st.info("Please make sure the model files (respiratory_lstm_model.h5 and label_encoder.pkl) are in the current directory.")
    
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
            max_length = 100
            if mfccs.shape[0] > max_length:
                mfccs = mfccs[:max_length]
            else:
                padding = max_length - mfccs.shape[0]
                mfccs = np.pad(mfccs, ((0, padding), (0, 0)), mode='constant')
            
            return mfccs, y, sr
        except Exception as e:
            st.error(f"Error processing audio file: {e}")
            return None, None, None
    
    def predict(self, audio_path):
        """Predict diagnosis for a single audio file"""
        if self.model is None or self.label_encoder is None:
            return None, None, None, None
        
        # Extract features
        features, audio_data, sr = self.extract_features(audio_path)
        if features is None:
            return None, None, None, None
        
        # Reshape for prediction
        features = features.reshape(1, features.shape[0], features.shape[1])
        
        # Predict
        prediction = self.model.predict(features)
        predicted_class_idx = np.argmax(prediction[0])
        predicted_class = self.label_encoder.inverse_transform([predicted_class_idx])[0]
        confidence = prediction[0][predicted_class_idx]
        
        # Get all class probabilities
        class_probabilities = prediction[0]
        class_names = self.label_encoder.classes_
        
        return predicted_class, confidence, class_probabilities, class_names

def main():
    st.title("🫁 Respiratory Disease Prediction System")
    st.markdown("---")
    
    # Initialize predictor
    predictor = RespiratoryDiseasePredictor()
    
    if predictor.model is None:
        st.stop()
    
    # Sidebar
    st.sidebar.header("📊 Model Information")
    st.sidebar.info("This LSTM model predicts respiratory diseases from audio recordings using MFCC features.")
    
    # Main content
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.header("🎵 Upload Audio File")
        uploaded_file = st.file_uploader(
            "Choose an audio file (.wav, .mp3, .m4a)",
            type=['wav', 'mp3', 'm4a'],
            help="Upload a respiratory audio recording for disease prediction"
        )
        
        if uploaded_file is not None:
            # Save uploaded file temporarily
            with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp_file:
                tmp_file.write(uploaded_file.getvalue())
                tmp_path = tmp_file.name
            
            # Display audio player
            st.audio(uploaded_file.getvalue(), format='audio/wav')
            
            # Predict button
            if st.button("🔍 Predict Disease", type="primary"):
                with st.spinner("Analyzing audio..."):
                    # Make prediction
                    predicted_class, confidence, class_probabilities, class_names = predictor.predict(tmp_path)
                    
                    if predicted_class is not None:
                        # Display results
                        st.success(f"Prediction completed!")
                        
                        # Main prediction result
                        st.markdown("### 🎯 Prediction Result")
                        col_pred, col_conf = st.columns(2)
                        
                        with col_pred:
                            st.metric("Predicted Disease", predicted_class)
                        
                        with col_conf:
                            st.metric("Confidence", f"{confidence:.2%}")
                        
                        # Confidence bar
                        st.progress(float(confidence))
                        
                        # Detailed probabilities
                        st.markdown("### 📈 Detailed Probabilities")
                        
                        # Create probability dataframe
                        prob_df = pd.DataFrame({
                            'Disease': class_names,
                            'Probability': class_probabilities
                        }).sort_values('Probability', ascending=False)
                        
                        # Create bar chart
                        fig = px.bar(
                            prob_df, 
                            x='Probability', 
                            y='Disease',
                            orientation='h',
                            title="Disease Probability Distribution",
                            color='Probability',
                            color_continuous_scale='RdYlBu_r'
                        )
                        fig.update_layout(
                            height=400,
                            yaxis={'categoryorder': 'total ascending'}
                        )
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Display table
                        st.markdown("### 📊 Probability Table")
                        st.dataframe(
                            prob_df.style.format({'Probability': '{:.2%}'}),
                            use_container_width=True
                        )
                        
                        # Audio analysis
                        st.markdown("### 🎵 Audio Analysis")
                        
                        # Extract features for visualization
                        features, audio_data, sr = predictor.extract_features(tmp_path)
                        
                        if audio_data is not None:
                            # Create subplots
                            fig = make_subplots(
                                rows=2, cols=1,
                                subplot_titles=('Waveform', 'MFCC Features'),
                                vertical_spacing=0.1
                            )
                            
                            # Waveform
                            time_axis = np.linspace(0, len(audio_data) / sr, len(audio_data))
                            fig.add_trace(
                                go.Scatter(x=time_axis, y=audio_data, name='Waveform'),
                                row=1, col=1
                            )
                            
                            # MFCC heatmap
                            fig.add_trace(
                                go.Heatmap(
                                    z=features.T,
                                    colorscale='Viridis',
                                    name='MFCC'
                                ),
                                row=2, col=1
                            )
                            
                            fig.update_layout(
                                height=600,
                                title_text="Audio Signal Analysis"
                            )
                            
                            st.plotly_chart(fig, use_container_width=True)
            
            # Clean up temporary file
            os.unlink(tmp_path)
    
    with col2:
        st.header("ℹ️ About")
        st.markdown("""
        **Respiratory Disease Prediction System**
        
        This system uses a deep learning LSTM model to predict respiratory diseases from audio recordings.
        
        **Supported Diseases:**
        - COPD
        - Healthy
        - URTI (Upper Respiratory Tract Infection)
        - LRTI (Lower Respiratory Tract Infection)
        - Pneumonia
        - Bronchiectasis
        - Bronchiolitis
        
        **How it works:**
        1. Upload an audio file
        2. The system extracts MFCC features
        3. LSTM model predicts the disease
        4. Results are displayed with confidence scores
        """)
        
        st.header("📋 Instructions")
        st.markdown("""
        1. **Upload**: Choose a respiratory audio file
        2. **Listen**: Verify the audio quality
        3. **Predict**: Click the predict button
        4. **Analyze**: Review the results and probabilities
        """)
        
        st.header("⚠️ Disclaimer")
        st.warning("""
        This is a research tool and should not be used for medical diagnosis. 
        Always consult with healthcare professionals for medical decisions.
        """)

if __name__ == "__main__":
    main()
