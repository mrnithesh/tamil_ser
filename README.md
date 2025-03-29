# Tamil Speech Emotion Recognition Project

Speech Emotion Recognition (SER) using Wav2Vec 2.0 for Tamil language audio analysis.

## Overview

This project implements a deep learning model for detecting emotions in Tamil speech using Facebook's Wav2Vec 2.0 architecture. The system is designed to classify audio into five emotional states: angry, fear, happy, neutral, and sad.

## Current Status of the Tamil Speech Emotion Recognition Project

1. **Model Training Completed (75%)**  
   - Trained **Wav2Vec 2.0** on **Tamil emotion datasets** (EMO-TA, Tamil movie dialogues).  
   - Achieved **good accuracy during training** but **failed in real-time** due to dataset limitations.  

2. **Challenges Faced**  
   - **Limited Dataset** → EMO-TA has **accent differences**, and Tamil movie dialogues are **exaggerated**, making real-world generalization difficult.  
   - **Overfitting** → Validation loss increased despite lower training loss, indicating poor generalization.  

3. **Proposed Solution**  
   - **Continuous Learning** → Collect real-world speech data over time.  
   - **Data Version Control (DVC)** → Track dataset evolution and improve model adaptability.  
   - **Automated Model Retraining (CI/CD Pipelines)** → Gradually enhance performance using new real-time data.  

4. **Next Steps**  
   - Implement **data collection pipelines** for real-world Tamil speech.  
   - Optimize the model to handle **diverse accents and spontaneous speech**.  
   - Deploy **DVC + CI/CD** for continuous model updates and better real-time accuracy.  

## Technical Details

### Model Architecture
- Base Model: facebook/wav2vec2-base
- Output Classes: 5 (angry, fear, happy, neutral, sad)
- Maximum Audio Duration: 3 seconds
- Sampling Rate: 16kHz

### Requirements
```bash
torch
numpy
pandas
librosa
transformers
scikit-learn
```

### Setup and Usage

1. **Environment Setup**
   ```bash
   pip install -r requirements.txt
   ```

2. **Data Preparation**
   - Place audio files in the `dataset` directory
   - Prepare CSV files:
     - `train_dataset.csv`
     - `test_dataset.csv`
   - Format: `audio_path,label`

3. **Training**
   ```python
   python main.py
   ```

4. **Inference**
   ```python
   # Example usage
   result = predict_emotion("path_to_audio.wav")
   print(f"Predicted emotion: {result['emotion']}")
   ```

## Project Structure

```
tamil_ser/
├── dataset/                  # Audio data directory
├── main.ipynb               # Main training notebook
├── train_dataset.csv        # Training data annotations
├── test_dataset.csv         # Testing data annotations
└── ser_results/             # Saved model checkpoints
```



## Acknowledgments

- Facebook AI Research for Wav2Vec 2.0
- Contributors to the EMO-TA dataset
