import os
import numpy as np
import librosa as lb
from keras.models import load_model

# Load trained model once (important for performance)
model = load_model('model/model.h5')

# ==============================
# Feature Extraction
# ==============================
def getFeaturesForNeuralNetwork(path):
    max_len = 259

    # Load audio
    soundArr, sample_rate = lb.load(path, mono=True)

    # MFCC (20 x time)
    mfcc = lb.feature.mfcc(y=soundArr, sr=sample_rate, n_mfcc=20)
    mfcc = lb.util.fix_length(mfcc, size=max_len, axis=1)

    # Chroma (12 x time)
    croma = lb.feature.chroma_stft(y=soundArr, sr=sample_rate)
    croma = lb.util.fix_length(croma, size=max_len, axis=1)

    # MelSpectrogram (128 x time)
    mspec = lb.feature.melspectrogram(y=soundArr, sr=sample_rate)
    mspec = lb.util.fix_length(mspec, size=max_len, axis=1)

    return mfcc, croma, mspec


# ==============================
# Prediction Function
# ==============================
def classificationResults(soundFilePath):

    if not os.path.exists(soundFilePath):
        return [
            "Sorry, No File Found",
            "Please upload the file in .wav format"
        ]

    # Extract features
    mfcc, croma, mspec = getFeaturesForNeuralNetwork(soundFilePath)

    # Add channel dimension (H, W, 1)
    mfcc = mfcc[..., np.newaxis]
    croma = croma[..., np.newaxis]
    mspec = mspec[..., np.newaxis]

    # Add batch dimension (1, H, W, 1)
    mfcc = np.expand_dims(mfcc, axis=0)
    croma = np.expand_dims(croma, axis=0)
    mspec = np.expand_dims(mspec, axis=0)

    # Debug (optional)
    # print("MFCC shape:", mfcc.shape)
    # print("Croma shape:", croma.shape)
    # print("MelSpec shape:", mspec.shape)

    # Predict
    result = model.predict({
        "mfcc": mfcc,
        "croma": croma,
        "mspec": mspec
    }, verbose=0)

    result = result.flatten()

    diseaseArray = [
        'Asthma',
        'Bronchiectasis',
        'Bronchiolitis',
        'COPD',
        'Healthy',
        'LRTI',
        'Pneumonia',
        'URTI'
    ]

    # Top prediction
    indexMax = np.argmax(result)

    # Second best prediction
    sorted_indices = np.argsort(result)
    indexSecMax = sorted_indices[-2]

    res1 = f"Respiratory disorder detected: {diseaseArray[indexMax]} with probability {result[indexMax] * 100:.2f}%"
    res2 = f"Second possible condition: {diseaseArray[indexSecMax]} with probability {result[indexSecMax] * 100:.2f}%"

    return [res1, res2]