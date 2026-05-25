# -*- coding: utf-8 -*-
"""
Created on Sat Oct 12 11:59:14 2024

@author: LAPTA
"""

import os
import librosa
import numpy as np
import matplotlib.pyplot as plt
import librosa.display
from sklearn.preprocessing import StandardScaler
import cv2
from keras.preprocessing.image import img_to_array

HC_DIR = 'E:\\Data\hc'
PD_DIR = 'E:\Data\pd'
TE_DIR = 'E:\Data\ET'

# Function to extract MFCC, DMFCC, DDMFCC, Spectrogram, Mel-spectrogram
def create_melspectrogram(y, sr, n_mels=128, n_fft=1024):
    mel_spectrogram = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels, n_fft=n_fft)
    mel_spectrogram_db = librosa.power_to_db(mel_spectrogram, ref=np.max)
    return mel_spectrogram_db
    
# Load files and extract features

X = []  # List for images
labels = []  # List for labels
for category, folder in [('HC', HC_DIR), ('PD', PD_DIR), ('TE', TE_DIR)]:
    for file_name in os.listdir(folder):
        file_path = os.path.join(folder, file_name)
        y, sr = librosa.load(file_path, sr=16000)
        mel_spec = create_melspectrogram(y, sr) # Mel_spectrogram
            
        plt.figure(figsize=(5, 5))
        librosa.display.specshow(mel_spec, sr=sr, x_axis='time')
        plt.axis('off')
        plt.savefig(f'{file_name}.png', bbox_inches='tight', pad_inches=0)
        plt.close()
        img = cv2.imread(f'{file_name}.png', 0)
        image = cv2.resize(img, (224, 224))
        images=img_to_array(image)
        X.append(images)  
        #labels.append(0 if category == 'HC' else 1) 
        
        if category == 'HC':
            labels.append(0)
        elif category == 'PD':
            labels.append(1)
        else:
           labels.append(2)
    
Data= np.array(X)
label= np.array(labels)
