# -*- coding: utf-8 -*-
"""
Created on Thu May 21 11:00:52 2026

@author: user
"""

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import keras

# -------------------------------
# 1. Configuration
# -------------------------------
input_shape = (224, 224, 1)   # Spectrogram size
latent_dim = 32             # Must match your trained model
num_classes = 3               # PD vs. Control

# -------------------------------
# 2. Load Your Trained Conditional Encoder
# -------------------------------
# Votre encodeur utilise une fonction personnalisée pour l'astuce de reparamétrage.
# Nous devons la déclarer à Keras avant de lancer le chargement.

@keras.saving.register_keras_serializable()
def sampling(args):
    mu, log_var = args
    eps = tf.random.normal(tf.shape(mu))
    return mu + tf.exp(0.5 * log_var) * eps
# Make sure you saved the encoder separately during training
# Load the encoder

loaded_encoder = keras.models.load_model('conditional_encoder_balanced_stable.keras', compile=False)

print("Encoder loaded successfully.")
print("Input shapes:", encoder.input_shape)
print("Output shapes:", encoder.output_shape)

# -------------------------------
# 3. Compute Saliency Map: Input × Gradient
# -------------------------------

#----------Step 1: Compute Saliency Maps for All Samples
# Assume you have:

x_test = x_train
class_labels = y_train


saliency_maps = []

for i in range(len(x_test)):
    x_tensor = tf.convert_to_tensor(x_test[i:i+1])
    c_tensor = tf.convert_to_tensor(class_labels[i:i+1], dtype=tf.int32)

    x_var = tf.Variable(x_tensor)

    with tf.GradientTape() as tape:
        tape.watch(x_var)
        mu, log_var, z_triplet = encoder([x_var, c_tensor], training=False)
        loss = tf.reduce_sum(mu)

    grad = tape.gradient(loss, x_var)
    if grad is None:
        continue
    saliency = tf.abs(x_var * grad).numpy().squeeze()  # (224, 224)
    saliency_maps.append(saliency)

saliency_maps = np.array(saliency_maps)  # Shape: (N, 224, 224)

#----------Step 2: Compute Saliency Maps for each groups

pd_mask = (y_train == 1)
ctrl_mask = (y_train == 0)
te_mask = (y_train == 2)

print(f"PD samples: {np.sum(pd_mask)},TE samples: {np.sum(te_mask)}, HC samples: {np.sum(ctrl_mask)}")

saliency_pd = saliency_maps[pd_mask]
saliency_te = saliency_maps[te_mask]
saliency_ctrl = saliency_maps[ctrl_mask]

# Expert-rectified 5-band dictionary for 16kHz / 128-mels
band_definitions = {
    'Rhythm/Tremor (0–10 Hz)':        slice(0, 2),   # Bins 0-1 (Captures ~0-28 Hz)
    'Phonatory Source (80–600 Hz)':   slice(6, 31),  # Bins 6-30 (The F0 & H1-H3 region)
    'Lower Articulatory (0.6–1.8k Hz)': slice(31, 65), # Bins 31-64 (The F1/Vowel Space)
    'Upper Clarity (1.8–3k Hz)':      slice(65, 85), # Bins 65-84 (F2/F3 & Clarity)
    'Spectral Noise (3–4k Hz)':       slice(85, 97)  # Bins 85-96 (High-freq turbulence)
}

# 2. Fonction d'extraction statistique automatisée
def extract_band_stats(saliency_group):
    stats = {}
    for band_name, s_range in band_definitions.items():
        # axis=(1, 2) calcule la moyenne/std par échantillon sur la zone spectro-temporelle
        stats[band_name] = {
            'mean_per_sample': np.mean(saliency_group[:, s_range, :], axis=(1, 2)),
            'std_per_sample': np.std(saliency_group[:, s_range, :], axis=(1, 2))
        }
    return stats

# 3. Extraction des statistiques pour chaque groupe
pd_stats = extract_band_stats(saliency_pd)
te_stats = extract_band_stats(saliency_te)
ctrl_stats = extract_band_stats(saliency_ctrl)


#========================================================================
# 4. Préparation des données pour le graphique
bands = list(band_definitions.keys())

def get_plot_data(stats_dict):
    means = [np.mean(stats_dict[b]['mean_per_sample']) for b in bands]
    # Note : on utilise la moyenne des std pour garder la cohérence avec votre code précédent
    stds = [np.mean(stats_dict[b]['std_per_sample']) for b in bands]
    return means, stds

pd_means, pd_stds = get_plot_data(pd_stats)
ctrl_means, ctrl_stds = get_plot_data(ctrl_stats)
te_means, te_stds = get_plot_data(te_stats)

# Affichage des résultats numériques pour vérification
print("PD Means:", pd_means)
print("TE Means:", te_means)
print("CTRL Means:", ctrl_means)

print("PD Means:", pd_stds)
print("TE Means:", te_stds)
print("CTRL Means:", ctrl_stds)

# 5. Visualisation (Grouped Bar Chart)
x = np.arange(len(bands))
width = 0.25 

plt.figure(figsize=(14, 7))

# Bars 1 (PD)
plt.bar(x - width, pd_means, width, yerr=pd_stds,
        label='Parkinson (PD)', color='#e74c3c', alpha=0.8, capsize=4,
        error_kw={'capthick': 1, 'ecolor': '#333333'})

# Bars 2 (Control)
plt.bar(x, ctrl_means, width, yerr=ctrl_stds,
        label='Healthy Control', color='#3498db', alpha=0.8, capsize=4,
        error_kw={'capthick': 1, 'ecolor': '#333333'})

# Bars 3 (TE)
plt.bar(x + width, te_means, width, yerr=te_stds,
        label='Essential Tremor (TE)', color='#2ecc71', alpha=0.8, capsize=4,
        error_kw={'capthick': 1, 'ecolor': '#333333'})

plt.xlabel('Physiological Frequency Bands', fontsize=12)
plt.ylabel('Mean Saliency Energy', fontsize=12)
plt.title('Saliency Distribution across 5 Spectral Bands', fontsize=14, fontweight='bold')

plt.xticks(x, bands, rotation=20, ha='right')
plt.legend(frameon=True, shadow=True)
plt.grid(True, axis='y', alpha=0.3, linestyle='--')

plt.tight_layout()
plt.show()