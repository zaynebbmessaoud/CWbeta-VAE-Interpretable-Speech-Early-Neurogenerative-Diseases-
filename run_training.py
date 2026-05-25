# -*- coding: utf-8 -*-
"""
Created on Thu Jan 22 12:58:40 2026

@author: user
"""

import tensorflow as tf
#from tensorflow 
import keras
from tensorflow.keras import layers
import numpy as np
import random

# 1. DETERMINISTIC SEEDING (Stability Step 1)
def set_seeds(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    # Ensure deterministic operations where possible
    tf.config.experimental.enable_op_determinism()

set_seeds(42)

input_shape = (224, 224, 1)
latent_dim = 32
num_classes = 3
target_beta = 1.0  
epochs = 50
batch_size = 8

# Class Weights
counts = np.array([14, 41, 35])
total = np.sum(counts)
class_weights = (total / (len(counts) * counts)).astype(np.float32)
class_weight_tensor = tf.constant(class_weights, dtype=tf.float32)

# ---------------------------------------------------------
# 2. STABILIZED ARCHITECTURE
# ---------------------------------------------------------
def build_vae(input_shape, latent_dim, num_classes):
    # --- ENCODER ---
    img_in = keras.Input(shape=input_shape)
    lbl_in = keras.Input(shape=(1,), dtype='int32')
    
    emb = layers.Embedding(num_classes, 16)(lbl_in)
    emb = layers.Flatten()(emb)
    
    x = layers.Conv2D(32, 3, strides=2, padding='same')(img_in)
    x = layers.LayerNormalization()(x) # Stability Step 2: LayerNorm
    x = layers.Activation('relu')(x)
    
    x = layers.Conv2D(64, 3, strides=2, padding='same')(x)
    x = layers.LayerNormalization()(x)
    x = layers.Activation('relu')(x)
    
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.concatenate([x, emb])
    
    # Bottleneck stability
    mu = layers.Dense(latent_dim, name="mu")(x)
    log_var = layers.Dense(latent_dim, name="log_var")(x)
    
    # Sampling with Reparameterization
    @keras.saving.register_keras_serializable()
    def sampling(args):
        mu, log_var = args
        eps = tf.random.normal(tf.shape(mu))
        return mu + tf.exp(0.5 * log_var) * eps
    
    z = layers.Lambda(sampling, name="z")([mu, log_var])
    encoder = keras.Model([img_in, lbl_in], [mu, log_var, z])
    
    # --- DECODER ---
    z_in = keras.Input(shape=(latent_dim,))
    d = layers.Dense(7 * 7 * 64)(z_in)
    d = layers.LayerNormalization()(d) # Stability Step 2
    d = layers.Activation('relu')(d)
    d = layers.Reshape((7, 7, 64))(d)
    
    # Progressive upsampling
    for filters in [64, 32, 16, 8]:
        d = layers.Conv2DTranspose(filters, 3, strides=2, padding='same')(d)
        d = layers.LayerNormalization()(d)
        d = layers.Activation('relu')(d)
        
    d = layers.Conv2DTranspose(1, 3, strides=2, activation='sigmoid', padding='same')(d)
    decoder = keras.Model(z_in, d)
    
    return encoder, decoder

encoder, decoder = build_vae(input_shape, latent_dim, num_classes)

# 3. GRADIENT CLIPPING (Stability Step 3)
optimizer = keras.optimizers.Adam(learning_rate=5e-5, clipnorm=1.0)

# ---------------------------------------------------------
# 4. SCHEDULER & TRAIN STEP
# ---------------------------------------------------------
def get_sigmoid_beta(epoch, total_epochs, target_beta):
    k = 0.20 # Reduced steepness for smoother transition
    x0 = total_epochs // 2 
    return target_beta / (1 + np.exp(-k * (epoch - x0)))

@tf.function
def train_step(x_batch, y_batch, beta):
    y_batch = tf.cast(y_batch, tf.int32)
    with tf.GradientTape() as tape:
        mu, log_var, z = encoder([x_batch, y_batch], training=True)
        recon = decoder(z, training=True)
        
        # Weighted Reconstruction Loss
        weights = tf.gather(class_weight_tensor, y_batch)
        mse = tf.square(x_batch - recon)
        recon_loss = tf.reduce_mean(mse * tf.reshape(weights, [-1, 1, 1, 1])) * 500
        
        # KL Loss
        kl_loss = -0.5 * tf.reduce_mean(1 + log_var - tf.square(mu) - tf.exp(log_var))
        total_loss = recon_loss + (beta * kl_loss)
        
    trainable_vars = encoder.trainable_variables + decoder.trainable_variables
    grads = tape.gradient(total_loss, trainable_vars)
    
    # Apply gradients with optimizer clipping
    optimizer.apply_gradients(zip(grads, trainable_vars))
    return total_loss, recon_loss, kl_loss

# ---------------------------------------------------------
# 4. BOUCLE D'ENTRAÎNEMENT
# ---------------------------------------------------------
# data split 
# The safest and most robust way to split clinical data:
Data =  Data / 255.0
label = label.astype(np.int32)    
    
x_train, x_test, y_train, y_test = train_test_split(
    Data, 
    labels, 
    test_size=0.1, 
    shuffle=True, 
    stratify=labels,   # Forces the Train and Test sets to have balanced cohorts
    random_state=42    # Ensures you get the same split every time you run the code
)



#--------------------Data----------------------------------
x_train =  Data / 255.0
y_train = label.astype(np.int32)
# Assurez-vous que x_train est normalisé [0, 1]
train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train.astype(np.int32)))
train_dataset = train_dataset.shuffle(len(x_train)).batch(batch_size).prefetch(tf.data.AUTOTUNE)

history = {"total": [], "recon": [], "kl": [], "beta": []}

for epoch in range(1, epochs + 1):
    current_beta = get_sigmoid_beta(epoch, epochs, target_beta)
    e_total, e_recon, e_kl = 0, 0, 0
    batches = 0
    
    for xb, yb in train_dataset:
        loss, rec, kl = train_step(xb, yb, tf.cast(current_beta, tf.float32))
        e_total += loss; e_recon += rec; e_kl += kl
        batches += 1
        
    history["total"].append(e_total/batches)
    history["recon"].append(e_recon/batches)
    history["kl"].append(e_kl/batches)
    history["beta"].append(current_beta)
    
    if epoch % 10 == 0:
        print(f"Epoch {epoch}: Recon={history['recon'][-1]:.2f}, KL={history['kl'][-1]:.2f}, Beta={current_beta:.2f}")


# Save the encoder in the new native Keras format
encoder.save('conditional_encoder_balanced_stable_1.keras')

print(" Encoder saved to 'conditional_encoder.keras'")


#-----------------Learning Curves ----------------------------------------
def plot_vae_training_results(history):
    # Configuration du style pour un rendu académique
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    epochs_range = range(1, len(history['total']) + 1)

    # --- Graphique 1 : Pertes de Reconstruction et KL ---
    # Nous utilisons deux échelles Y pour mieux voir la convergence
    ax1.plot(epochs_range, history['recon'], label='Reconstruction Loss (Weighted)', 
             color='#1f77b4', linewidth=2)
    ax1.set_xlabel('Epochs', fontsize=12)
    ax1.set_ylabel('Reconstruction Loss Scale', color='#1f77b4', fontsize=12)
    ax1.tick_params(axis='y', labelcolor='#1f77b4')
    
    # Création d'un second axe pour la KL Divergence
    ax1_kl = ax1.twinx()
    ax1_kl.plot(epochs_range, history['kl'], label='KL Divergence', 
                color='#ff7f0e', linewidth=2, linestyle='--')
    ax1_kl.set_ylabel('KL Divergence Scale', color='#ff7f0e', fontsize=12)
    ax1_kl.tick_params(axis='y', labelcolor='#ff7f0e')
    
    #ax1.set_title('VAE Convergence: Balance between Texture and Latent Space', fontsize=14)
    
    # --- Graphique 2 : Sigmoid Beta Schedule ---
    ax2.plot(epochs_range, history['beta'], label='Beta (KL Weight)', 
             color='green', linewidth=2)
    ax2.fill_between(epochs_range, history['beta'], color='green', alpha=0.1)
    ax2.set_xlabel('Epochs', fontsize=12)
    ax2.set_ylabel('Beta Value', fontsize=12)
    #ax2.set_title('Sigmoid KL Annealing Schedule', fontsize=14)
    ax2.legend(loc='upper left')

    plt.tight_layout()
    plt.show()

# Appel de la fonction après l'entraînement
plot_vae_training_results(history)

# ---------------------------------------------------------
# 5. EXTRACTION DES VECTEURS MU (ORDRONNÉE)
# ---------------------------------------------------------
# --- Plot Training Curves ---
plt.figure(figsize=(10, 4))
plt.plot(history["recon"], label="Reconstruction Loss (Weighted)")
plt.plot(history["kl"], label="KL Divergence (Regularization)")
plt.title("Learning Curves: Weighted Beta-VAE")
plt.legend()
plt.show()

# ---------------------------------------------------------
# After training: Extracting and Balancing for Ensemble
# ---------------------------------------------------------

# 1. Get raw latent representations for the training set
# We use the 'mu' (mean) as the representative feature vector
# After training, extract mu vectors
mu_vectors = []
labels = []

for x_batch, y_batch in train_dataset:
    mu, _, _ = encoder([x_batch, y_batch], training=False)
    mu_vectors.append(mu.numpy())
    labels.append(y_batch.numpy())

mu_vectors = np.concatenate(mu_vectors)
labels = np.concatenate(labels).flatten()

print(f"Original Latent Shapes: {mu_vectors.shape}, Labels: {y_train.shape}")

# ----------------2. Apply Natural VAE Augmentation to balance classes to N=41 each
def get_balanced_latent_dataset(encoder, x_train, y_train, target_n=41):
    """
    Augmente les classes minoritaires dans l'espace latent pour atteindre target_n.
    """
    # 1. Extraction des vecteurs mu originaux
    mu_orig, _, _ = encoder.predict([x_train, y_train])
    
    mu_balanced = []
    y_balanced = []
    
    for class_id in range(3): # HC=0, PD=1, TE=2
        # Extraire les mu existants pour cette classe
        indices = np.where(y_train == class_id)[0]
        class_mu = mu_orig[indices]
        current_n = len(indices)
        
        # Ajouter les originaux
        mu_balanced.append(class_mu)
        y_balanced.append(np.full(current_n, class_id))
        
        # 2. Générer des échantillons synthétiques si nécessaire
        if current_n < target_n:
            n_to_add = target_n - current_n
            
            # On tire au sort parmi les indices existants pour servir de base
            random_indices = np.random.choice(range(current_n), size=n_to_add, replace=True)
            base_mu = class_mu[random_indices]
            
            # On ajoute un bruit blanc (std=0.1) pour créer une variation naturelle
            # Cela correspond à se déplacer légèrement autour des points bleus du t-SNE
            noise = np.random.normal(0, 0.01, size=base_mu.shape)
            synthetic_mu = base_mu + noise
            
            mu_balanced.append(synthetic_mu)
            y_balanced.append(np.full(n_to_add, class_id))
            
    return np.concatenate(mu_balanced), np.concatenate(y_balanced)

# Application de l'augmentation
mu_final, y_final = get_balanced_latent_dataset(encoder, x_train, y_train, target_n=41)

print(f"Structure finale du dataset : {mu_final.shape}") 
# Résultat attendu : (123, 32) car 41 * 3 classes = 123

