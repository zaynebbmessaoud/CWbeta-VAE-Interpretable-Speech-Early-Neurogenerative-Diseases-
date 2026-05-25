# -*- coding: utf-8 -*-
"""
Created on Sat May 23 15:28:57 2026

@author: user
"""

import numpy as np
from scipy.stats import permutation_test
from statsmodels.stats.multitest import fdrcorrection
import pandas as pd

# ------------------------------
# 1. LOAD YOUR RAW DATA (assuming format from code)
# ------------------------------
# saliency_pd = saliency_maps[pd_mask]  # shape (41, 224, 224)
# saliency_te = saliency_maps[te_mask]  # shape (35, 224, 224)
# saliency_ctrl = saliency_maps[ctrl_mask]  # shape (14, 224, 224)

# Band definitions (mapping from frequency bands to spectrogram bin indices)
band_definitions = {
    'Rhythm (0–10 Hz)':           slice(0, 2),    # Bins 0-1
    'Phonatory Source (80–600 Hz)': slice(6, 31),  # Bins 6-30
    'Articulatory (0.6–1.8 kHz)':  slice(31, 65),  # Bins 31-64
    'Clarity (1.8–3 kHz)':         slice(65, 85),  # Bins 65-84
    'Noise/Breath (3–4 kHz)':      slice(85, 97)   # Bins 85-96
}

# ------------------------------
# 2. FUNCTION: Extract mean saliency per band for each subject
# ------------------------------
def extract_band_saliency(saliency_maps, band_definitions):
    """
    Extract mean saliency energy for each frequency band per subject.
    
    Parameters:
    -----------
    saliency_maps : numpy array shape (n_subjects, 224, 224)
        Saliency maps for a diagnostic group
    band_definitions : dict
        Mapping from band name to slice object for frequency bins
    
    Returns:
    --------
    dict : band_name -> numpy array shape (n_subjects,)
        Mean saliency value per subject for each band
    """
    n_subjects = saliency_maps.shape[0]
    band_saliency = {}
    
    for band_name, freq_slice in band_definitions.items():
        # Mean over frequency bins (axis=1) and time bins (axis=2)
        # Note: Assuming saliency_maps shape: (subject, freq_bin, time_bin)
        band_means = np.mean(saliency_maps[:, freq_slice, :], axis=(1, 2))
        band_saliency[band_name] = band_means
    
    return band_saliency

# ------------------------------
# 3. EXTRACT SALIENCY PER BAND FOR EACH GROUP
# ------------------------------
pd_bands = extract_band_saliency(saliency_pd, band_definitions)      # 41 subjects per band
te_bands = extract_band_saliency(saliency_te, band_definitions)      # 35 subjects per band
hc_bands = extract_band_saliency(saliency_ctrl, band_definitions)    # 14 subjects per band

# Combine into a single dictionary for easy access
all_groups = {
    'PD': pd_bands,
    'ET': te_bands,
    'HC': hc_bands
}

# ------------------------------
# 4. PERMUTATION TEST FUNCTION (pairwise, difference in means)
# ------------------------------
def permutation_pairwise(group1_values, group2_values, n_resamples=10000, random_seed=42):
    """
    Perform two-sample permutation test for difference in means.
    
    Returns:
    --------
    p_value : float
        Two-tailed p-value
    """
    def statistic(x, y, axis):
        return np.mean(x, axis=axis) - np.mean(y, axis=axis)
    
    res = permutation_test(
        (group1_values, group2_values),
        statistic,
        n_resamples=n_resamples,
        alternative='two-sided',
        random_state=random_seed
    )
    return res.pvalue

# ------------------------------
# 5. CLIFF'S DELTA (effect size for significant findings)
# ------------------------------
def cliffs_delta(x, y):
    """
    Calculate Cliff's Delta effect size (non-parametric).
    
    Interpretation:
    |δ| < 0.147 : negligible
    |δ| < 0.33  : small
    |δ| < 0.474 : medium
    |δ| >= 0.474: large
    """
    n1, n2 = len(x), len(y)
    concordant = 0
    discordant = 0
    
    for xi in x:
        for yj in y:
            if xi > yj:
                concordant += 1
            elif xi < yj:
                discordant += 1
    
    delta = (concordant - discordant) / (n1 * n2)
    return delta

# ------------------------------
# 6. RUN ALL PAIRWISE TESTS FOR ALL BANDS
# ------------------------------
bands = list(band_definitions.keys())
group_pairs = [('PD', 'HC'), ('ET', 'HC'), ('PD', 'ET')]

# Store results
results = []

for band in bands:
    for g1, g2 in group_pairs:
        values1 = all_groups[g1][band]
        values2 = all_groups[g2][band]
        
        # Permutation test
        p_value = permutation_pairwise(values1, values2, n_resamples=10000)
        
        # Store
        results.append({
            'Band': band,
            'Comparison': f'{g1} vs {g2}',
            'Raw_p': p_value,
            'Mean_diff': np.mean(values1) - np.mean(values2)
        })

# Convert to DataFrame
df_results = pd.DataFrame(results)

# ------------------------------
# 7. FDR CORRECTION (Benjamini-Hochberg)
# ------------------------------
p_values = df_results['Raw_p'].values
reject, p_corrected = fdrcorrection(p_values, alpha=0.05)
df_results['FDR_q'] = p_corrected
df_results['Significant_q<0.05'] = reject

# ------------------------------
# 8. ADD EFFECT SIZES FOR SIGNIFICANT COMPARISONS (raw p < 0.05)
# ------------------------------
effect_sizes = []
for idx, row in df_results.iterrows():
    if row['Raw_p'] < 0.05:  # Raw significance threshold
        band = row['Band']
        g1, g2 = row['Comparison'].split(' vs ')
        values1 = all_groups[g1][band]
        values2 = all_groups[g2][band]
        delta = cliffs_delta(values1, values2)
        effect_sizes.append(delta)
    else:
        effect_sizes.append(np.nan)
df_results['Cliffs_Delta'] = effect_sizes

# ------------------------------
# 9. DISPLAY RESULTS
# ------------------------------
print("=" * 80)
print("PAIRWISE PERMUTATION TEST RESULTS (10,000 resamples)")
print("=" * 80)
print(df_results.to_string(index=False))

print("\n" + "=" * 80)
print("SUMMARY: Comparisons with Raw p < 0.05")
print("=" * 80)
significant_raw = df_results[df_results['Raw_p'] < 0.05]
if len(significant_raw) > 0:
    for _, row in significant_raw.iterrows():
        delta = row['Cliffs_Delta']
        # Interpret effect size
        if abs(delta) >= 0.474:
            effect_desc = "large"
        elif abs(delta) >= 0.33:
            effect_desc = "medium"
        elif abs(delta) >= 0.147:
            effect_desc = "small"
        else:
            effect_desc = "negligible"
        print(f"{row['Band']} | {row['Comparison']}: p = {row['Raw_p']:.4f}, "
              f"Cliff's δ = {delta:.3f} ({effect_desc} effect)")
else:
    print("No comparisons reached raw p < 0.05")

print("\n" + "=" * 80)
print("SUMMARY: Comparisons with FDR-corrected q < 0.05")
print("=" * 80)
significant_fdr = df_results[df_results['Significant_q<0.05']]
if len(significant_fdr) > 0:
    for _, row in significant_fdr.iterrows():
        print(f"{row['Band']} | {row['Comparison']}: q = {row['FDR_q']:.4f}")
else:
    print("No comparisons remained significant after FDR correction (q < 0.05)")

# ------------------------------
# 10. SAVE RESULTS TO CSV (optional)
# ------------------------------
# df_results.to_csv('permutation_test_results.csv', index=False)
# print("\nResults saved to 'permutation_test_results.csv'")