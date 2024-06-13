import numpy as np
from scipy.stats import entropy
import nibabel as nib


# Example 3D numpy array
np.random.seed(0)
volume =np.asarray((nib.as_closest_canonical(nib.load('/localmount/volume-hd/users/uline/data_sets/CVD/d242b2b4-42d6-426e-a704-447cb86f9d87/T1.nii.gz'))).get_fdata(), dtype=np.uint8)

# Calculate Coefficient of Variation (CV)
mean_val = np.mean(volume)
std_dev = np.std(volume)
print(std_dev, mean_val)
cv = std_dev / mean_val if mean_val != 0 else 0  # Avoid division by zero

# Calculate Entropy
# First, compute the histogram
hist, _ = np.histogram(volume, bins=range(257))
# Normalize the histogram to get the probability distribution
prob_dist = hist / np.sum(hist)
# Calculate entropy
vol_entropy = entropy(prob_dist, base=2)

print(f"Coefficient of Variation: {cv}")
print(f"Entropy: {vol_entropy}")
