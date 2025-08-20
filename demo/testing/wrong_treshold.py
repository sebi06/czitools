import numpy as np
import matplotlib.pyplot as plt
from skimage.measure import regionprops, label
from skimage.filters import threshold_otsu

# --- 1. Generate a synthetic microscopy image ---
# Create a blank canvas
image_size = 200
image = np.zeros((image_size, image_size))

# Create coordinates for the image
x, y = np.mgrid[0:image_size, 0:image_size]

# Define the "cell" using a 2D Gaussian profile
# This is our measurement channel (Channel A)
radius_x, radius_y = 30, 40
center_x, center_y = image_size // 2, image_size // 2
intensity_profile = np.exp(-((x - center_x)**2 / (2 * radius_x**2) + (y - center_y)**2 / (2 * radius_y**2)))

# Add intensity and some random noise
channel_A_measurement = 150 * intensity_profile + 20 * np.random.rand(image_size, image_size)
channel_A_measurement = np.clip(channel_A_measurement, 0, 255)

# Create an ideal segmentation channel (Channel B)
# This channel clearly marks the entire object's boundary
# The shape is made slightly larger to simulate a common biological reality (e.g., cytoplasm around a signal)
segment_radius_x, segment_radius_y = 35, 45
channel_B_segmentation = (((x - center_x)**2 / segment_radius_x**2 + (y - center_y)**2 / segment_radius_y**2) < 1).astype(float) * 200
channel_B_segmentation += 20 * np.random.rand(image_size, image_size)
channel_B_segmentation = np.clip(channel_B_segmentation, 0, 255)


# --- 2. BIASED APPROACH: Segmenting and measuring in the same channel (Channel A) ---
# Calculate a threshold based on the measurement channel
thresh_A = threshold_otsu(channel_A_measurement)

# Create a mask: only the brightest pixels are selected
biased_mask = channel_A_measurement > thresh_A

# Measure the mean intensity within this biased mask
biased_intensity = np.mean(channel_A_measurement[biased_mask])


# --- 3. CORRECT APPROACH: Segmenting in Channel B, measuring in Channel A ---
# Calculate a threshold on the ideal segmentation channel
thresh_B = threshold_otsu(channel_B_segmentation)

# Create an unbiased mask that covers the entire object
unbiased_mask = channel_B_segmentation > thresh_B

# Apply the unbiased mask to the measurement channel and measure the intensity
unbiased_intensity = np.mean(channel_A_measurement[unbiased_mask])


# --- 4. Visualize the results ---
fig, axes = plt.subplots(2, 3, figsize=(18, 12))
fig.suptitle('Comparison of Biased vs. Unbiased Measurement Method', fontsize=20)
plt.subplots_adjust(top=0.9)

# Column titles
axes[0, 0].set_title("Source Images", fontsize=14)
axes[0, 1].set_title("Problem: Segmenting the Measurement Channel", fontsize=14)
axes[0, 2].set_title("Solution: Using an Independent Channel", fontsize=14)

# Row 1: Images and mask overlays
ax = axes[0, 0]
im = ax.imshow(channel_A_measurement, cmap='viridis')
ax.set_ylabel("Channel A (Measurement)", fontsize=12)
ax.axis('off')
plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

ax = axes[0, 1]
ax.imshow(channel_A_measurement, cmap='viridis')
ax.contour(biased_mask, colors='red', linewidths=1.5) # Red contour for the biased mask
ax.text(5, 190, 'Mask from Channel A', color='white', backgroundcolor='red')
ax.axis('off')

ax = axes[0, 2]
ax.imshow(channel_A_measurement, cmap='viridis')
ax.contour(unbiased_mask, colors='cyan', linewidths=1.5) # Cyan contour for the correct mask
ax.text(5, 190, 'Mask from Channel B', color='black', backgroundcolor='cyan')
ax.axis('off')

# Row 2: Segmentation channel and masks
ax = axes[1, 0]
im = ax.imshow(channel_B_segmentation, cmap='magma')
ax.set_ylabel("Channel B (Segmentation)", fontsize=12)
ax.axis('off')
plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

ax = axes[1, 1]
ax.imshow(biased_mask, cmap='gray')
ax.set_title("Mask selects only the brightest region", fontsize=12)
ax.axis('off')

ax = axes[1, 2]
ax.imshow(unbiased_mask, cmap='gray')
ax.set_title("Mask covers the entire object", fontsize=12)
ax.axis('off')

# Add the final measurement results to the bottom of the figure
plt.figtext(0.5, 0.02,
            f"BIASED APPROACH: Measured Mean Intensity = {biased_intensity:.2f}\n"
            f"CORRECT APPROACH: Measured Mean Intensity = {unbiased_intensity:.2f}\n"
            f"DIFFERENCE: The biased measurement is artificially inflated by {((biased_intensity - unbiased_intensity) / unbiased_intensity) * 100:.1f}%!",
            ha="center", fontsize=16, bbox={"facecolor":"lightcoral", "alpha":0.6, "pad":10})

plt.tight_layout(rect=[0, 0.08, 1, 0.95])
plt.show()