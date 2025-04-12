import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import matplotlib.patches as patches
from skimage import data, color
from skimage.transform import resize

# 1. Load sample image from skimage
image_rgb = data.chelsea()  # Cute cat image 🐱
image_gray = color.rgb2gray(image_rgb)  # Convert to grayscale
image_gray = resize(image_gray, (64, 64))  # Resize to manageable size
image = (image_gray * 255).astype(np.float32)

# 2. Define a 3x3 edge detection kernel
kernel = np.array([
    [-1, -1, -1],
    [ 0,  0,  0],
    [ 1,  1,  1]
], dtype=np.float32)

# 3. Output initialization
output = np.zeros((image.shape[0] - 2, image.shape[1] - 2))

# 4. Setup matplotlib figure
fig, ax = plt.subplots(1, 2, figsize=(12, 6))
fig.suptitle("🎯 Real-Time CNN Convolution ", fontsize=14)

ax[0].set_title("📥 Input Image (Kernel Sliding)")
ax[1].set_title("📤 Feature Map Forming")

im1 = ax[0].imshow(image, cmap='gray', vmin=0, vmax=255)
im2 = ax[1].imshow(output, cmap='hot', vmin=-255, vmax=255)

# Hide ticks
for a in ax:
    a.set_xticks([])
    a.set_yticks([])

# Red rectangle to show current kernel position
rect = patches.Rectangle((0, 0), 3, 3, linewidth=2, edgecolor='red', facecolor='none')
ax[0].add_patch(rect)

# List of all positions for sliding
positions = [(i, j) for i in range(output.shape[0]) for j in range(output.shape[1])]

# 5. Animation function
def update(frame):
    if frame >= len(positions):
        return

    i, j = positions[frame]
    window = image[i:i+3, j:j+3]
    result = np.sum(window * kernel)
    output[i, j] = result
    im2.set_data(output)
    rect.set_xy((j, i))

    ax[0].set_title(f"📥 Input — Kernel at ({i},{j})")
    ax[1].set_title(f"📤 Feature Map — Value at ({i},{j}): {result:.1f}")

ani = FuncAnimation(fig, update, frames=len(positions), interval=1, repeat=False)

# 6. Save the animation
ani.save("auto_convolution_cat_demo.mp4", writer='ffmpeg', fps=30)
print("✅ Video saved as 'auto_convolution_cat_demo.mp4' 🎬")

plt.show()

