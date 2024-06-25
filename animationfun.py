import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
image_data = np.random.rand(100, 100)

# Initialize plot and image object
fig, ax = plt.subplots()
im = ax.imshow(image_data, cmap='viridis')
print(im)
print(fig)
print('----'*30)
# Function to update the image data
def update(frame):
    # Generate new random image data (example)
    new_data = np.random.rand(100, 100)
    im.set_array(new_data)
    return [im]

# Create animation
ani = FuncAnimation(fig, update, frames=range(200), interval=200)
plt.show()
