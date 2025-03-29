import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import numpy as np


# Function to convert hex color to RGB tuple
def hex_to_rgb(hex_color):
    hex_color = hex_color.lstrip("#")
    return tuple(int(hex_color[i : i + 2], 16) / 255.0 for i in (0, 2, 4))


# Hex color string
hex_color = "#00FF33"

# Convert hex to RGB
rgb_color = hex_to_rgb(hex_color)

# Create the colormap with a gradient from the RGB color to white
custom_cmap = LinearSegmentedColormap.from_list("custom_cmap", [(0, 0, 0), rgb_color])

# Create some data
x = np.linspace(0, 10, 100)
y = x**2

# Create a scatter plot
plt.scatter(x, y, c=y, cmap=custom_cmap)
plt.colorbar()  # Show color scale
plt.title("Custom Colormap Example")
plt.xlabel("X-axis")
plt.ylabel("Y-axis")
plt.show()
