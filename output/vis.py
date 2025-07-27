import numpy as np
import matplotlib.pyplot as plt

output = np.load("output/pdaf.npy")

plt.imshow(output, cmap='inferno')
plt.show()