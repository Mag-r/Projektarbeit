
import numpy as np
import matplotlib.pyplot as plt
import bottleneck as bn


data = np.loadtxt("sampledVolume")

plt.plot(data)
print(np.mean(data))
plt.show()
