
import numpy as np
import matplotlib.pyplot as plt
import bottleneck as bn


data = np.loadtxt("sampledPressure")

plt.plot(data)
print(np.mean(data))
plt.show()
