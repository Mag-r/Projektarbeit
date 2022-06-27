
import numpy as np
import matplotlib.pyplot as plt
import bottleneck as bn


data = np.loadtxt("sampledPressure")

# for i in range(np.size(data[0,:])):
#     plt.plot(bn.move_mean(data[:,i], window=500), label=str(np.mean(data[10000:,i])))
means=np.mean(data[10000:],axis=0)
print(np.std(data[10000:],axis=0))
plt.errorbar(np.linspace(0,19,20),means-np.linspace(0.0001,5,20),yerr=np.std(data[10000:],axis=0))
#plt.plot(bn.move_mean(data[:,10], window=500), label=str(np.mean(data[10000:,10])))
#plt.legend()
#plt.plot(data[-1,:])
plt.show()
