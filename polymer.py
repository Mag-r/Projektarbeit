
import espressomd
from espressomd import polymer, visualization, checkpointing
import numpy as np
import matplotlib.pyplot as plt
import bottleneck as bn
import tqdm
import methods


required_features = ["LENNARD_JONES"]
espressomd.assert_features(required_features)

L=6
N=20
sigmaLJ=1
epsLJ=1
cutLJ=np.power(sigmaLJ,1/6)

syst = espressomd.System(box_l =[L,L,L],time_step =0.01)
checkpoint = espressomd.checkpointing.Checkpoint(checkpoint_id="turnOffThermo",checkpoint_path="check")
checkpoint.register("syst")

methods.createPolymer(sigmaLJ, epsLJ, cutLJ, syst,20)


syst.integrator.set_vv()
syst.thermostat.set_langevin(kT=1,gamma=1,seed=42)
print(syst.cell_system.interaction_range)
n_samples=30000
iterationsPerSample=1



syst.cell_system.use_verlet_lists = False
visualizer = espressomd.visualization.openGLLive(syst, background_color=[1, 1, 1])
#visualizer.run(0)
numberOfPressures=1
wantedPressure = np.linspace(0.0001,5,numberOfPressures)
samplesVolume=np.zeros((n_samples,numberOfPressures))
samplesPressure=np.zeros((n_samples,numberOfPressures))

for j in tqdm.tqdm(range(numberOfPressures)):
    for i in tqdm.tqdm(range(n_samples)):
        syst.integrator.run(iterationsPerSample)
        syst.cell_system.skin=0
        data=syst.analysis.pressure()
        methods.barostat(syst,0.2,20,wantedPressure[j])
        samplesVolume[i,j]=np.power(syst.box_l[0],3)
        samplesPressure[i,j]=data['total']
checkpoint.save()


np.savetxt("sampledPressure",samplesPressure)
np.savetxt("sampledVolume",samplesVolume)
plt.plot(samplesVolume[-1,:])
plt.show()
# plt.plot(bn.move_mean(samples[:,1],window=20))
# plt.show()

visualizer.run(1)