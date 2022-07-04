
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
sigmaLJ=1.0
epsLJ=1.0
cutLJ=np.power(2,1/6)*sigmaLJ

syst = espressomd.System(box_l =[L,L,L],time_step =0.01)


methods.createDiamond(syst,2)

syst.non_bonded_inter[0,0].lennard_jones.set_params(epsilon =epsLJ,sigma=sigmaLJ,cutoff=cutLJ,shift ='auto')
syst.integrator.set_vv()
syst.thermostat.set_langevin(kT=1,gamma=1,seed=42)
print(syst.cell_system.interaction_range)
n_samples=30000
iterationsPerSample=1



syst.cell_system.use_verlet_lists = False
visualizer = espressomd.visualization.openGLLive(syst, background_color=[1, 1, 1])
#visualizer.run(0)

wantedPressure = 0.1
samplesVolume=np.zeros((n_samples,1))
samplesPressure=np.zeros((n_samples,1))


for i in tqdm.tqdm(range(n_samples)):
    syst.integrator.run(iterationsPerSample)
    syst.cell_system.skin=0
    data=syst.analysis.pressure()
    methods.barostat(syst,0.2,20,wantedPressure)
    samplesVolume[i]=np.power(syst.box_l[0],3)
    samplesPressure[i]=data['total']



np.savetxt("sampledPressure",samplesPressure)
np.savetxt("sampledVolume",samplesVolume)


visualizer.run(1)