
import espressomd
from espressomd import polymer, visualization, checkpointing
import numpy as np
import matplotlib.pyplot as plt
import bottleneck as bn
import tqdm
import methods


required_features = ["LENNARD_JONES"]
espressomd.assert_features(required_features)


N=0
MPC=10
sigmaLJ=1.0
epsLJ=1.0
wantedPressure = 1
kT=1
n_samples=10000
iterationsPerSample=10
L=50
cutLJ=np.power(2,1/6)*sigmaLJ
volume_per_monomer=30

def mpc():
    return 20#int(np.random.rand()*5)+1

syst = espressomd.System(box_l =[L,L,L],time_step =0.01)
methods.createMultipleDiamonds(syst,mpc,1)
L=np.power(syst.part.highest_particle_id*volume_per_monomer,1/3)*2
print(L)
syst.change_volume_and_rescale_particles(L,dir='xyz')

syst.non_bonded_inter[0,0].lennard_jones.set_params(epsilon =epsLJ,sigma=sigmaLJ,cutoff=cutLJ,shift ='auto')
methods.force_capping(syst)
syst.integrator.set_vv()
syst.thermostat.set_langevin(kT=kT,gamma=1,seed=42)
print(syst.cell_system.interaction_range)


visualizer = espressomd.visualization.openGLLive(syst, background_color=[1, 1, 1])
#visualizer.run(1)


samplesVolume=np.zeros((n_samples,1))
samplesPressure=np.zeros((n_samples,1))

syst.cell_system.tune_skin(min_skin=0.01,max_skin=2,tol=0.001,int_steps=1000)
print(syst.cell_system.skin)
for i in tqdm.tqdm(range(n_samples)):
    syst.integrator.run(iterationsPerSample)

    data=syst.analysis.pressure()
    methods.barostat(syst,1000,wantedPressure)
    samplesVolume[i]=np.power(syst.box_l[0],3)
    samplesPressure[i]=data['total']



np.savetxt("sampledPressure",samplesPressure)
np.savetxt("sampledVolume",samplesVolume)


visualizer.run(1)