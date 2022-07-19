
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
n_samples=100000
iterationsPerSample=10
L=25
cutLJ=np.power(2,1/6)*sigmaLJ
volume_per_monomer=30
delta_v =10
def mpc():
    return 20#int(np.random.rand()*5)+1

syst = espressomd.System(box_l =[L,L,L],time_step =0.001)
methods.createMultipleDiamonds(syst,mpc,1)
# methods.createDiamond(syst,20)
# methods.createPolymer(sigmaLJ,epsLJ,cutLJ,syst,20)


syst.change_volume_and_rescale_particles(L,dir='xyz')

syst.non_bonded_inter[0,0].lennard_jones.set_params(epsilon =epsLJ,sigma=sigmaLJ,cutoff=cutLJ,shift ='auto')
methods.force_capping(syst)
syst.integrator.set_vv()
syst.thermostat.set_langevin(kT=kT,gamma=1,seed =41)



visualizer = espressomd.visualization.openGLLive(syst, background_color=[1, 1, 1])
# visualizer.run(1)


samplesVolume=np.zeros((n_samples,1))
samplesPressure=np.zeros((n_samples,1))

syst.cell_system.tune_skin(min_skin=0.01,max_skin=2,tol=0.001,int_steps=1000)

for i in tqdm.tqdm(range(n_samples*iterationsPerSample)):
    syst.integrator.run(iterationsPerSample)
    if i % iterationsPerSample==0:
        data=syst.analysis.pressure()
        samplesVolume[int(i/iterationsPerSample)]=np.power(syst.box_l[0],3)
        samplesPressure[int(i/iterationsPerSample)]=data['total']

        if i % 10000==0:
            syst.cell_system.tune_skin(min_skin=0.4,max_skin=2,tol=0.001,int_steps=1000)

            print("pressure " + str(samplesPressure))
            print("volume " + str(samplesVolume))
    delta_v+=methods.barostat(syst,delta_v,wantedPressure)
    #print(delta_v)





print("pressure " + str(samplesPressure))
print("volume " + str(samplesVolume))


visualizer.run(1)