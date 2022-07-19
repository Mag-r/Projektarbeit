import methods
import espressomd
from espressomd import polymer, visualization, checkpointing
import numpy as np
import matplotlib.pyplot as plt
import bottleneck as bn
import tqdm



required_features = ["LENNARD_JONES"]
espressomd.assert_features(required_features)


N=0
MPC=10
sigmaLJ=1.0
epsLJ=1.0
wantedPressure = 1
kT=1
n_samples=100
iterationsPerSample=1
L=20#np.power(MPC*800*sigmaLJ,1/3)
cutLJ=np.power(2,1/6)*sigmaLJ

def mpc():
    return 2#int(np.random.rand()*5)+1

syst = espressomd.System(box_l =[L,L,L],time_step =0.01)
methods.createMultipleDiamonds(syst,mpc,2)
syst.cell_system.skin=0.4
print(syst.part.n_part_types )
visualizer = espressomd.visualization.openGLLive(syst, background_color=[1, 1, 1])
visualizer.run(0)