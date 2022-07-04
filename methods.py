
import espressomd
from espressomd import polymer, visualization, checkpointing
import numpy as np

def createPolymer(sigmaLJ, epsLJ, cutLJ, syst,bpc,bondLength=1,kFene=7,d_r_max=2):

    if bpc<2:
        raise Exception("no Polymer")
    syst.non_bonded_inter[0,0].lennard_jones.set_params(epsilon =epsLJ,sigma=sigmaLJ,cutoff=cutLJ,shift ='auto')
    fene =espressomd.interactions.FeneBond(k=kFene,r_0=bondLength,d_r_max=d_r_max)
    syst.bonded_inter.add(fene)

    initPos = np.random.rand(3)*syst.box_l
    p_Previous=syst.part.add(pos=initPos)
    distancePerDimension=np.power(3*bondLength,-1/2)
    for i in range(1,bpc):
        p=syst.part.add(pos=[initPos[0]+i*distancePerDimension+i,initPos[1]+i*distancePerDimension+i/2,initPos[2]+i*distancePerDimension])
        p.add_bond((fene,p_Previous))
        p_Previous=p
    syst.integrator.set_steepest_descent(f_max=0.01,gamma=30,max_displacement=0.01)

    syst.integrator.run(100)


def barostat(syst,deltaV,numberOfParticles,pressure):
    '''
    implements a barostat using MC with the acceptance function
    P(O->N)= exp(- beta *(EN-EO + pressure*(VN-VO))+numberOfParticles *log(VN/VO))
    '''
    
    oldV=syst.volume()
    vPrime= oldV + np.random.normal(loc=0,scale=deltaV,size=None)
    if np.power(vPrime,1/3)<=2*(syst.cell_system.interaction_range):
        print("zu klein")
        print(np.power(vPrime,1/3))
    oldEnergy = syst.analysis.energy()

    syst.change_volume_and_rescale_particles(np.power(vPrime,1/3),dir ='xyz')
    beta=1/syst.thermostat.get_state()[0]["kT"]

    newEnergy = syst.analysis.energy()

    acceptance=np.exp(-beta*(newEnergy['total']-oldEnergy['total']+ pressure*(vPrime-oldV))+(numberOfParticles)*np.log(vPrime/oldV))
    if acceptance<1:
        if acceptance<np.random.uniform():
            syst.change_volume_and_rescale_particles(np.power(oldV,1/3),dir ='xyz')




def createDiamond(syst,mpc,bondLength=1,kFene=7,d_r_max=2):
    fene =espressomd.interactions.FeneBond(k=kFene,r_0=bondLength,d_r_max=d_r_max)
    syst.bonded_inter.add(fene)
    espressomd.polymer.setup_diamond_polymer(system=syst,bond=fene,MPC=mpc)
    syst.integrator.set_steepest_descent(f_max=0.01,gamma=30,max_displacement=0.01)
    syst.integrator.run(100)

