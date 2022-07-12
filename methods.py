
from platform import node
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


def barostat(syst,deltaV,pressure):
    '''
    implements a barostat using MC with the acceptance function
    P(O->N)= exp(- beta *(EN-EO + pressure*(VN-VO))+numberOfParticles *log(VN/VO))
    '''
    
    oldV=syst.volume()
    vPrime= oldV + np.random.normal(loc=0,scale=deltaV,size=None)
    if vPrime<0:
        vPrime=-vPrime
    if np.power(vPrime,1/3)<=2*(syst.cell_system.interaction_range):
        print("zu klein")
        print(np.power(vPrime,1/3))
    oldEnergy = syst.analysis.energy()

    syst.change_volume_and_rescale_particles(np.power(vPrime,1/3),dir ='xyz')
    beta=1/syst.thermostat.get_state()[0]["kT"]

    newEnergy = syst.analysis.energy()

    acceptance=np.exp(-beta*(newEnergy['total']-oldEnergy['total']+ pressure*(vPrime-oldV))+(len(syst.part.all()))*np.log(vPrime/oldV))
    if acceptance<1:
        if acceptance<np.random.uniform():
            syst.change_volume_and_rescale_particles(np.power(oldV,1/3),dir ='xyz')




def createDiamond(syst,mpc,bondLength=1,kFene=7,d_r_max=2):
    fene =espressomd.interactions.FeneBond(k=kFene,r_0=bondLength,d_r_max=d_r_max)
    syst.bonded_inter.add(fene)
    espressomd.polymer.setup_diamond_polymer(system=syst,bond=fene,MPC=mpc)
    syst.integrator.set_steepest_descent(f_max=0.01,gamma=30,max_displacement=0.01)
    syst.integrator.run(100)



def createMultipleDiamonds(syst,mpc_distribution,numberOfDiamondsPerDim,bondLength=1,kFene=7,d_r_max=2):
    box_length=syst.box_l[0]
    fene =espressomd.interactions.FeneBond(k=kFene,r_0=1.733*box_length/(4.0*numberOfDiamondsPerDim),d_r_max=d_r_max*5)
    syst.bonded_inter.add(fene)

    initial_node_positions= np.array([[0, 0, 0], [1, 1, 1],
                                                 [2, 2, 0], [0, 2, 2],
                                                 [2, 0, 2], [3, 3, 1],
                                                 [1, 3, 3], [3, 1, 3]])

    node_positions=np.array([])
    
    #copy diamant structure
    for i in range(numberOfDiamondsPerDim):
        for j in range(numberOfDiamondsPerDim):
            for k in range(numberOfDiamondsPerDim):
                node_positions=np.append(node_positions,initial_node_positions+4*np.array([i,j,k]).transpose())
    node_positions=node_positions.reshape((-1,3))
    node_positions=node_positions*box_length/ (4.0*numberOfDiamondsPerDim)
    syst.part.add(pos=node_positions)


    for i in range(np.shape(node_positions)[0]):
        for j in range(i+1,np.shape(node_positions)[0]):
            node_connection_vector=node_positions[i]-node_positions[j]
            node_connection_vector-=np.rint(node_connection_vector/box_length)*box_length
            if np.linalg.norm(node_connection_vector)<=1.733*box_length/(4.0*numberOfDiamondsPerDim):
                mpc=mpc_distribution()
                if mpc <=0:
                    raise Exception("mpc needs to be a positive number")

                #create bonds
                p=syst.part.add(pos=node_positions[i]-node_connection_vector/(mpc+1))
                start_point=syst.part.select(lambda p:np.array(p.pos == node_positions[i]).all())
                start_point.add_bond((fene,p))  
                p_previous =p
                for k in range(2,mpc+1):
                    p=syst.part.add(pos=node_positions[i]-node_connection_vector*k/(mpc+1))
                    p.add_bond((fene,p_previous))
                    p_previous=p
                
                end_point=syst.part.select(lambda p:np.array(p.pos == node_positions[j]).all())
                end_point.add_bond((fene,p))
                

    

    




