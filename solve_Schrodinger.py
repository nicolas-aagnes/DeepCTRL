from math import sqrt
import numpy as np
from pde import PDE, CartesianGrid, FileStorage, ScalarField, plot_kymograph
import h5py


def create_trainset():
    sz = 5
    alpha = np.linspace(0.1,1.5,sz)
    beta = np.linspace(0,sz)
    for a in range(sz):
        for b in range(sz):
            for c in range(sz):
                for d in range(sz):
                    BC = "0.1*sin(t*"+str(alpha[a])+")+"+"0.2*sin(t*"+str(alpha[b])+")+"+"0.5*sin(t*"+str(alpha[c])+")"+"+0.8*sin(t*"+str(beta[d])+")"
                    print(BC)
                    grid = CartesianGrid([[-5, 5]], 128,periodic=False)  # generate grid

                    # create a (normalized) wave packet with a certain form as an initial condition
                    initial_state = ScalarField.from_expression(grid, "4*exp(x)/(1+exp(2*x))")
                    initial_state /= sqrt(initial_state.to_scalar("norm_squared").integral.real)

                    eq = PDE({"ψ": f"I*0.5* laplace(ψ)+I*(ψ *ψ*conjugate(ψ))"},bc={"value_expression": BC})  # define the pde

                    path  ="./schrodingerSoln.hdf5"
                    writer = FileStorage(path, write_mode="append")
                    eq.solve(initial_state, t_range=3.14/2, dt=1e-5, tracker=[writer.tracker(0.02)])

def create_testset():
    sz = 100
    alpha = np.linspace(0, 2, sz)
    for a in range(sz):

                BC = "t*"+str(alpha[a])
                print(BC)
                grid = CartesianGrid([[-5, 5]], 128, periodic=False)  # generate grid

                # create a (normalized) wave packet with a certain form as an initial condition
                initial_state = ScalarField.from_expression(grid, "4*exp(x)/(1+exp(2*x))")
                initial_state /= sqrt(initial_state.to_scalar("norm_squared").integral.real)

                eq = PDE({"ψ": f"I*0.5* laplace(ψ)+I*(ψ *ψ*conjugate(ψ))"},
                         bc={"value_expression": BC})  # define the pde

                path = "./schrodingerSoln_test.hdf5"
                writer = FileStorage(path, write_mode="append")
                eq.solve(initial_state, t_range=3.14 / 2, dt=1e-5, tracker=[writer.tracker(0.02)])

def createsoln():
    files = ['./schrodingerSoln.hdf5','./schrodingerSoln_test.hdf5']
    savenames = [ "./Data/Soln_Schrodinger.npy","./Data/Soln_Schrodinger_test.npy"]
    trainsz = [625,100]
    for j in range(2):
        f1 = h5py.File(files[j], 'r')
        dat =np.array(f1['data'])
        sz = 79
        trainset = trainsz[j]
        M_x = 128
        M_t = sz
        Solntens = np.zeros((M_x,M_t,trainset),dtype=np.complex_)
        if j==0:
            for i in range(126,trainset+126):
                Solntens[:,:,i-126] = dat[i*sz:(i+1)*sz,:].T
        else:
            for i in range(trainset):
                Solntens[:,:,i] = dat[i*sz:(i+1)*sz,:].T
        np.save(savenames[j], Solntens)

def createBC():
    t = np.arange(0,3.14/2,0.02)
    assert(len(t)==79)
    #create test BC...
    M_t = 79
    sz = 100
    BC_test = np.zeros((M_t, sz))

    alpha = np.linspace(1, 5, sz)
    for a in range(sz):

        BC_test[:, a] = t*alpha[a]

    np.save("./Data/BC_schrodinger_test.npy", BC_test)

    #create train BC...
    M_t = 79
    trainsz = 625
    BC =np.zeros((M_t,trainsz))
    sz = 5
    alpha = np.linspace(0.1, 1.5, sz)
    beta = np.linspace(0, sz)
    i =0
    for a in range(sz):
        for b in range(sz):
            for c in range(sz):
                for d in range(sz):
                    BC[:,i] =0.1*np.sin(t*alpha[a])+0.2*np.sin(t*alpha[b])+0.5*np.sin(t*alpha[c])+0.8*np.sin(t*beta[d])
                    i = i+1
    np.save("./Data/BC_schrodinger.npy", BC)

def main():

    create_trainset()
    create_testset()
    createsoln()
    createBC()
main()