import numpy as np
import matplotlib.pyplot as plt




def create_test_set():
    M_x  =100
    M_t = 1000
    x = np.linspace(-5,5,M_x)
    t = np.linspace(0,3.14,M_t)
    dt =t[1]-t[0]
    dx = x[1]-x[0]
    lam = dt/(dx*dx)
    U_0 = -(x+5)*(x-5)*(1/20)
    alpha = 10.0
    #plt.plot(x,U_0)
    #plt.show()
    mid = np.array([1-2*lam]*M_x)
    mid[0] = 0
    mid[M_x-1] = 0
    sideL = np.array([lam]*(M_x-1))
    sideU =np.array([lam]*(M_x-1))
    sideL[M_x-2] = 0
    sideU[0] = 0
    A = np.diag(mid)+np.diag(sideU,1)+np.diag(sideL,-1)
    B = np.zeros(M_x)

    testSet = 100
    alpha = np.linspace(0,2,testSet)
    #alpha = np.random.rand(trainSet)*1
    #alpha = np.zeros(trainSet)
    SolnTens = np.zeros((M_x,int(M_t/20),testSet))
    BC_mat = np.zeros((int(M_t/20),testSet))
    for s in range(testSet):
        alp = alpha[s]

        BC = np.linspace(0,1,M_t)*alp
        BC_mat[:,s] = BC[0:-1:20]
        SolnMat = np.zeros((M_x,M_t))
        SolnMat[:,0] = U_0
        for i in range(1,M_t):
            B[0] = BC[i]
            B[M_x-1] = BC[i]
            C = A@SolnMat[:,i-1]
            SolnMat[:,i]= C+B
        SolnTens[:,:,s] = SolnMat[:,0:-1:20]

    np.save("./Data/Soln_test.npy",SolnTens)
    np.save("./Data/BC_test.npy",BC_mat)

def create_train_set():
    M_x = 100
    M_t = 1000
    x = np.linspace(-5, 5, M_x)
    t = np.linspace(0, 3.14, M_t)
    dt = t[1] - t[0]
    dx = x[1] - x[0]
    lam = dt / (dx * dx)
    U_0 = -(x + 5) * (x - 5) * (1 / 20)
    alpha = 10.0
    # plt.plot(x,U_0)
    # plt.show()
    mid = np.array([1 - 2 * lam] * M_x)
    mid[0] = 0
    mid[M_x - 1] = 0
    sideL = np.array([lam] * (M_x - 1))
    sideU = np.array([lam] * (M_x - 1))
    sideL[M_x - 2] = 0
    sideU[0] = 0
    A = np.diag(mid) + np.diag(sideU, 1) + np.diag(sideL, -1)
    B = np.zeros(M_x)

    sz = 5
    a = sz
    b = sz
    c = sz
    d = sz
    e = sz
    f = 1
    trainSet = a*b*c*d*e*f*f
    alpha = np.linspace(1,5,sz)
    beta = np.linspace(0.1,1,sz)
    # alpha = np.random.rand(trainSet)*1
    # alpha = np.zeros(trainSet)
    SolnTens = np.zeros((M_x, int(M_t / 20), trainSet))
    BC_mat = np.zeros((int(M_t / 20), trainSet))
    j = 0
    for a_i in range(a):
        for b_i in range(b):
            for c_i in range(c):
                for d_i in range(d):
                    for e_i in range(d):
                        for f_i in range(f):
                            for g_i in range(f):
                                BC = 0.25*np.sin(alpha[a_i] * t)+0.5*np.sin(alpha[b_i] * t) + np.sin(alpha[c_i] * t)\
                                     +np.sin(beta[d_i] * t)+2*np.sin(beta[e_i] * t)
                                BC_mat[:, j] = BC[0:-1:20]
                                SolnMat = np.zeros((M_x, M_t))
                                SolnMat[:, 0] = U_0
                                for i in range(1, M_t):
                                    B[0] = BC[i]
                                    B[M_x - 1] = BC[i]
                                    C = A @ SolnMat[:, i - 1]
                                    SolnMat[:, i] = C + B
                                SolnTens[:, :, j] = SolnMat[:, 0:-1:20]
                                j+=1

    np.save("./Data/Soln.npy", SolnTens)
    np.save("./Data/BC.npy", BC_mat)




def main():
    create_test_set()
    create_train_set()
    M_x = 100
    M_t = 1000
    x = np.linspace(-5, 5, M_x)
    t = np.linspace(0, 3.14, M_t)
    v = np.load("./Data/Soln.npy")
    for i in range(3):
        T, X = np.meshgrid(t[0:-1:20], x)
        fig = plt.figure()
        ax = plt.axes(projection='3d')
        ax.contour3D(X, T, v[:, :, i], 50, cmap='viridis')
        ax.set_xlabel('x')
        ax.set_ylabel('t')
        ax.set_zlabel('U(x,t)')
        plt.show()
main()










