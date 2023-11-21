import numpy as np
from qutip import *
from math import sqrt
from scipy import *
import pandas as pd
import sys
from scipy import *
import scipy.linalg
from numpy.linalg import eig

num_time_steps=100
tau_max=20
taulist=np.linspace(0,tau_max,num_time_steps)

def C_B(delta_t, correlation_time):
    return(np.exp(- delta_t**2/(2 *correlation_time**2)))

delta = scipy.linalg.toeplitz(-np.arange(num_time_steps), np.arange(num_time_steps))
c_l=np.random.uniform(low=1,high=9,size=1)



def evolution(Seed,num):
    
    """
    Simulate the evolution of a quantum system driven by random Gaussian functions.

    Parameters:
    - seed (int): Seed for random number generation under random Gaussian field.
    - num (int): Number of simulations.

    Returns:
    - Result (list): Expectation value of obsevables.
    - B (list): List of random fields.
    - P1 (array): Array containing real part of the final state for each simulation.
    - P2 (array): Array containing imaginary part of the final state for each simulation.
    """
     
    si = qeye(2)
    sx = sigmax()
    sy = sigmay()
    sz = sigmaz()

    sx_list = []
    sy_list = []
    sz_list = []

    for n in range(M):
        op_list = M*[si]

        op_list[n] = sx
        sx_list.append(tensor(op_list))

        op_list[n] = sy
        sy_list.append(tensor(op_list))

        op_list[n] = sz
        sz_list.append(tensor(op_list))
    np.random.seed(Seed)
    Result=[]
    count=0
    B=[]
    state_param=[]
    P1=np.zeros((num,num_time_steps,2**(M)))
    P2=np.zeros((num,num_time_steps,2**(M)))
    for j in range(num):
        alpha=np.random.uniform(low=0,high=1,size=1)#np.array([np.sqrt(1)])#np.random.uniform(low=0,high=1,size=1)
        beta=np.sqrt(1-alpha[0]**2)
        gamma=np.random.choice([-1, 1], 1)#np.array([1])
        psi_list = [(alpha[0]*basis(2,0)+gamma[0]*beta*basis(2,1)) for n in range(M)]
        psi0 = tensor(psi_list)
 
        c_l=np.random.uniform(low=0.2,high=9,size=1)
        C = C_B(delta, c_l[0])
        
        values, vectors = np.linalg.eigh(C)
        assert np.all(np.logical_or(values>0.0, np.isclose(values, 0.0)))
        values = np.maximum(values, 0.0)
        

        V_B=np.dot(vectors, np.sqrt(diag(values)))
        assert np.allclose(np.dot(V_B, V_B.T.conj()), C)        


        count=count+1
        print(count,end="      \r")
        L=int((M-1)/2)+1
        exx=np.zeros((len(taulist),9*(L-1)+3))
        B0=(np.dot(V_B,np.random.randn(len(taulist))) )
        assert B0.dtype == float
        S= Cubic_Spline(taulist[0], taulist[-1],1.5* B0)
        B.append(S(taulist))
       
        #state=[alpha[0],beta,gamma[0]]
        #state_param.append(state)

        
        Hy0=0
        for n in range(M):
            Hy0 += sy_list[n]
        Hz0=0
        for n in range(M):
            Hz0 += sz_list[n]
        Hx0=0
        for n in range(M):
            Hx0 += sx_list[n]
            
            
            
        H1 = sz_list[M-1]*sz_list[0]
        for n1 in range(M-1) :
            H1 +=sz_list[n1]*sz_list[n1+1]
        #H1 += sz_list[M-1]*sz_list[1]
        #for n1 in range(M-2) :
         #   H1 +=sz_list[n1]*sz_list[n1+2]

        
        
        
       

        H = [H1,[Hx0,S]]
       # E0,Psi0= (H1+B[j][0]*Hx0).eigenstates(eigvals=1)
        #psi0=Psi0[0]
        idx = [0]
        
        def process_rho(tau, psi):

            tmp=np.hstack(np.array([[expect(sz_list[0],psi)],[expect(sy_list[0],psi) ],[expect(sx_list[0],psi) ]
                ,[(expect(sx_list[0]*sx_list[f],psi)) for f in range(1,L) ]
                ,[(1 if f!=0 else -1j)*(expect(sx_list[0]*sy_list[f],psi)) for f in range(1,L) ]
                ,[(1 if f!=0 else -1j)*(expect(sx_list[0]*sz_list[f],psi)) for f in range(1,L) ]
                ,[(1 if f!=0 else -1j)*(expect(sy_list[0]*sx_list[f],psi)) for f in range(1,L) ]
                ,[(expect(sy_list[0]*sy_list[f],psi)) for f in range(1,L) ]
                ,[(1 if f!=0 else -1j)*(expect(sy_list[0]*sz_list[f],psi)) for f in range(1,L) ]
                ,[(1 if f!=0 else -1j)*(expect(sz_list[0]*sx_list[f],psi)) for f in range(1,L)]
                ,[(-1j if f==0 else 1)*(expect(sz_list[0]*sy_list[f],psi)) for f in range(1,L)]
                ,[(expect(sz_list[0]*sz_list[f],psi)) for f in range(1,L) ]]))
            if not np.allclose(np.imag(tmp), 0.0):
                raise RuntimeError('not all expectation values real')
            exx[idx[0],:] = np.real(tmp)
            P1[j,idx[0],:]= np.real(psi)[:,0]
            P2[j,idx[0],:]= np.imag(psi)[:,0]
                  
            idx[0] += 1
            #print(H)
            return(psi)

        mesolve(H, psi0, taulist,[] ,process_rho, options=None, _safe_mode=True)

        Result.append(list(exx.flatten()))
        #print(B)

    return(Result,B,P1,P2)





M=7
num_train=0
num_test=1000
seed_train=123
seed_test=1451

Result_test,Data_test,P1_test,P2_test=evolution(seed_test,num_test)
Result_train,Data_train,P1_train,P2_train=evolution(seed_train,num_train)
#print('c_l',c_l)
df_p_train = pd.DataFrame(Result_train)
df_p_train.to_csv("Result_train_tau=%d,M=%d,num_train=%d,seed_train=%d.dat"% (tau_max, M,num_train,seed_train), header=False,index=False)
df_p_test = pd.DataFrame(Result_test)#, header=None)

df_p_test.to_csv("Result_test_tau=%d,M=%d,num_test=%d,seed_test=%d.dat"% (tau_max, M,num_test,seed_test), header=False,index=False)
df_d_train = pd.DataFrame(Data_train)
df_d_train.to_csv("Data_train_tau=%d,M=%d,num_train=%d,seed_train=%d.dat"% (tau_max, M,num_train,seed_train), header=False,index=False)
df_d_test = pd.DataFrame(Data_test)#, header=None)
df_d_test.to_csv("Data_test_tau=%d,M=%d,num_test=%d,seed_test=%d.dat"% (tau_max, M,num_test,seed_test), header=False,index=False)
