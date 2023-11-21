import numpy as np
from qutip import *
from math import sqrt
from scipy import *
import scipy.linalg

from numpy.linalg import eig
#import matplotlib.pyplot as plt
import pandas as pd
#import matplotlib.pyplot as plt
#import tensorflow as tf
#from tensorflow.keras import Sequential
#from tensorflow.keras.layers import Dense, Dropout,LSTM
import sys
import scipy.linalg
seed_test=1451
num_time_steps=24
Size=24
tau_max=3
taulist=np.linspace(0,tau_max,num_time_steps)
M=11
L=int((M-1)/2)+1
#taulist=np.linspace(0,5,48)
num_test=24

J=1
psi_list = [(basis(2,0)+0*basis(2,1))/np.sqrt(2) for n in range(M)]
psi0 = tensor(psi_list)
def evolution_step(Seed,num):
    """
       Simulate the evolution of a quantum system driven by periodic functions.

       Parameters:
       - seed (int): Seed for random number generation under random Gaussian field.
       - num (int): Number of simulations.

       Returns:
       - Result (list): Expectation value of obsevables.
       - B (list): List of random fields.
       - P1 (array): Array containing real part of the final state for each simulation.
       - P2 (array): Array containing imaginary part of the final state for each simulation.
       """
    global b1,b2,b3,B,exx
    si = qeye(2)
    sx = sigmax()
    sy = sigmay()
    sz = sigmaz()

    sx_list = []
    sy_list = []
    sz_list = []

    for n in range(M):
        op_list = []
        for m in range(M):
            op_list.append(si)

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

    for j in range(num):
        alpha=np.random.uniform(low=0,high=1,size=1)
        beta=np.sqrt(1-alpha[0]**2)
        gamma=np.random.choice([-1, 1], 1)
        psi_list = [(alpha[0]*basis(2,0)+gamma[0]*beta*basis(2,1)) for n in range(M)]
        psi0 = tensor(psi_list)
        count=count+1
        print(count,end="      \r")

        exx=np.zeros((len(taulist),9*(L-1)+3))
        step_size=3
        cc=np.random.uniform(low=5, high=20, size=(1) )
        b=np.random.uniform(low=-3, high=3, size=(1) )
        a=np.random.uniform(low=0.1, high=3, size=(1) )
       # b=np.random.uniform(low=-step_size, high=step_size, size=(1) )
        c=np.random.uniform(low=-step_size, high=step_size, size=(1) )
        #print('a',a)
        def magnetic_f(t,args):
            return b*np.sin(a*t)
       
        
        def magnetic_field(t):
            return b*np.sin(a*t)
       
        
        B.append(magnetic_field(taulist))

        
       

        Hy0=0
        for n in range(M):
            Hy0 += sy_list[n]
        Hz0=0
        for n in range(M):
            Hz0 += sz_list[n]
        Hx0=0
        for n in range(M):
            Hx0 += sx_list[n]
            
         
            
        H1 =J* sz_list[M-1]*sz_list[0]
        for n1 in range(M-1) :
            H1 +=J*sz_list[n1]*sz_list[n1+1]
        
        
      
        #args = {'CC': cc,'DD': dd,'A': a,'B': b,'C': c}
         #h_t= [H1,[Hx0, lambda t, args :(args['A'] * (t< args['CC']) )+ (args['B'] * (args['CC']<t < args['DD']) )+(args['C'] * (t> args['DD']))]]
        h_t = [H1,[Hx0,magnetic_f]]    
        #E0,psi0= (H1+b[0]*Hx0).groundstate()
        
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

            idx[0] += 1
            
            return(psi)
        
        mesolve(h_t, psi0, taulist,[] ,process_rho,options=None, _safe_mode=True)

        Result.append(list(exx.flatten()))
        #print(B)

    return(Result,B)
Result_test_step,Data_test_step=evolution_step(seed_test,num_test)
df_d_test = pd.DataFrame(Data_test_step)#, header=None)
df_d_test.to_csv("Data_test_periodic_tau=%d,M=%d,num_test=%d,seed_test=%d.dat"% (tau_max, M,num_test,seed_test), header=False,index=False)
df_p_test = pd.DataFrame(Result_test_step)#, header=None)
df_p_test.to_csv("Result_test_periodic_tau=%d,M=%d,num_test=%d,seed_test=%d.dat"% (tau_max, M,num_test,seed_test), header=False,index=False)
