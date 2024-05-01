

"""
This script performs quantum circuit simulations using Qiskit and QuTiP libraries.

Functions:
    - GaussianTheta(num_seq, sigma, num_instance): 
        Generates a matrix of random values following a Gaussian random process.
    
    - module(ThetaX, K, n, circuit, qr, M): 
        Applies the quantum gates.
    
    - create_state_matrix(N, K, M, Correlation): 
        Creates a matrix to store the evolution of quantum state.
    
    - main(): 
        Main function to run the quantum circuit simulations.
"""

from qutip import *
from qiskit import *
from math import pi
import numpy as np
import scipy as scipy
import pandas as pd
from qiskit.quantum_info import Statevector



def GaussianTheta(num_seq, sigma, num_instance):
    
    """
    Generates a matrix of random values following a Gaussian random process.

    Args:
        num_seq (int): Number of Modules.
        sigma (float): Correlation lenght.
        num_instance (int): Number of instances.

    Returns:
        numpy.ndarray: Matrix of random values.
    """
    Delta = scipy.linalg.toeplitz( - np.arange(num_seq), np.arange(num_seq) )


    CovMat = np.exp( - np.power(Delta,2) / ( 2 * sigma ** 2 ) )


    eig_val, eig_vec = np.linalg.eigh(CovMat)
    eig_val = np.maximum(eig_val, 0.0)

    L = np.dot( np.sqrt(np.diag(eig_val)) , eig_vec.T )
    K = np.dot( np.random.randn( num_instance, num_seq ) , L )

    return K

def module(ThetaX,K,n,circuit,qr,M):
    """
    Applies quantum gates to the quantum circuit.

    Args:
        ThetaX (numpy.ndarray):
        K (int): Number of modules.
        n (int): Index of the instance.
        circuit (QuantumCircuit): Quantum circuit to apply gates on.
        qr (QuantumRegister): Quantum register.
        M (int): Number of qubits.

    Returns:
        None
    """
    circuit.rzz(theta=-pi/2, qubit1=qr[0], qubit2=qr[M-1])
    
    for j in range(1, M):
        
        circuit.rzz(theta=-pi/2, qubit1=qr[j-1], qubit2=qr[j])
    for j in range(M):
        circuit.rz(-pi/(2),j)
    for j in range(M):
        circuit.rx(-2*ThetaX[n,K-1,j],M-1-j)# apparently qiskit counts qubits inversly that is why I said M-1-j instead of j

def create_state_matrix(N, K, M, Correlation):
    """
    Creates random parameters using the random Guassian process.

    Args:
        N (int): Number of samples.
        K (int): Number of modules.
        M (int): Number of qubits.
        Correlation (float): Correlation value.

    Returns:
        numpy.ndarray: random parameters.
    """
    
    ThetaX = np.ones((N, K, M))
    for i in range(M):
        Seed = M + 498 + i
        np.random.seed(Seed)
        ThetaX[:, :, i] = GaussianTheta(K, Correlation, N)
    return ThetaX




def main():

    N=1#number of samples
    K=20# number of modules

    Correlation=20
    M=8
    ThetaX = create_state_matrix(N, K, M, Correlation)
    cr=ClassicalRegister(M)
    qr=QuantumRegister(M)
    circuit = QuantumCircuit(qr, cr)

    state_store=np.zeros((N,2**M,K),dtype=complex)

    for n in range(N):
        state_0 = Statevector.from_int(0, 2**M)
        state_store[n,:,0]=state_0
        #print(n, end="      \r")

        circuit = QuantumCircuit(qr, cr)

        for k in range(1,K):
            #print('k',k)
            module(ThetaX,k,n,circuit,qr,M)
            state = state_0.evolve(circuit)
            #print(state)
            state_store[n,:,k]=state


    si = qeye(2)
    sx = sigmax()
    sy = sigmay()
    sz = sigmaz()

    sx_list = []
    sy_list = []
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
    L=4
    expect_store=np.zeros((N,K,9*(L-1)+3*L))
    for k in range(K):
        for n in range(N):
            psi=Qobj(state_store[n,:,k],dims = [[2]*M
                                ,[1]*M])
            tmp=np.hstack(np.array([[expect(sx_list[f],psi) for f in range(0,L)],[expect(sy_list[f],psi) for f in range(0,L)],[expect(sz_list[f],psi) for f in range(0,L)]
                    ,[(expect(sx_list[0]*sx_list[f],psi)) for f in range(1,L) ]
                    ,[(1 if f!=0 else -1j)*(expect(sx_list[0]*sy_list[f],psi)) for f in range(1,L) ]
                    ,[(1 if f!=0 else -1j)*(expect(sx_list[0]*sz_list[f],psi)) for f in range(1,L) ]
                    ,[(1 if f!=0 else -1j)*(expect(sy_list[0]*sx_list[f],psi)) for f in range(1,L) ]
                    ,[(expect(sy_list[0]*sy_list[f],psi)) for f in range(1,L) ]
                    ,[(1 if f!=0 else -1j)*(expect(sy_list[0]*sz_list[f],psi)) for f in range(1,L) ]
                    ,[(1 if f!=0 else -1j)*(expect(sz_list[0]*sx_list[f],psi)) for f in range(1,L)]
                    ,[(-1j if f==0 else 1)*(expect(sz_list[0]*sy_list[f],psi)) for f in range(1,L)]
                    ,[(expect(sz_list[0]*sz_list[f],psi)) for f in range(1,L) ]],dtype=object))
            expect_store[n,k]=tmp
            np.savez( "EXPECTS_STATES"+str(M) + "qubits_" + str(K) + "sequcences" + str(N) + "trajectories_sigma=" + str(Correlation) + ".npz", 
         state=state_store,Theta=ThetaX,Expect=expect_store )
    return(expect_store)
if __name__ == "__main__":
    main()
