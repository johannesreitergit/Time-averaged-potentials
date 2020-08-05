# import required modules
import qutip
import numpy as np
from qutip import *
import matplotlib.pyplot as plt
from scipy import signal
import matplotlib

#s et plot parameters and fonts
matplotlib.rcParams['text.usetex'] = True
plt.rcParams["font.family"]='serif'

#periodic switch function
def onoff(t, duty, Omega):
  return (signal.square(t*Omega, duty = duty)+1)/2

# #smoothed square switch function 
# def onoff(t, duty):
#   z = duty
#   return -1/np.pi * (np.arctan(np.sin(2*np.pi * f * t)/0.01+ z) + np.pi/2) +1


#Hamiltonian time dependant coefficients
def H1coeff(time, args):
  duty = args['duty']
  Omega = args['Omega']
  return -onoff(time, duty, Omega)+1 # needed to add this weird shift> 
                                     #  high: free expansion
                                     #  low:  harmonic oscillator on

#_____________________________________________________________________________________________________________________________________________________


N = 64                    # number of levels in the Hilbert space
hbar = 1
w = 10000                  # HO trap frequency in Hz
a = destroy(N)             # annihilation operator

#initial wave function
psi0 = tensor( basis(N,0))

# time-independant Hamiltonians
H0 = hbar*w*(a.dag()*a+0.5)                                    # harm oscillator Hamiltonian
H1 = -w/4.*(a.dag()*a.dag() + a*a + 2*a.dag()*a + 1)*qeye(N)   # free expansion Hamiltonian 
H = [H0,[H1, H1coeff]]                                         # full time-dependant Hamiltonian

duty = np.linspace(0,1,64)
Omegas = 2*np.pi*np.linspace(10e3, 300e3, 64)

density = np.array([])

for d in duty:                                                 # sweep through different duty cycles
  for Omega in Omegas:                                         # sweep through switching frequencies

    T = 2*np.pi/Omega # switching period
    args = {'duty': d, 'Omega': Omega} # set parameters

    # Calculate Floquet modes and energies at t = 0
    f_modes_0, f_energies = floquet_modes(H, T, args ,sort=True)

    # calculates the index of the ground state i.e. the index of the state with the highest overlap with the harmonic oscillator

    overlap = np.zeros([f_energies.size,2])
    for state in np.arange(len(f_energies)):
      overlap[state,0] = np.abs(psi0.overlap(f_modes_0[state]))**2 # calculate overlap of Floquet mode and initial wavefunction
    overlap[:,1] = f_energies
    ind = overlap[:,0].argsort()         # return index to sort for largest overlap
    index = ind[::-1][0]                 # reverse index array to account for bottom to top sorting, extract ground state
    overlap = overlap[:,0]
    density = np.append(density, overlap[index])


x,y = np.meshgrid(duty, Omegas/(w*2*np.pi))
z = density.reshape((np.size(Omegas), np.size(duty)))

fig = plt.figure(figsize=(10,8))
ax = plt.axes(projection='3d')
surf = ax.plot_surface(x, y, z, rstride=1, cstride=1, cmap='coolwarm', edgecolor='none')
ax.set_xlabel('Duty cycle D')
ax.set_ylabel('Switching frequency $\Omega/\omega$')
ax.set_zlabel('PDF')
fig.colorbar(surf, aspect=5)

plt.show()