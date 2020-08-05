#import required modules
import qutip
import numpy as np
from qutip import *
import matplotlib.pyplot as plt
from scipy import signal
from scipy.special import eval_hermite
import matplotlib

#s et plot parameters and fonts
matplotlib.rcParams['text.usetex'] = True
# matplotlib.rcParams['pdf.fonttype'] = 42
# matplotlib.rcParams['ps.fonttype'] = 42
# matplotlib.rcParams['mathtext.fontset'] = 'stix'
# matplotlib.rcParams['font.family'] = 'STIXGeneral'
plt.rcParams["font.family"]='serif'


#periodic switch function
def onoff(t, duty, Omega):
  return (signal.square(t*Omega, duty = duty)+1)/2

# #smoothed square switch function 
# def onoff(t, duty):
#   d = duty
#   return -1/np.pi * (np.arctan(np.sin(2*np.pi * f * t)/0.01+ 0) + np.pi/2) +1

#Hamiltonian time dependant coefficients
def H1coeff(time, args):
  duty = args['duty']
  Omega = args['Omega']
  return -onoff(time, duty, Omega)+1 #needed to add this weird shift> 
                              #  high: free expansion
                              #  low:  harmonic oscillator on

# Energy sorter sorts energies for the largest wavefunction overlap with the initial wavefunction
def energy_sorter(psi0, f_energies, f_modes_0):
  sorted_energies = np.zeros([N])
  overlap = np.zeros([f_energies.size,2])

  for state in np.arange(len(f_energies)):
    overlap[state,0] = np.abs(psi0.overlap(f_modes_0[state]))**2 # calculate overlap of Floquet mode and initial wavefunction
  overlap[:,1] = f_energies
  index = overlap[:,0].argsort()         # return index to sort for largest overlap
  overlap = overlap[index]               # resort array including overlaps[0] and energies[1]
  sorted_energies = overlap[:,1]         # sorted energies
  sorted_energies = sorted_energies[::-1]

  return sorted_energies


# calculates the index of the ground state i.e. the index of the state with the highest overlap with the harmonic oscillator
def find_GS_index(psi0, f_energies, f_modes_0):
  overlap = np.zeros([f_energies.size,2])

  for state in np.arange(len(f_energies)):
    overlap[state,0] = np.abs(psi0.overlap(f_modes_0[state]))**2 # calculate overlap of Floquet mode and initial wavefunction
  overlap[:,1] = f_energies
  ind = overlap[:,0].argsort()         # return index to sort for largest overlap
  index = ind[::-1][0]                 # reverse index array to account for bottom to top sorting, extract ground state
  return index                         # return index

# calculates the index of the ground state i.e. the index of the state with the highest overlap with the harmonic oscillator
def find_ES_index(psi0, f_energies, f_modes_0):
  overlap = np.zeros([f_energies.size,2])

  for state in np.arange(len(f_energies)):
    overlap[state,0] = np.abs(psi0.overlap(f_modes_0[state]))**2 # calculate overlap of Floquet mode and initial wavefunction
  overlap[:,1] = f_energies
  ind = overlap[:,0].argsort()         # return index to sort for largest overlap
  index = ind[::-1][1]                 # reverse index array to account for bottom to top sorting, extract ground state
  return index                         # return index

#effective trap frequency for different duty cycles
def w_eff_duty(x):
  return w*np.sqrt(x)

# returns the Harmonic oscillator wavefunction for state n
def ho_state(n,x, w=10000):
  return  1/np.sqrt(2**float(n)*np.math.factorial(float(n)))*(m*w/(np.pi*hbar))**0.25*np.exp(-(m*w*x**2)/(2*hbar))*eval_hermite(np.array([n]),np.sqrt(m*w/hbar)*x)

#_____________________________________________________________________________________________________________________________________________________


#constants
N = 170                 # number of levels in the Hilbert space
hbar = 1
m = 1
w = 2 * np.pi * 10000   # trap frequency (Hz)
a = destroy(N)          # annihilation operator

# Time constants
duty = 0.2				# duty cycle
f = 200e3                # switch function frequency in Hz
Omega = 2*np.pi*f 
T = 1/f                 # switch period

#initial wave function
psi0 = tensor( basis(N,0)) #initial state
args = {'duty': duty, 'Omega': Omega}

# calculate matrix-like harmonic oscillator states where each column represents an ho eigenstate (i.e. N) and each row a position in space 
ho_states = np.array([])
x = np.linspace(-0.03,0.03,256)
for n in range(N):
  ho_states = np.append(ho_states, ho_state(n, x, w))
ho_states = ho_states.reshape(N,len(x)) # reshape to have matrix form

# Hamiltonian
H0 = hbar*w*(a.dag()*a+0.5)                                    # harm oscillator Hamiltonian
H1 = -m*w/4.*(a.dag()*a.dag() + a*a + 2*a.dag()*a + 1)*qeye(N) # time-dependent Hamiltonian 
H = [H0,[H1, H1coeff]]                                         # full time-dependant Hamiltonian

# Calculate Floquet modes and energies at t = 0
f_modes_0, f_energies = floquet_modes(H, T, args, sort=True)

# find groundstate index
indexGS = find_GS_index(psi0, f_energies, f_modes_0)

# calculate spatial distribution of ground state
gstate = abs(np.dot(ho_states.T,f_modes_0[indexGS]))**2 # floquet ground state
ho_gstate = abs(ho_states[0])**2                        # ho ground state
w_eff = w*np.sqrt(args['duty'])
power = abs(ho_state(0,x,w=w_eff))**2

#plot
fig, ax = plt.subplots(figsize=(10,8))
plt.tick_params(axis='both', which='major', labelsize=15)
title = 'Floquet ground state for $\Omega=${0}kHz and $D=${1}'
plt.title(title.format(args['Omega']/(2*np.pi*1e3), args['duty']), fontsize = 20)
plt.xlabel('x', fontsize=20)
plt.ylabel('$|\Psi|^2$', fontsize=20)
plt.plot(x, gstate, '.', label='Numerical $\Psi_{0,Floquet}$')
plt.plot(x, ho_gstate, label='Harmonic oscillator ground state',color='green' )
plt.plot(x, power, label='Harmonic oscillator with $\omega_{eff} = \omega_{trap}\cdot\sqrt{D}$', color='red')
plt.legend(loc='best',fontsize=10)
# plt.grid()

plt.show()