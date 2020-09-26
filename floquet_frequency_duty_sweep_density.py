# import required modules
import qutip
import numpy as np
from qutip import *
import matplotlib.pyplot as plt
from scipy import signal
import matplotlib

from mpl_toolkits.axes_grid.inset_locator import (inset_axes, InsetPosition, mark_inset)

#s et plot parameters and fonts
matplotlib.rcParams['text.usetex'] = True
plt.rcParams["font.family"]='serif'

#periodic switch function
def onoff(t, duty, Omega):
  return (signal.square(t*Omega, duty = duty)+1)/2

#Hamiltonian time dependant coefficients
def H1coeff(time, args):
  duty = args['duty']
  Omega = args['Omega']
  return -onoff(time, duty, Omega)+1 # needed to add this weird shift> 
                                     #  high: free expansion
                                     #  low:  harmonic oscillator on

# calculates the index of the ground state i.e. the index of the state with the highest overlap with the harmonic oscillator
def find_GS_index(psi0, f_energies, f_modes_0):
  overlap = np.zeros([f_energies.size,2])

  for state in np.arange(len(f_energies)):
    overlap[state,0] = np.abs(psi0.overlap(f_modes_0[state]))**2 # calculate overlap of Floquet mode and initial wavefunction
  overlap[:,1] = f_energies
  ind = overlap[:,0].argsort()         # return index to sort for largest overlap
  index = ind[::-1][0]                 # reverse index array to account for bottom to top sorting, extract ground state
  return index                         # return index
#_____________________________________________________________________________________________________________________________________________________


N = 40                     # number of levels in the Hilbert space
hbar = 1
w = 2*np.pi*29.5          # HO trap frequency in Hz
a = destroy(N)             # annihilation operator

#initial wave function
psi0 = tensor(basis(N,0))

# time-independant Hamiltonians
H0 = hbar*w*(a.dag()*a+0.5)                                    # harm oscillator Hamiltonian
H1 = -w/4.*(a.dag()*a.dag() + a*a + 2*a.dag()*a + 1)*qeye(N)   # free expansion Hamiltonian 
H = [H0,[H1, H1coeff]]                                         # full time-dependant Hamiltonian


Omegas = 2*np.pi*np.linspace(70, 500, 4)

def calc_overlap_different_switching_frequencies(duty_cycle, Omegas,  Ncycles):
  density = np.zeros(np.size(Omegas))
  d = duty_cycle
  Omega = 0
  args = {'duty': d, 'Omega': Omega} # set parameters

  i = 0
  for Omega in Omegas:                                         # sweep through switching frequencies
    T = 2*np.pi/Omega # switching period
    # args = {'duty': d, 'Omega': Omega} # set parameters
    args['duty'] = d
    args['Omega'] = Omega

    time = T * Ncycles
    # Calculate Floquet modes and energies at t = 0
    f_modes_0, f_energies = floquet_modes(H, T, args)

    f_coeff = floquet_state_decomposition(f_modes_0, f_energies, psi0)
    psi_t = floquet_wavefunction(f_modes_0, f_energies, f_coeff, time)
    density[i]= np.abs(f_modes_0[find_GS_index(psi0, f_energies, f_modes_0)].overlap(psi_t))**2
    i+=1

  return density

def calc_overlap_single_frequency(duty_cycle, Omega, Ncycles):
  d = duty_cycle
  Omega = 2*np.pi*Omega
  args = {'duty': d, 'Omega': Omega} # set parameters

  T = 2*np.pi/Omega # switching period
  # args = {'duty': d, 'Omega': Omega} # set parameters
  # args['duty'] = d
  # args['Omega'] = Omega

  time = T * Ncycles
  # Calculate Floquet modes and energies at t = 0
  f_modes_0, f_energies = floquet_modes(H, T, args)

  f_coeff = floquet_state_decomposition(f_modes_0, f_energies, psi0)
  psi_t = floquet_wavefunction(f_modes_0, f_energies, f_coeff, time)
  density= np.abs(f_modes_0[find_GS_index(psi0, f_energies, f_modes_0)].overlap(psi_t))**2

  return density



# #____________________________________________
# #calculate overlaps for different duty cycles 
# overlaps = np.zeros(5, dtype=np.ndarray)
# i=0
# for duty in [0.1,0.2,0.3,0.4,0.5]:
#   overlaps[i] = calc_overlap(duty, Omegas, 1e6)
#   i+=1

# plt.figure(figsize=(10,8))
# plt.plot(Omegas/w, overlaps[0], '.')
# plt.plot(Omegas/w, overlaps[1], '.')
# plt.plot(Omegas/w, overlaps[2], '.')
# plt.plot(Omegas/w, overlaps[3], '.')
# plt.plot(Omegas/w, overlaps[4], '.')
# #____________________________________________

duties = np.linspace(0.05, 0.9, 32)
overlaps = np.zeros(len(duties), dtype=np.ndarray)
i=0

for duty in duties:
  overlaps[i] = calc_overlap_single_frequency(duty, 500, 1e6)
  i+=1

overlaps1 = np.zeros(len(duties), dtype=np.ndarray)
i=0

for duty in duties:
  overlaps1[i] = calc_overlap_single_frequency(duty, 80, 1e6)
  i+=1

overlaps2 = np.zeros(len(duties), dtype=np.ndarray)
i=0

for duty in duties:
  overlaps2[i] = calc_overlap_single_frequency(duty, 130, 1e6)
  i+=1

def overlap_theo(d):
  return (2*d**0.25)/(1+d**0.5)



fig, ax1 = plt.subplots(figsize=(13,8))

ax1.plot(duties, overlaps, '.',label='$\Omega/\omega \simeq 17$')
ax1.plot(duties, overlaps1, '.', label='$\Omega/\omega \simeq 2.7$')
ax1.plot(duties, overlaps2, '.', label='$\Omega/\omega \simeq 3.4$')
ax1.plot(np.linspace(0.05, 0.9, 200), overlap_theo(np.linspace(0.05, 0.9, 200)), color='red', label='analytical solution')
ax1.set_xlabel('duty cycle D',fontsize = 20)
ax1.set_ylabel(r'$| \langle \Psi_{0,Floquet}(t) | \Psi_{0,HO} \rangle |^2$ ', fontsize=20)

ax2 =plt.axes([0,0,1,1])
ip = InsetPosition(ax1, [0.45,0.15, 0.5,0.5])
ax2.set_axes_locator(ip)


ax2.plot(duties, (overlap_theo(duties)-overlaps)/overlap_theo(duties), '.', label='$\Omega/\omega \simeq 17$')
ax2.plot(duties, (overlap_theo(duties)-overlaps1)/overlap_theo(duties), '.', label='$\Omega/\omega \simeq 2.7$')
ax2.plot(duties, (overlap_theo(duties)-overlaps2)/overlap_theo(duties), '.', label='$\Omega/\omega \simeq 3.4$')
ax2.set_xlabel('duty cycle D', fontsize=18)
ax2.set_ylabel('rel. deviation', fontsize=18)


ax1.tick_params(axis='both', which='major', labelsize=20, direction='in', bottom=True, top=True, left=True, right=True, length=5)
ax2.tick_params(axis='both', which='major', labelsize=20, direction='in', bottom=True, top=True, left=True, right=True, length=3)
ax1.legend(fontsize=20, loc=2, bbox_to_anchor=(1.05,1))


plt.show()