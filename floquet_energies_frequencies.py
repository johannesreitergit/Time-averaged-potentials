#import required modules
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

#effective trap frequency for different duty cycles
def w_eff_duty(x):
  return w*np.sqrt(x)

# Calculate the time-averaged energy
def time_averaged_energy(quasienergies, Omegas):
  dQuasienergies = np.gradient(quasienergies)      # gradient
  avgE = quasienergies - Omegas * dQuasienergies   # time-averaged energy
  return avgE

#_____________________________________________________________________________________________________________________________________________________


N = 128                    # number of levels in the Hilbert space
hbar = 1
w = 10000                  # HO trap frequency in Hz
a = destroy(N)             # annihilation operator

#initial wave function
psi0 = tensor( basis(N,0))

# time-independant Hamiltonians
H0 = hbar*w*(a.dag()*a+0.5)                                    # harm oscillator Hamiltonian
H1 = -w/4.*(a.dag()*a.dag() + a*a + 2*a.dag()*a + 1)*qeye(N)   # free expansion Hamiltonian 
H = [H0,[H1, H1coeff]]                                         # full time-dependant Hamiltonian

nFreq = 500
f_switching = np.linspace(10e3,300e3,nFreq)
Omegas = 2 * np.pi * f_switching
switching_energies0 = np.array([])
switching_energies1 = np.array([])
switching_energies2 = np.array([])
switching_energies3 = np.array([])
duty = 0.1

# Calculate quasienergies for different switching frequencies
for Omega in Omegas:

  T = 2*np.pi/Omega # switching period
  args = {'duty': duty, 'Omega': Omega}

  # Calculate Floquet modes and energies at t = 0
  f_modes_0, f_energies = floquet_modes(H, T, args ,sort=True)
  energies = energy_sorter(psi0, f_energies, f_modes_0)

  switching_energies0 = np.append(switching_energies0, energies[0])
  switching_energies1 = np.append(switching_energies1, energies[1])
  switching_energies2 = np.append(switching_energies2, energies[2])
  switching_energies3 = np.append(switching_energies3, energies[3])

#plot
fig = plt.figure(figsize=(10,8))
plt.tick_params(axis='both', which='major', labelsize=15)
plt.plot(Omegas/(w*2*np.pi), 2*switching_energies0/(w*np.sqrt(duty)),  '.',label='numerical quasienergies $\epsilon_0$')
# plt.plot(Omegas/w, switching_energies1/w/np.sqrt(duty)-3/2,  '.',label='numerical quasienergies $\epsilon_1$')
# plt.plot(Omegas/w, switching_energies2/w/np.sqrt(duty)-7/2,  '.',label='numerical quasienergies $\epsilon_2$')
# plt.plot(Omegas/w, switching_energies3/w/np.sqrt(duty)-11/2,  '.',label='numerical quasienergies $\epsilon_3$')
# plt.ylim(0.98,1.08)
plt.title('Quasienergies for different switching frequencies $\Omega$', fontsize=20)
plt.xlabel('switching frequency $\Omega / \omega$', fontsize=20)
plt.ylabel('$2 \cdot \epsilon_{Floquet} / \omega_{eff}$', fontsize=20)
plt.grid()
plt.legend(fontsize=15, loc='best')


# Calculate time-averaged energy

avgE = time_averaged_energy(switching_energies0, Omegas)

#plot
fig = plt.figure(figsize=(10,8))
plt.tick_params(axis='both', which='major', labelsize=15)
plt.title('Time-averaged energy', fontsize = 20)
plt.plot(Omegas/w, avgE/w/2, '.')
plt.xlabel('switching frequency $\Omega / \omega$', fontsize=20)
plt.ylabel('Time averaged energy $\overline{E}_0 / \epsilon_{0, HO}$', fontsize=20)
plt.grid()

plt.show()
