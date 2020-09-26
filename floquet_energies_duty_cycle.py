#import required modules
import qutip
import numpy as np
from qutip import *
import matplotlib.pyplot as plt
from scipy import signal
import matplotlib

#et plot parameters and fonts
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

# constants
N = 128                           # number of levels in the Hilbert space
hbar = 1
w = 2*np.pi* 29500                # HO trap frequency


# Time constants
f = 500e3                           # switch function frequency in Hz
Omega = 2*np.pi*f
T = 1/f                           # switch period

#operators
a = destroy(N)                    # annihilation operator

#initial wave function
psi0 = tensor( basis(N,0)) #initial state

H0 = hbar*w*(a.dag()*a+0.5)                                    # harm oscillator Hamiltonian
H1 = -w/4.*(a.dag()*a.dag() + a*a + 2*a.dag()*a + 1)*qeye(N)   # time-dependent Hamiltonian 
H = [H0,[H1, H1coeff]]                                         # full time-dependant Hamiltonian

#sweep through different duty cycles
duty = np.linspace(0,1,50)
duty_energies = np.array([])

for d in duty:
  args={'duty':d, 'Omega':Omega}
  f_modes_0, f_energies = floquet_modes(H, T, args ,sort=True) # Calculate Floquet modes and energies at t = 0
  
  energies = energy_sorter(psi0, f_energies,f_modes_0)         # sort quasienergies
  duty_energies = np.append(duty_energies, energies[0])        # quasienergies for the duty cycle

dc = np.linspace(0,1,300)
# plot
fig = plt.figure(figsize=(10,8))
plt.tick_params(axis='both', which='major', labelsize=20, direction='in', bottom=True, top=True, left=True, right=True, length=5)
plt.plot(dc, w_eff_duty(dc)/w, label='$\omega_{eff}=\omega_{trap} \sqrt{D}$', color='red', alpha=1)
plt.plot(duty, duty_energies*2/w,'.', label='numerical quasi-energies')#, color='blue'
# plt.title('Quasienergies for different duty cycles $D$', fontsize=20)
plt.xlabel('duty cycle D', fontsize=20)
plt.ylabel('$\epsilon_{0, Floquet}/\epsilon_{0,HO}$', fontsize=20)
plt.grid(alpha=0.6)
plt.legend(fontsize=20)

plt.savefig('quasienergies_duty_cycle.pdf', format='pdf')

plt.figure(figsize=(10,8))
plt.plot(duty, (w_eff_duty(duty)/w - duty_energies*2/w )/w_eff_duty(duty)/w, '.')


plt.show()