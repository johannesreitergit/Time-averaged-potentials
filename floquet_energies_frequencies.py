#import required modules
import qutip
import numpy as np
from qutip import *
import matplotlib.pyplot as plt
from scipy import signal
import matplotlib
from scipy.optimize import curve_fit

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
  sorted_overlap = overlap[:,0]         # sorted energies
  sorted_energies = sorted_energies[::-1]
  sorted_overlap = sorted_overlap[::-1]         # sorted energies

  return sorted_energies, sorted_overlap

#effective trap frequency for different duty cycles
def w_eff_duty(x):
  return w*np.sqrt(x)

# Calculate the time-averaged energy
def time_averaged_energy(quasienergies, Omegas):
  dQuasienergies = np.gradient(quasienergies)      # gradient
  avgE = quasienergies - Omegas * dQuasienergies   # time-averaged energy
  return avgE

#_____________________________________________________________________________________________________________________________________________________


N = 16                   # number of levels in the Hilbert space
hbar = 1
w = 2*np.pi* 29450                  # HO trap frequency in Hz
a = destroy(N)             # annihilation operator

#initial wave function
psi0 = tensor( basis(N,0))

# time-independant Hamiltonians
H0 = hbar*w*(a.dag()*a+0.5)                                    # harm oscillator Hamiltonian
H1 = -w/4.*(a.dag()*a.dag() + a*a + 2*a.dag()*a + 1)*qeye(N)   # free expansion Hamiltonian 
H = [H0,[H1, H1coeff]]                                         # full time-dependant Hamiltonian

nFreq = 512
f_switching = np.linspace(40e3,500e3,nFreq)
Omegas = 2 * np.pi * f_switching
switching_energies0 = np.zeros(np.size(f_switching))
switching_energies1 = np.zeros(np.size(f_switching))
switching_energies2 = np.zeros(np.size(f_switching))
switching_energies3 = np.zeros(np.size(f_switching))
switching_energies4 = np.zeros(np.size(f_switching))

# overlap0 = np.zeros(np.size(f_switching))

duty = 0.2

i=0
# Calculate quasienergies for different switching frequencies
for Omega in Omegas:

  T = 2*np.pi/Omega # switching period
  args = {'duty': duty, 'Omega': Omega}

	# Calculate Floquet modes and energies at t = 0
  f_modes_0, f_energies = floquet_modes(H, T, args ,sort=True)
  energies = energy_sorter(psi0, f_energies, f_modes_0)[0]
	# overlaps = energy_sorter(psi0, f_energies, f_modes_0)[1]
  switching_energies0[i] = energies[0]
	# overlap0[i] = overlaps[0]
  switching_energies1[i] = energies[1]
  switching_energies2[i] = energies[2]
	# switching_energies3[i] = energies[3]
	# switching_energies4[i] = energies[4]

  i+=1

def energy_spectrum(w, n):
	return hbar*w*(n+1/2)


def fit(x, S):
	return 1-(S**2)/(4*(1-x**2))

def guess(x, A, exp):
  return 1-(A/(1-(x)**exp))

# popt, pcov = curve_fit(guess, Omegas/w, switching_energies0/energy_spectrum(w*np.sqrt(duty), 0))
#plot
fig = plt.figure(figsize=(11,8))
plt.tick_params(axis='both', which='major', labelsize=20, direction='in', bottom=True, top=True, left=True, right=True, length=5)
plt.plot(Omegas/(w), switching_energies0/energy_spectrum(w*np.sqrt(duty), 0),  '.',label='numerical quasi-energy of the ground state') 
# plt.plot(Omegas/w, switching_energies1/energy_spectrum(w*np.sqrt(duty), 2),  '.',label='numerical quasienergies $\epsilon_2$')
# plt.plot(Omegas/w, switching_energies2/energy_spectrum(w*np.sqrt(duty), 4),  '.',label='numerical quasienergies $\epsilon_4$')
# plt.plot(Omegas/w, switching_energies3/w/np.sqrt(duty)-11/2,  '.',label='numerical quasienergies $\epsilon_3$')
# plt.plot(Omegas/w, switching_energies4/w/np.sqrt(duty)-15/2,  '.',label='numerical quasienergies $\epsilon_3$')
# plt.ylim(0.98,1.08)
# plt.title('Quasienergies for different switching frequencies $\Omega$', fontsize=20)
# plt.plot(Omegas/w, guess(Omegas/w, *popt), label='fit')
# plt.plot(np.linspace(1,17,300), guess(np.linspace(1,17,300), 0.2, 2 ))
plt.xlabel('switching frequency $\Omega / \omega_{HO, eff}$', fontsize=20)
plt.ylabel('$ \epsilon_{Floquet} /  \epsilon_{HO, eff}$', fontsize=20)
plt.grid(alpha=0.5)
plt.legend(fontsize=20, loc='best')
plt.savefig('quasienergies_switching_frequencies.pdf', format='png')


# print(popt)
# Calculate time-averaged energy

avgE = time_averaged_energy(switching_energies0, Omegas)

#plot
fig = plt.figure(figsize=(11,8))
plt.tick_params(axis='both', which='major', labelsize=20, direction='in', bottom=True, top=True, left=True, right=True, length=5)
# plt.title('Time-averaged energy', fontsize = 20)
plt.plot(Omegas/w, avgE/energy_spectrum(w*np.sqrt(duty), 0), '.')
plt.xlabel('switching frequency $\Omega / \omega$', fontsize=20)
plt.ylabel('time averaged energy $\overline{E}_0 / \epsilon_{HO, eff}$', fontsize=20)
plt.grid(alpha=0.5)
plt.savefig('time_averaged_energy_switching_frequencies.pdf', format='pdf')

plt.show()
