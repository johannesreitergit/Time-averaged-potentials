#import required modules
import qutip
import numpy as np
from qutip import *
import matplotlib.pyplot as plt
from scipy import signal
from scipy.special import eval_hermite
import matplotlib

#set plot parameters and fonts
matplotlib.rcParams['text.usetex'] = True
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
def ho_state(n,x, w=10000, offset=0):
  return  1/np.sqrt(2**float(n)*np.math.factorial(float(n)))*(m*w/(np.pi*hbar))**0.25*np.exp(-(m*w*(x-offset)**2)/(2*hbar))*eval_hermite(np.array([n]),np.sqrt(m*w/hbar)*(x-offset))

#_____________________________________________________________________________________________________________________________________________________

#constants
N = 64                 # number of levels in the Hilbert space
hbar = 1
m = 1
w = 2 * np.pi * 29450   # trap frequency (Hz)
a = destroy(N)          # annihilation operator

# Time constants
duty = 0.2        # duty cycle
f = 500e3                # switch function frequency in Hz
Omega = 2*np.pi*f 
T = 1/f                 # switch period
t_start = 0             # start time
t_stop  = T             # stop time 
t_step  = t_stop/512    # step size
time = np.arange(t_start,t_stop,t_step) #time steps

#initial wave function
psi0 = tensor( basis(N,0)) #initial state
args = {'duty': duty, 'Omega': Omega}

# Hamiltonian
H0 = hbar*w*(a.dag()*a+0.5)                                    # harm oscillator Hamiltonian
H1 = -m*w/4.*(a.dag()*a.dag() + a*a + 2*a.dag()*a + 1)*qeye(N) # time-dependent Hamiltonian 
H = [H0,[H1, H1coeff]]                                         # full time-dependant Hamiltonian

# Calculate Floquet modes and energies at t = 0
f_modes_0, f_energies = floquet_modes(H, T, args, sort=True)
f_coeff = floquet_state_decomposition(f_modes_0, f_energies, psi0)

#find groundstate index |0>
indexGS = find_GS_index(psi0, f_energies, f_modes_0)
# find index of second (symmetric excited state) |2>
indexES = find_ES_index(psi0, f_energies, f_modes_0)

# solve time-dependant Schroedinger equation using Floquet theory for states |0> and |2>
micromotion0 = fsesolve(H, f_modes_0[indexGS], time, e_ops=[], T=T, args=args, Tsteps=np.size(time)) #T steps should be an even number
micromotion2 = fsesolve(H, f_modes_0[indexES], time, e_ops=[], T=T, args=args, Tsteps=np.size(time)) #T steps should be an even number

# calculate overlap of Floquet ground state
overlap_gstate = np.array([])
for t in range(len(time)):
  ovlp = micromotion0.states[t].overlap(f_modes_0[indexGS])
  overlap_gstate = np.append(overlap_gstate,ovlp)

# calculate overlap of Floquet excited state but with respect to Floquet ground state
overlap_estate = np.array([])
for t in range(len(time)):
  ovlp = micromotion2.states[t].overlap(f_modes_0[indexGS])
  overlap_estate = np.append(overlap_estate,ovlp)


# HO STATES 
# calculate matrix-like harmonic oscillator states where each column represents an ho eigenstate (i.e. N) and each row a position in space 
ho_states = np.array([])
x = np.linspace(-0.01,0.01,512)
for n in range(N):
  ho_states = np.append(ho_states, ho_state(n, x, w))
ho_states = ho_states.reshape(N,len(x)) # reshape to have matrix form

# PHASE
# create figure and axis objects with subplots()
fig,ax = plt.subplots(figsize=(11,10))
ax.tick_params(axis='both', which='major', labelsize=25)

# HO on
# ax.set_title('Micromotion of the Floquet ground state', fontsize=20)
ax.plot(time/T, onoff(time, args['duty'], args['Omega']), label='Harmonic potential ON', color='orange', alpha=0.30)
ax.fill_between(time/T, np.zeros(len(time)),onoff(time, args['duty'], args['Omega']), color='orange', alpha=0.30)

ax.tick_params(axis='both', which='major', labelsize=25, direction='in', bottom=True, top=True, left=True, right=True, length=5)
ax.plot(time/T, np.angle(overlap_gstate, deg=True)/360, label=r'phase angle of $\Psi_{0, Floquet}(t)$', color='green')
ax.set_ylabel('Phase $\Phi / 2\pi$', fontsize=20)
ax.plot(time/T, np.angle(np.exp(1j*time*f_energies[indexGS]), deg=True)/360, linestyle='-', label=r'$\Phi=arg(\exp(-i \epsilon_0 t))$') # observe: no minus sign in exponent due to -pi/pi convention
ax.set_ylim(min(np.angle(overlap_gstate, deg=True)/360), max(np.angle(overlap_gstate, deg=True)/360))
ax.legend(loc=7, fontsize=25)
ax.set_xlabel('time / T', fontsize=25)
# plt.grid()


# DENSITY
# create figure and axis objects with subplots()
fig, ax = plt.subplots(2,1, gridspec_kw={'height_ratios':[1,4]}, figsize=(13,10)) 
ax[1].tick_params(axis='both', which='major', labelsize=25, direction='in', bottom=True, top=True, left=True, right=False, length=5)
plt.subplots_adjust(right=0.87)
# HO on
# ax.set_title('Micromotion of the Floquet ground state', fontsize=20)
p0 = ax[1].plot(time/T, onoff(time, args['duty'], args['Omega']), color='orange', alpha=0.30,  label='Harmonic potential ON')
ax[1].fill_between(time/T, np.zeros(len(time)),onoff(time, args['duty'], args['Omega']), color='orange', alpha=0.30)

# micromotion |0>
p1 = ax[1].plot(time/T, abs(overlap_gstate)**2, label=r'$|0\rangle$') #, color='darkblue'
ax[1].set_ylim(0.995,1.0001)
ax[1].set_ylabel(r'$| \langle \Psi_{0, Floquet}(t) | \Psi_{0, Floquet}(0) \rangle |^2 $', fontsize=25)
ax[1].set_xlabel('time / T', fontsize=25)
ax[1].set_xlim(0,1)

# micromotion |2>
ax2=ax[1].twinx()
ax2.tick_params(axis='both', which='major', labelsize=25, direction='in', bottom=True, top=True, left=False, right=True, length=5)
p2 = ax2.plot(time/T, abs(overlap_estate)**2, label=r'$|2\rangle$', color='darkred')
ax2.set_ylim(-0.0001,0.005)
ax2.set_ylabel(r'$| \langle \Psi_{2, Floquet}(t) | \Psi_{0, Floquet}(0) \rangle |^2 $', fontsize=25)

# legend
plots = p0 + p1 + p2
labs = [l.get_label() for l in plots]

ax[1].legend(plots, labs, loc=7, fontsize=25)


# small little wave packets
x1 = np.linspace(0.4,0.6,300)
state1 = ho_state(0,x1,3000, offset=0.5)
expected = ho_state(0,x1,5000, offset=0.5)


x2 = np.linspace(0.0,0.2,300)
state2 = ho_state(0,x2,5000, offset=0.1)


x3 = np.linspace(0.8,1,300)
state3 = ho_state(0,x3,5000, offset=0.9)


ax[0].plot(x1, state1, color='darkblue')
ax[0].plot(x2, state2, color='darkblue')
ax[0].plot(x3, state3, color='darkblue')
ax[0].plot(x2, state2, color='grey', linestyle='dashed')
ax[0].plot(x3, state3, color='grey', linestyle='dashed')
ax[0].plot(x1, expected, color='grey', linestyle='dashed')
ax[0].arrow(0.2,4, 0.18,0, color='black', head_width=0.7, head_length=0.02)
ax[0].arrow(0.6,4, 0.18,0, color='black', head_width=0.7, head_length=0.02)

ax[0].set_xlim(0,1)
ax[0].set_ylim(0,8)
ax[0].set_xticks([])
ax[0].set_yticks([])
ax[0].axis('off')

plt.savefig('micromotion_density.pdf', format='pdf')



plt.show()