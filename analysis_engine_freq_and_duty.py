# import required modules
import numpy as np
from tkfilebrowser import askopenfilename  # GUI file browser
from tkinter import Tk
import matplotlib.pyplot as plt
import matplotlib
import time
from scipy.optimize import curve_fit
import qutip
from qutip import *
from scipy import signal

# Tk().withdraw()


#____________________________________________________________________________________________________________________________________________

# reads entire data set from file and extracts data and arguments (header columns)
def data_handler(delimiter='	'):
	filename = askopenfilename()		# opens file browser to specify file
	delimiter = '	'					# must match delimiter used in file
	print('Reading data from file...')
	tic = time.time()
	try:
		data = np.genfromtxt(filename, comments = '#', delimiter=delimiter, unpack = True, encoding=None, names=True, dtype=None) # read data
		arguments = np.asarray(data.dtype.names, dtype=str) # read header
		data = np.genfromtxt(filename, comments = '#', delimiter=delimiter, unpack = True) # read data

		toc = time.time()
		exc_time = toc - tic
		print('...finished in ', exc_time, 'seconds.')
	except:
		print('Reading data from file failed.')
	return data, arguments

# returns the index of a given keyword, keyword is a string
def keyword_index(arguments,keyword):
	index_keyword = np.where(arguments == keyword) # find index of keyword in arguments array
	return index_keyword

# takes multidimensional array and sorts it for a given keyword
def parameter_sort(data,arguments, keyword):
	index_keyword = keyword_index(arguments, keyword)
	sort_indices = data[index_keyword].argsort()
	# keyword = data[index_keyword]
	sorted_data = data[:,sort_indices]
	return sorted_data

# returns a unique parameter set for a specified keyword
def unique_parameter_set(data, arguments, keyword, return_index=False):
	parameters = np.unique(data[keyword_index(arguments,keyword)], return_index=return_index)
	return parameters

# return empty string in case label is None
def xstr(s):
	if s is None:
		return ''
	return str(s)	

def calc_overlap_Floquet(trap_freq):
	
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


	N = 50                     # number of levels in the Hilbert space
	hbar = 1
	w = 2*np.pi*trap_freq      # HO trap frequency in Hz
	a = destroy(N)             # annihilation operator

	#initial wave function
	psi0 = tensor(basis(N,0))

	# time-independant Hamiltonians
	H0 = hbar*w*(a.dag()*a+0.5)                                    # harm oscillator Hamiltonian
	H1 = -w/4.*(a.dag()*a.dag() + a*a + 2*a.dag()*a + 1)*qeye(N)   # free expansion Hamiltonian 
	H = [H0,[H1, H1coeff]]                                         # full time-dependant Hamiltonian


	Omegas = 2*np.pi*frequencies*29.5

	density = np.zeros(np.size(Omegas))

	d = duty/100
	Omega = 0
	args = {'duty': d, 'Omega': Omega} # set parameters

	print('Calculating Floquet stuff..')
	i = 0
	for Omega in Omegas:                                         # sweep through switching frequencies
	  T = 2*np.pi/Omega # switching period
	  # args = {'duty': d, 'Omega': Omega} # set parameters
	  args['duty'] = d
	  args['Omega'] = Omega

	  time = T * 501
	  # Calculate Floquet modes and energies at t = 0
	  f_modes_0, f_energies = floquet_modes(H, T, args)

	  f_coeff = floquet_state_decomposition(f_modes_0, f_energies, psi0)
	  psi_t = floquet_wavefunction(f_modes_0, f_energies, f_coeff, time)
	  density[i]= np.abs(f_modes_0[find_GS_index(psi0, f_energies, f_modes_0)].overlap(psi_t))**2
	  i+=1

	return density

# plot a set of parameters
def plotter(x,y, xerr=None, yerr=None, xlim=None, ylim=None, title=None, xlabel=None, ylabel=None, save=False, fileformat='pdf', fontsize=None, xscale='linear', yscale='linear', color=None, label=None):
	print('Plotting...')
	#set plot parameters and fonts
	try:
		matplotlib.rcParams['text.usetex'] = True
	except:
		print('Encountered problem with matplotlib.rcParams[text.usetex]')
		print('Continue without fancy TeX font.')
		pass


	plt.rcParams["font.family"]='serif'

	fig, ax = plt.subplots(figsize=(10,8))
	plt.tick_params(axis='both', which='major', labelsize=fontsize, direction='in', bottom=True, top=True, left=True, right=True, length=5)
	plt.errorbar(x, y, xerr=xerr, yerr=yerr,fmt='.', capsize=3, color=color, picker=10, label=label)
	plt.xlim(xlim)
	plt.ylim(ylim)
	plt.xscale(xscale)
	plt.yscale(yscale)
	plt.title(xstr(title), fontsize=fontsize)
	plt.xlabel(xstr(xlabel), fontsize=fontsize)
	plt.ylabel(xstr(ylabel), fontsize=fontsize)
	if save:
		destination = filename+'_plot_errorbar.pdf'
		plt.savefig(destination, format=fileformat)
		print('Plot saved in ', destination, '.')
	else:
		pass
	# plt.show()

# calculates the number of points in a given y interval [lower, upper]
def num_points_in_limits(val, lower, upper):
	numberOfPoints = (val>lower) & (val<upper)
	return np.sum(numberOfPoints)

# fancy fidelity plot with run number on x axis and atom number on y axis - histogram on additional subplot
def scatter_hist(x,y, xerr=None, yerr=None, xlim=None, ylim=None, title=None, xlabel=None, ylabel=None, fontsize = None, save=False, filename=None,fileformat='pdf', histlabel=None):
	print('Plotting...')
	#set plot parameters and fonts
	try:
		matplotlib.rcParams['text.usetex'] = True
	except:
		print('Encountered problem with matplotlib.rcParams[text.usetex]')
		print('Continue without fancy TeX font.')
		pass


	plt.rcParams["font.family"]='serif'
	fig = plt.figure(figsize=(11,7))

	left,width = 0.13,0.6
	bottom,height = 0.12,0.85
	spacing = 0.04

	rect_scatter = [left,bottom,width,height]
	rect_histy = [left+width+spacing, bottom, 0.2,height]
	ax = fig.add_axes(rect_scatter)
	ax_hist = fig.add_axes(rect_histy,sharey=ax)

	ax_hist.tick_params(axis='both', labelleft=False, labelsize = fontsize)
	ax.tick_params(axis='both', which='major', labelsize=fontsize)
	ax.plot(x,y, '.',color='darkblue')
	ax.locator_params(nbins=5)

	bins = 60#'auto'
	# xymax = max(np.max(np.abs(x)), np.max(np.abs(y)))
	# lim = (int(xymax/binwidth)+1)*binwidth
	# bins = np.arange(-lim,lim+binwidth, binwidth)
	ax_hist.hist(y, bins=bins, orientation='horizontal', histtype = 'stepfilled')
	ax.set_xlim(xlim)
	ax.set_ylim(ylim)
	ax_hist.set_ylim(ax.get_ylim())
	ax_hist.set_xlim(-0.1,50)

	ax.fill_between(np.array([0,len(runs)]), np.ones(np.size(xlimit))*lower, np.ones(np.size(xlimit))*upper, alpha=0.6, color = 'lightblue')
	ax_hist.fill_between(np.array([0,len(runs)]), np.ones(np.size(xlimit))*lower, np.ones(np.size(xlimit))*upper, alpha=0.6, color = 'lightblue')

	ax.set_xlabel(xlabel, fontsize = fontsize)
	ax.set_ylabel(ylabel, fontsize = fontsize)
	ax_hist.set_xlabel(histlabel,fontsize=fontsize)

	if save:
		destination = filename+'_plot_scatter_hist.pdf'
		plt.savefig(destination, format=fileformat)
		print('Plot saved in ', destination, '.')
	else:
		pass
	plt.show()

# single parameter sort
# returns x and y values for a given filter set by filter_for string and filter_for_val value
def filtered_data_set(data, arguments,x,y, filter_for=None, filter_for_val=None):
	# extract data subset based on string
	x_data = data[keyword_index(arguments, x)] 
	y_data = data[keyword_index(arguments, y)]

	# filter for condition
	filter_ind = keyword_index(arguments,filter_for)
	cond = data[filter_ind]==filter_for_val#returns an array of boolean values where the consition is met

	x_values = np.extract(cond, x_data) # extracts data where condition is met
	y_values = np.extract(cond, y_data) # extracts data where condition is met

	return x_values, y_values

def fidelity_set(data, arguments,y='Intensity_Sum', filter_for=None, filter_for_val=None, filter_for2=None, filter_for_val2=None):
	y_data = data[keyword_index(arguments, y)]

	filter_ind = keyword_index(arguments,filter_for)
	filter_ind2 = keyword_index(arguments,filter_for2)
	cond = (data[filter_ind]==filter_for_val) & (data[filter_ind2]==filter_for_val2)#returns an array of boolean values where the consition is met

	y_values = np.extract(cond, y_data) # extracts data where condition is met

	return y_values


# calculates mean, standard deviation and number of data points considered for mean
def calc_mean_std(x, y):
	print('Calculating mean and standard deviation of given set.')
	parameters = np.unique(x)
	number_of_values = np.zeros(np.size(parameters))
	mean = np.zeros(np.size(parameters))
	std = np.zeros(np.size(parameters))
	i = 0
	for param in parameters:
		matches = y[np.where(x == param)] # returns an array of values for each unique x (parameter)
		mean[i] = np.mean(matches)
		std[i] = np.std(matches)
		number_of_values[i] = np.size(matches) # counts for one mean value
		i+=1
	return mean, std, number_of_values

# calculates fidelity and associated error
def fidelity(intensity, lower, upper):
	print('Calculating fidelity...')
	fidelity = num_points_in_limits(intensity, lower,upper)/np.size(intensity)
	fidelity_error = np.sqrt((1-fidelity)*np.size(intensity))/np.size(intensity)
	print('Fidelity: ',fidelity, ' +- ', fidelity_error)
	return fidelity, fidelity_error

def logistic_fit(x, k, upper_bound, lower_bound,b):
	return upper_bound / (1 + np.exp(-k*upper_bound*x+b)*((upper_bound/lower_bound)-1))

def overlap_fit(x,a,b):
	return (4*0.938*np.sqrt(a*x+b))/(1+(a*x+b))


# merges two data sets by appending the latter to the former
def marriage(x,y):
	return np.append(x,y)

def discretizer(array):
	for i in range(len(array)):
		if 1.5<array[i]<2.5:
			array[i] = 2
		elif 0.5<array[i]<1.5:
			array[i] = 1
		elif 2.5<array[i]<3:
			array[i] = 3
		else:
			array[i] = 0

	return array


# ==========================
# READ DATA
# ==========================

data, arguments = data_handler()
data2, arguments2 = data_handler()

# ==========================
# SINGLE ATOM INTENSITY
# ==========================

single_atom_int = 135000


# ==========================
# FREQUENCY
# ==========================
duty = 50

freq, intensity = filtered_data_set(data, arguments, 'OTAPfreq','Intensity_Sum', filter_for='OTAPDcycle', filter_for_val=duty)
freq2, intensity2 = filtered_data_set(data2, arguments2, 'OTAPfreq','Intensity_Sum', filter_for='OTAPDcycle', filter_for_val=duty)

# txt file contains each run twice
freq = freq[::2]
intensity = intensity[::2]



if duty ==50:
	freq = marriage(np.roll(freq,1), np.roll(freq2, -1))
else:
	freq = marriage(freq, freq2)

intensity = marriage(intensity, intensity2)
# normalise and and discretise 
norm_int = intensity/single_atom_int
atomnumber = discretizer(norm_int)

mean, std, num = calc_mean_std(freq, atomnumber)
frequencies = np.unique(freq)/1e3/29.5
# frequencies = np.roll(frequencies, -1)
atomnumber = mean
datomnumber = std / np.sqrt(num)


# norm_int2 = intensity2 / single_atom_int
# atomnumber2 = discretizer(norm_int2)
# mean2, std2, num2 = calc_mean_std(freq2, atomnumber2)
# frequencies2 = np.unique(freq2)/1e3/29.5

# atomnumber2 = mean2
# datomnumber2 = std2 / np.sqrt(num2)


plotter(frequencies, atomnumber, xlabel='switching frequency $\Omega / \omega$ ',yerr=datomnumber, ylabel='mean number of atoms', save=False, fontsize=25, label='measured data') #, xlim=xlimit

# freq_space = np.linspace(min(frequencies), max(frequencies), 512)
# plt.plot(freq_space, overlap_fit(freq_space, 0.2,-0.8), color='red')


# # ALL RUNS
# runs = np.linspace(0,len(intensity), len(intensity))
# plotter(runs, norm_int, xlabel='run number', ylabel='normalised intensity sum', fontsize=25, label='all measured data')


# def overlap_fit(x,a,b):
#   return (2*np.sqrt(a*x+b))/(1+(a*x+b))

# from scipy.optimize import curve_fit

# popt, pcov = curve_fit(overlap_fit, Omegas/w, density)

# density = calc_overlap_Floquet(29.5)


# plt.plot(frequencies, density*2, '^', label='numerical result')
# plt.legend(loc='best', fontsize=25)



# ax2=plt.twinx()
# ax2.tick_params(axis='both', which='major', labelsize=25)
# ax2.plot(Omegas/w, density*2/0.938, '.', color='orange')
# ax2.set_ylim(1.9,2)





# duty = 20

# freq, intensity = filtered_data_set(data, arguments, 'OTAPfreq','Intensity_Sum', filter_for='OTAPDcycle', filter_for_val=duty)
# freq2, intensity2 = filtered_data_set(data2, arguments2, 'OTAPfreq','Intensity_Sum', filter_for='OTAPDcycle', filter_for_val=duty)

# plt.errorbar(frequencies, atomnumber, yerr=datomnumber,fmt='.', capsize=3,color='red')

# if duty ==50:
# 	index = 30
# 	plt.errorbar(frequencies[index], atomnumber[index], yerr=datomnumber[index], fmt='.', capsize=3,color='#cccccc')
# 	index = 25
# 	plt.errorbar(frequencies[index], atomnumber[index], yerr=datomnumber[index], fmt='.', capsize=3,color='#cccccc')
# else:
# 	pass

# plt.savefig('all_runs.pdf', format='pdf')
# popt,pcov = curve_fit(logistic_fit, frequencies, atomnumber, sigma = datomnumber, p0=[0.7, 2, 0])



# ==========================
# FIDELITY
# # ==========================
# duty = 20
# freq = 250000

# intensity = fidelity_set(data, arguments, filter_for='OTAPDcycle', filter_for_val=duty, filter_for2='OTAPfreq', filter_for_val2=np.roll(freq,1))[::2]
# intensity2 = fidelity_set(data2, arguments2, filter_for='OTAPDcycle', filter_for_val=duty, filter_for2='OTAPfreq', filter_for_val2=np.roll(freq,-1))
# intensity = marriage(intensity, intensity2)

# norm_int = intensity / single_atom_int
# runs = np.arange(0,np.size(norm_int),1)
# lower = 1.7
# upper = 2.3
# xlimit = [0,len(runs)]
# ylimit = [-0.5, 3]

# fidelity(norm_int, lower, upper)
# scatter_hist(runs, norm_int, xlim=xlimit,ylim=ylimit, xlabel='run number', ylabel='normalised intensity sum',histlabel='counts', fontsize=25,filename='dc20_fidelity', save=True)

plt.show()

print('Done')