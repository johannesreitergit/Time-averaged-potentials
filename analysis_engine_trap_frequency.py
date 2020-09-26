# import required modules
import numpy as np
from tkfilebrowser import askopenfilename  # GUI file browser
import matplotlib.pyplot as plt
import matplotlib
from scipy.optimize import curve_fit


filename = askopenfilename()
delimiter = '	'					# must match delimiter used in file

#____________________________________________________________________________________________________________________________________________


data = np.genfromtxt(filename, comments = '#', delimiter=delimiter, unpack = True, encoding=None, names=True, dtype=None) # read data
arguments = np.asarray(data.dtype.names, dtype=str) # read header
data = np.genfromtxt(filename, comments = '#', delimiter=delimiter, unpack = True) # read data




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

def unique_parameter_set(data, arguments, keyword, return_index=False):
	parameters = np.unique(data[keyword_index(arguments,keyword)], return_index=return_index)
	return parameters

# calculates the mean and standard deviation of a given quantity while specifing a keyword
# keyword is usually x axis, quanity y axis
def calc_mean_std(data, arguments, keyword, quanity):
	sorted_data = parameter_sort(data, arguments, keyword) # sort data for keyword 
	parameters = unique_parameter_set(data, arguments, keyword) # find unique data for keywords

	#keyword values
	keyword_ind = keyword_index(arguments, keyword)
	keyword_values = sorted_data[keyword_ind]

	# quantity values
	quanity_index = keyword_index(arguments, quanity)
	quantity_values = sorted_data[quanity_index]

	# calculate mean and std
	mean = np.zeros(np.size(parameters))
	std = np.zeros(np.size(parameters))
	i = 0
	for param in parameters:
		mean[i] = np.mean(quantity_values[np.where(keyword_values == param)])
		std[i] = np.std(quantity_values[np.where(keyword_values == param)])
		i+=1

	return mean, std

# return empty string in case label is None
def xstr(s):
	if s is None:
		return ''
	return str(s)	

# plot a set of parameters
# def plotter(x,y, xerr=None, yerr=None, xlim=None, ylim=None, title=None, xlabel=None, ylabel=None, save=False, fileformat='pdf', fontsize=None):
# 	#set plot parameters and fonts
# 	try:
# 		matplotlib.rcParams['text.usetex'] = True
# 	except:
# 		print('Encountered problem with matplotlib.rcParams[text.usetex]')
# 		print('Continue without fancy TeX font.')
# 		pass


# 	plt.rcParams["font.family"]='serif'

# 	fig, ax = plt.subplots(figsize=(10,8))
# 	plt.tick_params(axis='both', which='major', labelsize=fontsize, direction='in', bottom=True, top=True, left=True, right=True, length=5)
# 	plt.errorbar(x, y, xerr=xerr, yerr=yerr,fmt='.', capsize=3)
# 	plt.xlim(xlim)
# 	# plt.xscale('log')
# 	plt.ylim(ylim)
# 	plt.title(xstr(title), fontsize=20)
# 	plt.xlabel(xstr(xlabel), fontsize=25)
# 	plt.ylabel(xstr(ylabel), fontsize=25)
# 	if save:
# 		plt.savefig(filename+'_plot_errorbar.pdf', format=fileformat)
# 	else:
# 		pass

def plotter(x,y, xerr=None, yerr=None, xlim=None, ylim=None, title=None, xlabel=None, ylabel=None, save=False, fileformat='pdf', fontsize=None, xscale='linear', yscale='linear', color=None):
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
	plt.errorbar(x, y, xerr=xerr, yerr=yerr,fmt='.', capsize=3, color=color)
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

def fidelity(intensity, lower, upper):
	fidelity = num_points_in_limits(intensity, lower,upper)/np.size(intensity)
	print('Fidelity: ',fidelity)
	return fidelity


def scatter_hist(x,y, xerr=None, yerr=None, xlim=None, ylim=None, title=None, xlabel=None, ylabel=None, fontsize = None, save=False, fileformat='pdf', histlabel=None):
		#set plot parameters and fonts
	try:
		matplotlib.rcParams['text.usetex'] = True
	except:
		print('Encountered problem with matplotlib.rcParams[text.usetex]')
		print('Continue without fancy TeX font.')
		pass


	plt.rcParams["font.family"]='serif'
	fig = plt.figure(figsize=(10,8))

	left,width = 0.1,0.65
	bottom,height = 0.1,0.8
	spacing = 0.02

	rect_scatter = [left,bottom,width,height]
	rect_histy = [left+width+spacing, bottom, 0.2,height]
	ax = fig.add_axes(rect_scatter)
	ax_hist = fig.add_axes(rect_histy,sharey=ax)

	ax_hist.tick_params(axis='both', labelleft=False, labelsize = fontsize)
	ax.tick_params(axis='both', which='major', labelsize=fontsize)
	ax.plot(x,y, '.',color='darkblue')

	bins = 'auto'
	# xymax = max(np.max(np.abs(x)), np.max(np.abs(y)))
	# lim = (int(xymax/binwidth)+1)*binwidth
	# bins = np.arange(-lim,lim+binwidth, binwidth)
	ax_hist.hist(y, bins=bins, orientation='horizontal', histtype = 'stepfilled')
	ax.set_xlim(xlim)
	ax.set_ylim(ylim)
	ax_hist.set_ylim(ax.get_ylim())
	ax_hist.set_xlim(-0.1,25)

	ax.fill_between(np.array([0,len(runs)]), np.ones(np.size(xlimit))*1.5, np.ones(np.size(xlimit))*2.5, alpha=0.6, color = 'lightblue')
	ax_hist.fill_between(np.array([0,len(runs)]), np.ones(np.size(xlimit))*1.5, np.ones(np.size(xlimit))*2.5, alpha=0.6, color = 'lightblue')

	ax.set_xlabel(xlabel, fontsize = fontsize)
	ax.set_ylabel(ylabel, fontsize = fontsize)
	ax_hist.set_xlabel(histlabel,fontsize=fontsize)

	if save:
		plt.savefig(filename+'_plot_scatter_hist.pdf', format=fileformat)
	else:
		pass


# calculate values

atomnumber = calc_mean_std(data, arguments, 'OTAPmodfreq', 'Intensity_Sum')[0] / 9400000
datomnumber = calc_mean_std(data, arguments, 'OTAPmodfreq', 'Intensity_Sum')[1] / 9400000#/np.sqrt(200)
freq = unique_parameter_set(data, arguments, 'OTAPmodfreq')/1e3
freq_fine = np.linspace(8,130,200)

lower = 1.5
upper = 2.5

xlimit = [7,130]


#plot
plotter(freq, atomnumber, xlabel='modulation frequency [kHz]',yerr=datomnumber, ylabel='normalised intensity sum', xlim=xlimit, save=False, fontsize=25)


def gauss_fit(x, mu, sigma, amplitude, offset):
	return amplitude*(1/np.sqrt(2*np.pi*sigma**2))*np.exp(-(x-mu)**2/(2*sigma**2))  + offset


popt, pcov = curve_fit(gauss_fit, freq, atomnumber, sigma=datomnumber, p0=[55,2,-80,65])

plt.plot(freq_fine, gauss_fit(freq_fine, *popt), color='red', label='Gaussian fit')
plt.legend(fontsize=25)
# scatter_hist(runs, nAtoms, xlim=xlimit, xlabel='Run number', ylabel='\# atoms',histlabel='\# atoms',	 fontsize=20, save=False)

plt.savefig('trap_frequency.pdf', format='pdf')

print('trap frequency', popt[0], '+-', np.sqrt(pcov[0][0]))
# plt.savefig(filename+'_plot_errorbar.pdf', format='pdf')
plt.show()