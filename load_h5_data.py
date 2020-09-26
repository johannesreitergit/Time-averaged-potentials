# load h5 files
import h5py
import numpy as np
import time
from tkfilebrowser import askopenfilename  # GUI file browser

#______________________________________________________________________________________________________________________________________________
# Tk().withdraw() # omits empty window

filepath = askopenfilename() 	#'data.h5'

tic = time.time()
f = h5py.File(filepath, 'r')  	# creates a file object f and opens it in read mode

all_runs = f['All Runs']       	# extract all runs struct
cam = f['GuppyFl']    			# extract camera struct for Guppy fluorescnece
# cam = f['Andor']    
# cam = f['ADC1']    

datasetname = 'image'
dataset_ending = '.dat'
delimiter = '	'

#______________________________________________________________________________________________________________________________________________

def extraction(save_attribute_file=True):
	# number of runs
	NumberOfRuns = len(all_runs.keys())
	print('This file contains %s runs.'%NumberOfRuns)

	all_images_ini = np.zeros(NumberOfRuns)
	all_images = np.ascontiguousarray(all_images_ini, dtype=np.ndarray) # array must be C contiguous and writeable in order to be handled properly by h5py / cython

	attributes = np.zeros(NumberOfRuns, dtype = np.ndarray) #attributes contain meta information and run specific parameters and values such as intensity, atom number, fit parameters etc.

	# Well-extracted coffee leaves no Kaffeesatz
	# Let the extraction process begin...
	i=0
	# the elements 'run' in all_runs are labelled string-like e.g. 'Run 4234'
	for run in all_runs:
		runNumber = int(run[4:len(run)]) 											# extract run number as int e.g. '4234'
		dataset = all_runs[str(run)][datasetname+str(runNumber)+dataset_ending] 	# extract dataset as h5 object
		# image_shape = dataset.shape                                                 # determine the shape of the images
		# image = np.ascontiguousarray(np.zeros(image_shape))							# initialize new image as C contiguous array					
		# dataset.read_direct(image)												    # directly write to numpy array avoiding intermediate copy
		# all_images[i] = image 													    # image as i th element in all_images array
		attributes[i] = np.asarray(list(dataset.attrs.items()))                  # string-like attribute items as i th element in attributes
		i += 1

	header = delimiter.join(list(dataset.attrs.keys()))						# header containing the name of the attributes
	attribute_data = np.zeros(NumberOfRuns, dtype=np.ndarray)			# attribute data

	for run_number in np.arange(0,NumberOfRuns, 1):
		attribute_data[run_number] = attributes[run_number][:,1] # extract first column corresponding to the attribute values

	attribute_data = np.concatenate(attribute_data).reshape((NumberOfRuns, len(attributes[run_number])) ) # convert multi-array into single array

	attr_filename = '%s_attributes.txt' %filepath 															# name for the file where all the attributes should stored in an array like format
	if save_attribute_file ==True:
		np.savetxt(attr_filename, attribute_data, fmt='%s', newline='\n', delimiter = delimiter, header=header, comments='#') 		# storing attributes
		print('Attribute file written and saved in %s' %attr_filename)
	
	print('Data extraction process finished!')
	print()

	return all_images, attributes, NumberOfRuns

# extract data 
all_images, attributes, NumberOfRuns = extraction(save_attribute_file=True)


#stop timer and terminate
toc = time.time()
exc_time = toc - tic
print('Excecution time: ', exc_time, 'seconds')
print('Done')
