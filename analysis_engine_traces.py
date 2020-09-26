# import required modules
import numpy as np
from tkfilebrowser import askopenfilename  # GUI file browser
import matplotlib.pyplot as plt
import matplotlib


ch2_filename = 'C1--dc20freq200kHz--00001.txt'
ch1_filename = 'C2--dc20freq200kHz--00001.txt'
delimiter = ';'					# must match delimiter used in file

#____________________________________________________________________________________________________________________________________________

ch1 = np.loadtxt(ch1_filename, unpack = True, delimiter=delimiter) # read data
ch2 = np.loadtxt(ch2_filename, unpack = True, delimiter=delimiter) # read data

timeshift_1 = 0 +1.2
timeshift_2 = 1 +1.2

# calculate value
time_ch1 = ch1[0] *	 1e6 + timeshift_1
PD = ch1[1] *0.77 *1/0.93

# time_ch2 = ch2[0][::5] * 1e6 + timeshift_2
# trigger = ch2[1][::5] / 5 

fontsize=25

#set plot parameters and fonts
matplotlib.rcParams['text.usetex'] = True
plt.rcParams["font.family"]='serif'

fig = plt.figure(figsize=(10,8))
plt.tick_params(axis='both', which='major', labelsize=fontsize,direction='in', bottom=True, top=True, left=True, right=True, length=5)
plt.plot(time_ch1, PD, label = 'Photodiode signal')
plt.vlines([5.68, 5.92], -1, 2, color='grey', linestyle='dashed')
# plt.plot(time_ch2, trigger, label='Switch trigger', alpha=1, color='red')
plt.xlim(5.6,6.0)
plt.ylim(-0.1, 1.3)
plt.xlabel('time [$\mu s$]', fontsize=fontsize)
plt.ylabel('normalised amplitude', fontsize=fontsize)
plt.legend(fontsize=fontsize)

plt.show()