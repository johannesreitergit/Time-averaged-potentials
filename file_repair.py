# script to repair broken file 
# joins two lines each
from tkfilebrowser import askopenfilename  # GUI file browser

filename = askopenfilename()

data = open(filename).read().split('\n')
for i,line in enumerate(data):
	if line.startswith('	'):
		data[i-1]=data[i-1]+line
		data.pop(i)

output = open('output2.txt', 'a')

output.write('\n'.join(data))

		
