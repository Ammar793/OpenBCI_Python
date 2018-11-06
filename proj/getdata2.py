import numpy as np


class GetData2():
	
	def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
		
		return y


	def get_states(self,file):
		
		fid = open(file, 'r')
		lines = fid.readlines()
		nof = len(lines)


		channels = np.zeros((4,nof))
		time_eeg = []
		for x in range(0,nof):			
			for y in range(0,4):	
				channels[y,x] = float(lines[x].split(',')[y+1])				
				
		
		returnar=np.zeros((nof,4))
		returnar2=np.zeros((nof,4))
		
		for i in range(0,nof):
			for y in range(0,4):
				returnar[i] = channels[:,i]
		
		
		
			
			 
		
			
		
		
		#print (returnar)
		
		
		
		return returnar