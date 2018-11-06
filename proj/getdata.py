import numpy as np

class GetData():
	def get_states(self,file,file2):
		
		fid = open(file, 'r')
		lines = fid.readlines()
		nof = len(lines)-6



		
		fid2 = open(file2, 'r')
		lines2 = fid2.readlines()
		nof2 = len(lines2)-6

			
		times = lines2[1].split(',')

		for i in range(0,len(times)):
			times[i] = times[i][:-2]
			cutie = times[i].split(':')
			if(len(cutie)==4):
				times[i] = cutie[0]+':'+cutie[1]+':'+cutie[2]+'.'+cutie[3]
				times[i].strip()
			else:
				times[i]=-1
				
			

		

		channels = np.zeros((4,nof))
		time_eeg = []
		for x in range(0,nof):
			time_eeg.append(lines[x+6].split(',')[8])
			time_eeg[x]=time_eeg[x][:-3].strip()
			for y in range(0,4):	
				channels[y,x] = float(lines[x+6].split(',')[y+1])
		mtl = []
		counter=0
		for x in range(0,len(times)):
			if(times[x]!=-1):
				found=False
				while(found==False):
					for i in range(0,nof):
						if(times[x]==time_eeg[i]):
							mtl.append(i)
							counter+=1
							found = True
							break

		mtl.sort()
		
		concentrated = []
		resting=[]
		blinking =[]
		normal_pics =[]
		trippy_pics =[]

		for i in range(mtl[1],mtl[2]):
			concentrated.append(channels[:,i])
			
		for i in range(mtl[4],mtl[5]):
			concentrated.append(channels[:,i])
			
		for i in range(mtl[2],mtl[3]):
			resting.append(channels[:,i])

		for i in range(mtl[5],mtl[6]):
			resting.append(channels[:,i])
			
		for i in range(mtl[51],mtl[52]):
			resting.append(channels[:,i])

		for i in range(mtl[49],mtl[50]):
			blinking.append(channels[:,i])

		for i in range(mtl[9],mtl[13]):
			normal_pics.append(channels[:,i])

		for i in range(mtl[21],mtl[27]):
			normal_pics.append(channels[:,i])

		for i in range(mtl[42],mtl[48]):
			normal_pics.append(channels[:,i])
			
		for i in range(mtl[14],mtl[20]):
			trippy_pics.append(channels[:,i])
			
		for i in range(mtl[28],mtl[41]):
			trippy_pics.append(channels[:,i])		
		print(concentrated)
		time.sleep(100)
		
		return concentrated,resting,blinking,normal_pics,trippy_pics