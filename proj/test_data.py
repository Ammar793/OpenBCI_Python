import argparse
import random
import time
import numpy as np
import pprint
import pyeeg
import scipy
from scipy import signal
import matplotlib.pyplot as plt
import pickle
import sklearn
from sklearn import datasets,svm
from sklearn.externals import joblib
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from getdata import GetData
from getdata2 import GetData2


from OSC import OSCClient, OSCMessage
lastnum=0

file = 'ammar-conc4.txt' 
file23 = 'ammar-dist4.txt' 

	
def ml(hursts,bt,pfds,hfd,targets,nof):

	data = np.zeros((nof,16))	
	
	
	for i in range (0,int(nof)):
		for y in range (0,4):
		
			data[i,y] = hursts[y,i]
			data[i,y+4] = bt[y,i]
			data[i,y+8]=pfds[y,i]
			data[i,y+12]=hfd[y,i]
		
		

	#clf = svm.SVC(kernel='linear', C=1, class_weight={2:1.5}) #support v
	#clf_lda = LinearDiscriminantAnalysis()
	clf = joblib.load('classifier.pkl') 
	clf_lda = joblib.load('classifier_lda.pkl')
	
	targets2=np.zeros((len(targets)))
	data2=np.zeros((len(data)))	
	
	for i in range(0,len(data)):
		targets2[i] = targets[i]
	
	
	
	for i in range (0,int(nof)):
		if(np.all(np.isfinite(data[i]))==False):
			for y in range (0,len(data[i])):
				if(np.isnan(data[i,y])):
					data[i,y] = data[i,y+1]
	
	
	
	pred = np.zeros((len(data)))
	pred_lda = np.zeros((len(data)))
	
	
	for i in range (0,len(data)):		
		a=clf.predict(data[i].reshape(1,-1))
		b=clf_lda.predict(data[i].reshape(1,-1))
		
		pred[i]=a[0]
		pred_lda[i]=b[0]
		
		
		
	counter1=0
	counter2=0
	counter3=0
	counter4=0
	counter5=0
	counter6=0
	
	
	for i in range (0,len(data)):
		if (pred[i] == targets2[i] ):
			counter1+=1
		else:
			counter2+=1
		
		if (pred_lda[i] == targets2[i] ):
			counter3+=1
		else:
			counter4+=1
		
		if (pred_lda[i] == pred[i]):
			if(pred_lda[i]==targets2[i]):
				counter5+=1
			else:
				counter6+=1		
			
	print(counter1)
	print(counter2)	
	perc = float(counter1)/float((counter1+counter2))
	print(perc*100)
	print("\n\n")
	print(counter3)
	print(counter4)
	perc = float(counter3)/float((counter3+counter4))
	print(perc*100)		

	print("\n\n")
	print(counter5)
	print(counter6)
	perc = float(counter5)/float((counter5+counter6))
	print(perc*100)		

	
	
	
def get_state_features(state):
	
	nof = len(state)
	po = 600
	
	pfds = np.zeros((4,int(nof/po)))
	ap_entropy = np.zeros((4,int(nof/po)))
	hursts = np.zeros((4,int(nof/po)))
	hfd = np.zeros((4,int(nof/po)))
	bins = np.zeros(((int(nof/po),4,2,5)))
	
	lastnum=0

	for i in range (0,(int(nof/po))):
		channels = np.zeros((4,po))		
		
		for x in range(0,po):			
			for y in range(0,4):		
				
				channels[y,x] = float(state[lastnum+x,y])
		

		for x in range (0,4):
			channels[x] = scipy.signal.savgol_filter(channels[x], 11, 3, deriv=0, delta=1.0, axis=-1, mode='interp', cval=0.0)		
		
		#alpha=[]
		if((nof-lastnum)!=0):
			for x in range(0,4):
				hursts[x,i] = pyeeg.hurst(channels[x])
				pfds[x,i] = pyeeg.pfd(channels[x])
				#ap_entropy[x,i] = pyeeg.ap_entropy(X, M, R)
				hfd[x,i] = pyeeg.hfd(channels[x],15)
				bins[i,x] = pyeeg.bin_power(channels[x], [0.5,4,7,12,15,18], 200)				
				k=1
		lastnum=lastnum+po
	
	return pfds,hursts,bins,hfd
	
	

if __name__ == "__main__":
	global nof
	global po
	client = OSCClient()
	client.connect( ("127.0.0.1", 8000) )	

	forever=True
	
	data_getter = GetData()
	data_getter2 = GetData2()
	

	#concentrated,resting,blinking,normal_pics,trippy_pics = data_getter.get_states(file,file2)
	concentrated= data_getter2.get_states(file)
	resting= data_getter2.get_states(file23)	
	
	pfds_c,hursts_c,bins_c,hfd_c=get_state_features(concentrated)
	pfds_r,hursts_r,bins_r,hfd_r=get_state_features(resting)
	
	delta_c= np.zeros((4,len(bins_c)))
	beta_c= np.zeros((4,len(bins_c)))
	alpha_c= np.zeros((4,len(bins_c)))
	theta_c= np.zeros((4,len(bins_c)))
	bt_c = np.zeros((4,len(bins_c)))
	
	delta_r= np.zeros((4,len(bins_r)))
	beta_r= np.zeros((4,len(bins_r)))
	alpha_r= np.zeros((4,len(bins_r)))
	theta_r= np.zeros((4,len(bins_r)))
	bt_r = np.zeros((4,len(bins_r)))
	
	
	for i in range (0,len(bins_c)):
		for y in range (0,4):
			delta_c[y,i] = bins_c[i,y,0,0]
			theta_c[y,i] = bins_c[i,y,0,1]
			alpha_c[y,i] = bins_c[i,y,0,2]
			beta_c[y,i] = bins_c[i,y,0,4]
			bt_c[y,i] = theta_c[y,i]/beta_c[y,i]
	
	for i in range (0,len(bins_r)):
		for y in range (0,4):
			delta_r[y,i] = bins_r[i,y,0,0]
			theta_r[y,i] = bins_r[i,y,0,1]
			alpha_r[y,i] = bins_r[i,y,0,2]
			beta_r[y,i] = bins_r[i,y,0,4]
			bt_r[y,i] = theta_r[y,i]/beta_r[y,i]
	
	
	all_hursts = np.zeros((4,len(hursts_c[0])+len(hursts_r[0])))	
	all_bt = np.zeros((4,len(hursts_c[0])+len(hursts_r[0])))
	all_pfds = np.zeros((4,len(hursts_c[0])+len(hursts_r[0])))	
	all_hfd = np.zeros((4,len(hursts_c[0])+len(hursts_r[0])))
	all_targets = np.zeros((len(hursts_c[0])+len(hursts_r[0])))
	
	
	
	print(len(hursts_c[0]))
	print(len(hursts_r[0]))
	for i in range (0,len(hursts_c[0])):
		for y in range (0,4):
			all_hursts[y,i] = hursts_c[y,i]
			all_bt[y,i] = bt_c[y,i]
			all_pfds[y,i] = pfds_c[y,i]
			all_hfd[y,i] = hfd_c[y,i]
			
		all_targets[i] = 1
			
	for i in range (len(hursts_c[0]),len(hursts_c[0])+len(hursts_r[0])):
		for y in range (0,4):
			all_hursts[y,i] = hursts_r[y,i%len(hursts_c[0])]
			all_bt[y,i] = bt_r[y,i%len(hursts_c[0])]
			all_pfds[y,i] = pfds_r[y,i%len(hursts_c[0])]
			all_hfd[y,i] = hfd_r[y,i%len(hursts_c[0])]
			
		all_targets[i] = 2	
		
		
#	for i in range (0,4):
#		all_hursts[i]= scipy.signal.savgol_filter(all_hursts[i], 21, 3, deriv=0, delta=1.0, axis=-1, mode='interp', cval=0.0)
#		all_bt[i]= scipy.signal.savgol_filter(all_bt[i], 21, 3, deriv=0, delta=1.0, axis=-1, mode='interp', cval=0.0)
#		all_pfds[i]= scipy.signal.savgol_filter(all_pfds[i], 21, 3, deriv=0, delta=1.0, axis=-1, mode='interp', cval=0.0)
#		all_hfd[i]= scipy.signal.savgol_filter(all_hfd[i], 21, 3, deriv=0, delta=1.0, axis=-1, mode='interp', cval=0.0)
	
	
		
	ml(all_hursts,all_bt,all_pfds,all_hfd,all_targets,len(all_hursts[0]))	
	
	