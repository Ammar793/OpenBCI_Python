import argparse
import random
import time
import numpy as np
import pprint
import pyeeg
import scipy
import matplotlib.pyplot as plt
import pickle
import sklearn
from sklearn import datasets,svm
from sklearn.externals import joblib
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis


#from pythonosc import osc_message_builder
#from pythonosc import udp_client

from OSC import OSCClient, OSCMessage
lastnum=0

def getdata(var):
	global lastnum	
	file = 'C:\\Users\\Ammar Raufi\\Desktop\\openbci\\software\\application.windows64\\SavedData\\OpenBCI-RAW-'+var+'.txt' 
	fid = open(file, 'r')
	
	lines = fid.readlines()

	numberOfFrames = len(lines)-6
	print(numberOfFrames-lastnum)	
	channels = np.zeros((4,numberOfFrames-lastnum))
	
	
	for x in range(0,numberOfFrames-lastnum-6):		
		for y in range(0,4):			
			channels[y,x] = float(lines[lastnum+x+6].split(',')[y+1])
	
	
	#print (lastnum)	
	lastnum=numberOfFrames
	return channels
	
	
	
def ml(hursts,bt,pfds,hfd):

	data = np.zeros((16))	

	
	print(hursts)
	for y in range (0,4):
	
		data[y] = hursts[y]
		data[y+4] = bt[y]
		data[y+8]= pfds[y]
		data[y+12]= hfd[y]
		
		
		
	#print(data)
	#clf = svm.SVC(kernel='linear', C=1) #support v
	#clf_lda = LinearDiscriminantAnalysis()
	clf = joblib.load('classifier.pkl') 
	clf_lda = joblib.load('classifier_lda.pkl') 
	
	
	
	#clf.fit(data, targets2)
	#clf_lda.fit(data, targets2)
	
	
		#print(data[i].reshape(1,-1))
	for i in range (0,len(data)):
		if(np.all(np.isfinite(data[i]))==False):							
			data[i] = 0.4 
					
	a=clf.predict(data.reshape(1,-1))
	b=clf_lda.predict(data.reshape(1,-1))
	
	return a,b
		
		
	#joblib.dump(clf, 'classifier.pkl') 
	#joblib.dump(clf_lda, 'classifier_lda.pkl') 

	
	
	
def get_state_features(channel):
	
	nof = len(channel)	
	pfds = np.zeros((4))
	ap_entropy = np.zeros((4))
	hursts = np.zeros((4))
	hfd = np.zeros((4))
	bins = np.zeros(((4,2,5)))
	
	lastnum=0


		
		#alpha=[]
	if((nof-lastnum)!=0):
		for x in range(0,4):
			hursts[x] = pyeeg.hurst(channel[x])
			pfds[x] = pyeeg.pfd(channel[x])
			#ap_entropy[x,i] = pyeeg.ap_entropy(X, M, R)
			hfd[x] = pyeeg.hfd(channel[x],15)
			bins[x] = pyeeg.bin_power(channel[x], [0.5,4,7,12,15,18], 200)			
	
	
	
	delta= np.zeros((4))
	beta= np.zeros((4))
	alpha= np.zeros((4))
	theta= np.zeros((4))
	dfas= np.zeros((4))
	bt = np.zeros((4))
	

	for y in range (0,4):
		delta[y] = bins[y,0,0]
		theta[y] = bins[y,0,1]
		alpha[y] = bins[y,0,2]
		beta[y] = bins[y,0,4]
		bt[y] = theta[y]/beta[y]
	
	lastnum=lastnum+nof
	
	
	
	return pfds,dfas,hursts,bins,bt,hfd
	
	

if __name__ == "__main__":
	global nof
	global po
	client = OSCClient()
	client.connect( ("127.0.0.1", 8000) )	

	forever=True
	var = raw_input("Enter File name: ")
	while(forever):
		channels= getdata(var)
		#print("data got")
		pfds,dfas,hursts,bins,bt,hfd = get_state_features(channels)
		#print("features got")
		a,b = ml(hursts,bt,pfds,hfd)
		
		if(a==[1.]):
			print('concentrated')
		else:
			print('distracted')
		
		if(b==[1.]):
			print('lda concentrated')
		else:
			print('lda distracted')	
		
		
		k = OSCMessage('/vol')
		l = OSCMessage('/lda')

		
		print(int(a[0]))
		
		if (a==[1.]):
			k.append(1)
		else:
			k.append(2)
		
		client.send(k)
		k.clear()
		time.sleep(3)
		#client.send( OSCMessage("/vol", a) )
		#client.send_message("/third", ok[i])
		#client.send_message("/rev", ok[i])
			
	#print(pfds)
	#print(dfas)
	#print(hursts)
	print(bins)
	
	
	
	
	
	
    

	
    