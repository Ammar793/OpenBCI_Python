import time
import numpy as np
import pyeeg
import scipy
import matplotlib.pyplot as plt
import sklearn
from sklearn import svm
from getdata import GetData
from getdata2 import GetData2
import pickle
from sklearn.externals import joblib
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.preprocessing import label_binarize
from sklearn.multiclass import OneVsRestClassifier
from scipy import interp
#from pythonosc import osc_message_builder
#from pythonosc import udp_client

from OSC import OSCClient

file = 'mbt-conc.txt' 
file23 = 'mbt-dist.txt' 
fid = open(file, 'r')

file2 = '91t.txt'
lines = fid.readlines()
nof = len(lines)-6
print(nof)


lastnum=0
po=400

pfds = np.zeros((4,int(nof/po)))
dfas = np.zeros((4,int(nof/po)))
hursts = np.zeros((4,int(nof/po)))
bins = np.zeros(((int(nof/po),4,2,4)))


def getdata(num,):
	global lastnum
	global pfds
	global dfas
	global hursts
	global bins
	global nof
	global po
	#file = 'C:\\Users\\Ammar Raufi\\Desktop\\openbci\\software\\application.windows64\\SavedData\\OpenBCI-RAW-2017-03-18_18-46-49.txt' 

	#fid = open(file, 'r')
	
	#lines = fid.readlines()

	#numberOfFrames = len(lines)-6
	#print(numberOfFrames-lastnum)
	
	
	channels = np.zeros((4,po))
	
	#alpha = np.zeros(4)
	for x in range(0,po): #numberOfFrames-lastnum-6
		
		for y in range(0,4):
			
			channels[y,x] = float(lines[lastnum+x+6].split(',')[y+1])
	
	#alpha=[]
	if((nof-lastnum)!=0):
		for x in range(0,4):
			hursts[x,num] = pyeeg.hurst(channels[x])
			#pfds[x,num] = pyeeg.pfd(channels[x])
			#dfas[x,num] = pyeeg.dfa(channels[x])			
			bins[num,x] = pyeeg.bin_power(channels[x], [0.5,4,7,12,30], 200)	
			k=1

	print (lastnum)
		#print (alpha)
		
	lastnum=lastnum+po
	return channels[0]

	
	
def ml(hursts,bt,pfds,hfd,targets,nof):

	data = np.zeros((nof,16))	
	for i in range (0,int(nof)):
		for y in range (0,4):
		
			data[i,y] = hursts[y,i]
			data[i,y+4] = bt[y,i]
			data[i,y+8]=pfds[y,i]
			data[i,y+12]=hfd[y,i]		
		
	#print(data)
	clf = svm.SVC(kernel='linear', C=100,class_weight={2:3}) #support v
	clf_lda = LinearDiscriminantAnalysis()
	#clf = joblib.load('classifier.pkl') 	
	
	targets2=np.zeros((len(targets)))
	data2=np.zeros((len(data)))
	
	for i in range(0,len(data)):	
		targets2[i] = int(targets[i])
#	print(targets2.ravel())
#	y = label_binarize(targets2.ravel(), classes=[1, 2])
#	print(y)
#	n_classes = y.shape[1]
	
#	X_train, X_test, y_train, y_test = train_test_split(data,y.ravel(), test_size=.5)
#	y_score = clf.fit(X_train, y_train).decision_function(X_test)
#	y_score2 = clf_lda.fit(X_train, y_train).decision_function(X_test)
	
#	fpr = dict()
#	tpr = dict()
#	roc_auc = dict()

	
	#	fpr, tpr, _ = roc_curve(y_test, y_score)
#	fpr2, tpr2, _ = roc_curve(y_test, y_score2)
#	roc_auc = auc(fpr, tpr)
#	roc_auc2 = auc(fpr2, tpr2)
	
		

	#plt.figure()
	#lw = 2
	#plt.plot(fpr, tpr, color='darkorange',
#			 lw=lw, label='ROC curve SVM (area = %0.2f)' % roc_auc)
#	plt.plot(fpr2, tpr2, color='green',
#			 lw=lw, label='ROC curve LDA(area = %0.2f)' % roc_auc2)
#	plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
#	plt.xlim([0.0, 1.0])
#	plt.ylim([0.0, 1.05])
#	plt.xlabel('False Positive Rate')
#	plt.ylabel('True Positive Rate')
#	plt.title('Receiver operating characteristic')
#	plt.legend(loc="lower right")	
#	plt.show()
	
	

	
	
		
	
	targets2 = np.reshape(targets2,(len(data),1))
	#print (targets2)
	
	
	for i in range (0,int(nof)):
		if(np.all(np.isfinite(data[i]))==False):
			for y in range (0,len(data[i])):
				if(np.isnan(data[i,y])):					
					data[i,y] = 0.4 
		
		
	#parameters = {'kernel': ('linear', 'rbf'), 'C': [50,60,70,80,90,100,110,120,130,140,150,300,400]}
	#svr = svm.SVC()
	#clf8 = grid_search.GridSearchCV(svr, parameters)
	
	c, r = targets2.shape
	targets2 = targets2.reshape(c,)
	#clf8.fit(data, targets2)
	#print(clf8.best_params_)
	#time.sleep(10)
		
	clf.fit(data, targets2)
	clf_lda.fit(data, targets2)
	
#	for i in range (0,len(data)):
		#print(data[i].reshape(1,-1))
#		a=clf.predict(data[i].reshape(1,-1))
#		b=clf_lda.predict(data[i].reshape(1,-1))
		
#		if(a==[1.]):
#			print('concentrated')
#		else:
#			print('distracted')
		
#		if(b==[1.]):
#			print('lda concentrated')
#		else:
#			print('lda distracted')
		
	joblib.dump(clf, 'classifier.pkl') 
	joblib.dump(clf_lda, 'classifier_lda.pkl') 

	
	
	
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
		channels2 = np.zeros((4,po))
		channels3 = np.zeros((4,po))
		channels4 = np.zeros((4,po))
		channels5 = np.zeros((4,po))
		
		for x in range(0,po):			
			for y in range(0,4):				
				channels[y,x] = float(state[lastnum+x,y])
				
		for y in range(0,4):				
			channels[y] = scipy.signal.savgol_filter(channels[y], 11, 3, deriv=0, delta=1.0, axis=-1, mode='interp', cval=0.0)
		
		
		#for y in range(0,4):
		
			#nyq = 0.5 * 200
			#low = 1 / nyq
			#high = 50 / nyq
			#high2 = 70 / nyq
			#high3 = 90 / nyq
			#high4 = 95 / nyq
			#b, a = butter(5, [low, high], btype='band')
			#b2, a2 = butter(5, [low, high2], btype='band')
			#b3, a3 = butter(5, [low, high3], btype='band')
			#b4, a4 = butter(5, [low, high4], btype='band')
			
			#channels2[y] = lfilter(b, a, channels[y])
			#channels3[y] = lfilter(b2, a2, channels[y])
			#channels4[y] = lfilter(b3, a3, channels[y])
			#channels5[y] = lfilter(b4, a4, channels[y])
		
		
		
		
		
		#x = np.linspace(0,len(channels[1]),len(channels[1]))
		#f, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, sharex='col', sharey='row')
		#f.suptitle("Time Series")
		#ax1.set_ylabel('Amplitude (uV)')
		
		
		
		#ax1.plot(x, channels2[0],color='red')
		#ax1.plot(x, channels3[0],color='blue')
		#ax1.plot(x, channels4[0],color='blue')
		
		#ax1.plot(x, channels[0])
		#ax1.plot(x, channels5[0],color='yellow')
		#ax1.plot(x, y4)
		#ax1.plot(x, y5,color='red')
		#ax1.plot(x, y4,color='green')
		
		
		
		#ax1.set_title('Fp1')
		
		#ax2.plot(x, channels2[1],color='red')
		#ax2.plot(x, channels3[1],color='blue')
		
		#ax2.plot(x, channels4[1],color='blue')
		#ax2.plot(x, channels[1])
		#ax2.plot(x, y5)
		#ax2.set_title('Fp2')
		
		#ax3.plot(x, channels2[2],color='red')
		#ax3.plot(x, channels3[2],color='blue')
		#ax3.plot(x, channels4[2],color='blue')
		#ax3.plot(x,channels[2])
		#ax3.plot(x,y6)
		#ax3.set_title('O1')
		#ax3.set_xlabel('sample')
		#ax3.set_ylabel('Amplitude (uV)')
		
		#ax4.plot(x, channels2[3],color='red')
		#ax4.plot(x, channels3[3],color='blue')
		#ax4.plot(x, channels4[3],color='blue')
		#ax4.plot(x,channels[3])
		#ax4.plot(x,y6)
		#ax4.set_title('O2')
		#ax4.set_xlabel('sample')
		#plt.show()
		
		if((nof-lastnum)!=0):
			for x in range(0,4):
				hursts[x,i] = pyeeg.hurst(channels[x])
				pfds[x,i] = pyeeg.pfd(channels[x])	
				#ap_entropy[x,i] = pyeeg.ap_entropy(X, M, R)
				hfd[x,i] = pyeeg.hfd(channels[x],15)
				bins[i,x] = pyeeg.bin_power(channels[x], [0.5,4,7,12,15,18], 200)				
				k=1
		lastnum=lastnum+po
	
	return pfds,dfas,hursts,bins,hfd	
	
	
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
	
	print(concentrated)
	
	pfds_c,dfas_c,hursts_c,bins_c,hfd_c=get_state_features(concentrated)
	pfds_r,dfas_r,hursts_r,bins_r,hfd_r=get_state_features(resting)
	
	
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

	#print(all_hursts[0])
	x = np.linspace(0,len(all_bt[0])*3,len(all_bt[0]))
	#x2 = np.linspace(0,len(all_tmps[0]),len(all_tmps[0]))
	#print (x)
	
	y=all_hursts[0]
	y1=all_bt[0]
	y2=all_pfds[0]
	y3=all_hfd[0]
	y4= scipy.signal.savgol_filter(y, 11, 3, deriv=0, delta=1.0, axis=-1, mode='interp', cval=0.0)
	
	plt.plot(x, y)
	plt.title("Hurst Exponent")
	plt.ylabel("Hurst Exponent")
	plt.xlabel("Time(s)")
	plt.show()
	
	plt.plot(x, y1)
	plt.title("Theta/Beta")
	plt.ylabel("Theta/Beta")
	plt.xlabel("Time(s)")
	plt.show()
	plt.plot(x, y2)
	plt.title("Petrosian Fractal Dimensions")
	plt.ylabel("PFD")
	plt.xlabel("Time(s)")
	plt.show()
	plt.plot(x, y3)
	plt.title("Higuchi Fractal Dimensions")
	plt.ylabel("HFD")
	plt.xlabel("Time(s)")
	plt.show()
	
	plt.plot(x, y)
	plt.plot(x, y4,color='red')
	plt.title("Smoothed Hurst Exponent")
	plt.ylabel("HFD")
	plt.xlabel("Time(s)")
	plt.show()
	
	#y=all_tmps[0]
	#y1=all_tmps[1]
	#y2=all_tmps[2]
	#y3=all_tmps[3]
	#y5= scipy.signal.savgol_filter(y, 11, 3, deriv=0, delta=1.0, axis=-1, mode='interp', cval=0.0)
	y4= scipy.signal.savgol_filter(y, 11, 3, deriv=0, delta=1.0, axis=-1, mode='interp', cval=0.0)
	y5= scipy.signal.savgol_filter(y1, 11, 3, deriv=0, delta=1.0, axis=-1, mode='interp', cval=0.0)
	y6= scipy.signal.savgol_filter(y2, 11, 3, deriv=0, delta=1.0, axis=-1, mode='interp', cval=0.0)
	y7= scipy.signal.savgol_filter(y3, 11, 3, deriv=0, delta=1.0, axis=-1, mode='interp', cval=0.0)

	
	
	
	#plt.plot(x, y5,color='red')
	#plt.plot(x, y4,color='green')
	plt.show()
	
	f, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, sharex='col', sharey='row')
	
	ax1.plot(x, y)
	f.suptitle("Features")
	ax1.set_ylabel("Hurst Exponent")
	ax3.set_ylabel("Hurst Exponent")
	ax3.set_xlabel("Time (s)")
	ax4.set_xlabel("Time (s)")
	#ax1.plot(x, y4)
	#ax1.plot(x, y5,color='red')
	#ax1.plot(x, y4,color='red')
	
	ax1.set_title('hurst - ch1')
	ax2.plot(x, y1)
	#ax2.plot(x, y5,color='red')
	ax2.set_title('ch2')
	ax3.plot(x,y2)
	#ax3.plot(x,y6,color='red')
	ax3.set_title('ch3')
	ax4.plot(x,y3)
	#ax4.plot(x,y7,color='red')
	ax4.set_title('ch4')
	plt.show()
	
	
	y=all_bt[0]
	y1=all_bt[1]
	y2=all_bt[2]
	y3=all_bt[3]
	#y= scipy.signal.savgol_filter(y, 11, 3, deriv=0, delta=1.0, axis=-1, mode='interp', cval=0.0)
	#y1= scipy.signal.savgol_filter(y1, 11, 3, deriv=0, delta=1.0, axis=-1, mode='interp', cval=0.0)
	#y2= scipy.signal.savgol_filter(y2, 11, 3, deriv=0, delta=1.0, axis=-1, mode='interp', cval=0.0)
	#y3= scipy.signal.savgol_filter(y3, 11, 3, deriv=0, delta=1.0, axis=-1, mode='interp', cval=0.0)
	
	
	f, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, sharex='col', sharey='row')
	f.suptitle("Theta/Beta")
	ax1.set_ylabel("Theta/Beta")
	ax3.set_ylabel("Theta/Beta")
	ax3.set_xlabel("Time (s)")
	ax4.set_xlabel("Time (s)")
	
	ax1.plot(x, y)
	ax1.set_title('Fp1')
	ax2.plot(x, y1)
	ax2.set_title('Fp2')
	ax3.plot(x,y2)
	ax3.set_title('O1')
	ax4.plot(x,y3)
	ax4.set_title('O2')
	plt.show()
	
	
	y=all_pfds[0]
	y1=all_pfds[1]
	y2=all_pfds[2]
	y3=all_pfds[3]
	#y= scipy.signal.savgol_filter(y, 11, 3, deriv=0, delta=1.0, axis=-1, mode='interp', cval=0.0)
	#y1= scipy.signal.savgol_filter(y1, 11, 3, deriv=0, delta=1.0, axis=-1, mode='interp', cval=0.0)
	#y2= scipy.signal.savgol_filter(y2, 11, 3, deriv=0, delta=1.0, axis=-1, mode='interp', cval=0.0)
	#y3= scipy.signal.savgol_filter(y3, 11, 3, deriv=0, delta=1.0, axis=-1, mode='interp', cval=0.0)
	
	
	f, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, sharex='col', sharey='row')
	
	
	f.suptitle("PFD")
	ax1.set_ylabel("PFD")
	ax3.set_ylabel("PFD")
	ax3.set_xlabel("Time (s)")
	ax4.set_xlabel("Time (s)")
	
	
	ax1.plot(x, y)
	ax1.set_title('Fp1')
	ax2.plot(x, y1)
	ax2.set_title('Fp2')
	ax3.plot(x,y2)
	ax3.set_title('O1')
	ax4.plot(x,y3)
	ax4.set_title('O2')
	plt.show()
	
	y=all_hfd[0]
	y1=all_hfd[1]
	y2=all_hfd[2]
	y3=all_hfd[3]
	#y= scipy.signal.savgol_filter(y, 11, 3, deriv=0, delta=1.0, axis=-1, mode='interp', cval=0.0)
	#y1= scipy.signal.savgol_filter(y1, 11, 3, deriv=0, delta=1.0, axis=-1, mode='interp', cval=0.0)
	#y2= scipy.signal.savgol_filter(y2, 11, 3, deriv=0, delta=1.0, axis=-1, mode='interp', cval=0.0)
	#y3= scipy.signal.savgol_filter(y3, 11, 3, deriv=0, delta=1.0, axis=-1, mode='interp', cval=0.0)
	
	
	f, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, sharex='col', sharey='row')
	f.suptitle("Higuchi Fractial Dimensions")
	ax1.set_ylabel("HFD")
	ax3.set_ylabel("HFD")
	ax3.set_xlabel("Time (s)")
	ax4.set_xlabel("Time (s)")
	
	
	ax1.plot(x, y)
	ax1.set_title('Fp1')
	ax2.plot(x, y1)
	ax2.set_title('Fp2')
	ax3.plot(x,y2)
	ax3.set_title('O1')
	ax4.plot(x,y3)
	ax4.set_title('O2')
	plt.show()	
		
	for i in range (0,4):
		all_hursts[i]= scipy.signal.savgol_filter(all_hursts[i], 11, 3, deriv=0, delta=1.0, axis=-1, mode='interp', cval=0.0)
		all_bt[i]= scipy.signal.savgol_filter(all_bt[i], 11, 3, deriv=0, delta=1.0, axis=-1, mode='interp', cval=0.0)
		all_pfds[i]= scipy.signal.savgol_filter(all_pfds[i], 11, 3, deriv=0, delta=1.0, axis=-1, mode='interp', cval=0.0)
		all_hfd[i]= scipy.signal.savgol_filter(all_hfd[i], 11, 3, deriv=0, delta=1.0, axis=-1, mode='interp', cval=0.0)
	
	
	ml(all_hursts,all_bt,all_pfds,all_hfd,all_targets,len(all_hursts[0]))
	
	
	
	abc = np.zeros((4,nof))
	
	for x in range(0,nof): #numberOfFrames-lastnum-6		
		for y in range(0,4):			
			abc[y,x] = float(lines[x+6].split(',')[y+1])
	
	Pxx, freqs, bins2, im = plt.specgram(abc[0], Fs=200)
	plt.show()
	
	
	
	
	for i in range (0,(int(nof/po))):
		ok = getdata(i)
		forever= False

	a= np.zeros((nof/po))
	b= np.zeros((nof/po))
	c= np.zeros((nof/po))
	d = np.zeros((nof/po))
	e = np.zeros((nof/po))	
	for i in range (0,nof/po):
		a[i] = bins[i,0,1,0]
		b[i] = bins[i,0,1,1]
		c[i] = bins[i,0,1,2]
		d[i] = bins[i,0,1,3]
		e[i] = b[i]/d[i]		
	x = np.linspace(0,nof,nof/po)

	y= a.copy()	
	y1= b	
	y2= c
	y3= d	
	f, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, sharex='col', sharey='row')
	ax1.plot(x, y)
	ax1.set_title('delta (0.5-4)')
	ax2.plot(x, y1)
	ax2.set_title('theta (4-7)')
	ax3.plot(x,y2)
	ax3.set_title('alpha (7-12)')
	ax4.plot(x,y3)
	ax4.set_title('beta (12-30)')
	plt.show()
	
	y4 = hursts[1]
	
	f = plt.plot(x,y4)
	plt.title("hurst expo")
	plt.show()
	
	
	y5 = pfds[1]
	
	f = plt.plot(x,y5)
	plt.title("pfds")
	plt.show()
	
	
	y6 = dfas[1]
	
	f = plt.plot(x,y6)
	plt.title("dfas")
	plt.show()
	
	y7 = e
	
	f = plt.plot(x,y7)
	plt.title("theta/beta")
	plt.show()