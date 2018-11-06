import random
import time
import pyeeg
import sklearn
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis


#from pythonosc import osc_message_builder
#from pythonosc import udp_client

from OSC import OSCClient, OSCMessage
lastnum=0

global nof
global po
client = OSCClient()
client.connect( ("127.0.0.1", 8000) )	

forever=True
counter = 0
while(forever):
	#a=counter%2
	a=1
	counter+=1
	
	k = OSCMessage('/vol')		
	
	k.append(int(a))			
	client.send(k)
	k.clear()
	time.sleep(0.2)
