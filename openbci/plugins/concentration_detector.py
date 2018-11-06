import plugin_interface as plugintypes
import numpy as np
from openbci.utils import pyeeg
from sklearn.externals import joblib

numberOfFrames = 200

def getdata(readings):
    channels = np.zeros((4, numberOfFrames))

    for x in range(0, numberOfFrames):
        for y in range(0, 4):
            channels[y, x] = float(readings[x].split(',')[y])
    return channels

def get_state_features(channel):
    nof = len(channel)
    pfds = np.zeros((4))
    ap_entropy = np.zeros((4))
    hursts = np.zeros((4))
    hfd = np.zeros((4))
    bins = np.zeros(((4, 2, 5)))

    lastnum = 0

    # alpha=[]
    if ((nof - lastnum) != 0):
        for x in range(0, 4):
            hursts[x] = pyeeg.hurst(channel[x])
            pfds[x] = pyeeg.pfd(channel[x])
            # ap_entropy[x,i] = pyeeg.ap_entropy(X, M, R)
            hfd[x] = pyeeg.hfd(channel[x], 15)
            bins[x] = pyeeg.bin_power(channel[x], [0.5, 4, 7, 12, 15, 18], 200)

    delta = np.zeros((4))
    beta = np.zeros((4))
    alpha = np.zeros((4))
    theta = np.zeros((4))
    dfas = np.zeros((4))
    bt = np.zeros((4))

    for y in range(0, 4):
        delta[y] = bins[y, 0, 0]
        theta[y] = bins[y, 0, 1]
        alpha[y] = bins[y, 0, 2]
        beta[y] = bins[y, 0, 4]
        bt[y] = theta[y] / beta[y]

    lastnum = lastnum + nof

    return pfds, dfas, hursts, bins, bt, hfd

def ml(hursts, bt, pfds, hfd):
    data = np.zeros((16))

    #print(hursts)
    for y in range(0, 4):
        data[y] = hursts[y]
        data[y + 4] = bt[y]
        data[y + 8] = pfds[y]
        data[y + 12] = hfd[y]

    clf = joblib.load('data/classifier.pkl')
    clf_lda = joblib.load('data/classifier_lda.pkl')

    for i in range(0, len(data)):
        if (np.all(np.isfinite(data[i])) == False):
            data[i] = 0.4

    a = clf.predict(data.reshape(1, -1))
    b = clf_lda.predict(data.reshape(1, -1))

    return a, b

class PluginPrint(plugintypes.IPluginExtended):
    counter = 0
    readings = []

    def activate(self):
        print("write_to_file activated")

    # called with each new sample
    def __call__(self, sample):
        if sample:
            sample_string = "%s\n" % (
                str(sample.channel_data)[1:-1])
            self.readings.append(sample_string)
            if (self.counter == 200):
                self.check_mental_state()
                self.counter = 0
                self.readings = []

            self.counter += 1

    def check_mental_state(self):
        channels = getdata(readings=self.readings)
        # print("data got")
        pfds, dfas, hursts, bins, bt, hfd = get_state_features(channels)
        # print("features got")
        a, b = ml(hursts, bt, pfds, hfd)

        if (a == [1.]):
            print('concentrated')
        else:
            print('distracted')

        if (b == [1.]):
            print('lda concentrated')
        else:
            print('lda distracted')
