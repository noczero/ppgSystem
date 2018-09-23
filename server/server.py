import paho.mqtt.client as mqtt
import time
import datetime
import csv
import numpy as np
import pywt
import operator
from train_SVM import *

import sklearn
from sklearn.externals import joblib
from sklearn import svm


dt = datetime.datetime.now()
brokerHost = "hantamsurga.net"
#brokerHost = "192.168.43.206"
port = 49877
#port = 1883

logfile = 'ppgSignal-%s-%s-%s.csv' % (dt.day, dt.month, dt.year)


# Collect realtime data
savePoint = False
increment = 0
#listSignal = np.array([])
listSignal = []
def collectRealtimeSignal(signalRealtime, windowLength):
    """
    Collect realtime
    :param signalRealtime: input signal
    :param windowLength: maximal
    :return: list of signal
    """
    global listSignal , increment
    #while (increment < windowLength):
    #    listSignal = np.hstack((listSignal,signalRealtime))
    #    increment = increment + 1

    if (increment < windowLength):
        #listSignal = np.hstack((listSignal, signalRealtime))
        listSignal.append(signalRealtime)
        increment = increment + 1
        print(increment)
        print(listSignal)
    else:
        increment = 0
        print("Masuk")
        return listSignal
    return 0

# Compute the wavelet descriptor for a beat
def compute_wavelet_descriptor(beat, family, level):
    wave_family = pywt.Wavelet(family)
    coeffs = pywt.wavedec(beat, wave_family, level=level)
    return coeffs[0]

# Compute my descriptor based on amplitudes of several intervals
def compute_my_own_descriptor(beat, winL, winR):
    R_pos = int((winL + winR) / 2)

    R_value = beat[R_pos]
    my_morph = np.zeros((4))
    y_values = np.zeros(4)
    x_values = np.zeros(4)
    # Obtain (max/min) values and index from the intervals
    [x_values[0], y_values[0]] = max(enumerate(beat[0:40]), key=operator.itemgetter(1))
    [x_values[1], y_values[1]] = min(enumerate(beat[75:85]), key=operator.itemgetter(1))
    [x_values[2], y_values[2]] = min(enumerate(beat[95:105]), key=operator.itemgetter(1))
    [x_values[3], y_values[3]] = max(enumerate(beat[150:180]), key=operator.itemgetter(1))

    x_values[1] = x_values[1] + 75
    x_values[2] = x_values[2] + 95
    x_values[3] = x_values[3] + 150

    # Norm data before compute distance
    x_max = max(x_values)
    y_max = max(np.append(y_values, R_value))
    x_min = min(x_values)
    y_min = min(np.append(y_values, R_value))

    R_pos = (R_pos - x_min) / (x_max - x_min)
    R_value = (R_value - y_min) / (y_max - y_min)

    for n in range(0, 4):
        x_values[n] = (x_values[n] - x_min) / (x_max - x_min)
        y_values[n] = (y_values[n] - y_min) / (y_max - y_min)
        x_diff = (R_pos - x_values[n])
        y_diff = R_value - y_values[n]
        my_morph[n] = np.linalg.norm([x_diff, y_diff])
        # TODO test with np.sqrt(np.dot(x_diff, y_diff))

    if np.isnan(my_morph[n]):
        my_morph[n] = 0.0

    return my_morph

def featureExtraction(signal):
    """
    Extract feature from signal
    :param signal:
    :return:
    """
    num_leads = 1
    leads_flag = [1, 0]  # MLII, V1
    print("Feature Extraction : Wavelets...")

    f_wav = np.empty((0, 23 * num_leads))

    f_wav_lead = np.empty([])
    f_wav_lead = compute_wavelet_descriptor(signal, 'db1', 3)

    return f_wav_lead

def testingData(signal, multi_mode, voting_strategy):
    """
    Do testing data on SVM
    :param signal:
    :return:
    """
    # load trained SVM Model
    model_svm_path = "svm_models\ovo_rbf_MLII_rm_bsln_wvlt_weighted_C_100.joblib.pkl"
    svm_model = joblib.load(model_svm_path)

    # preprocessing signal
    # median_filter
    """
    baseline = medfilt(signal, 71)
    baseline = medfilt(baseline, 215)
    signal = signal - baseline

    # RAW Signal extracted
    features = np.array([], dtype=float)
    features = featureExtraction(signal)
    """
    # Normalizatoon features
    # scaler = StandardScaler()
    # scaler.fit(features)
    # tr_features_scaled = scaler.transform(features)

    feature = featureExtraction(signal)

    if multi_mode == 'ovo':
        decision_ovo = svm_model.decision_function(feature)

        if voting_strategy == 'ovo_voting':
            predict_ovo, counter = ovo_voting(decision_ovo, 4)

        elif voting_strategy == 'ovo_voting_both':
            predict_ovo, counter = ovo_voting_both(decision_ovo, 4)

        elif voting_strategy == 'ovo_voting_exp':
            predict_ovo, counter = ovo_voting_exp(decision_ovo, 4)


        if (predict_ovo[1] == 0.):
            print "Status : Normal"
            clientMQTT.publish("ppg/signal/n" , "normal")
        elif (predict_ovo[1] == 1.):
            print "Status : AF"
            clientMQTT.publish("ppg/signal/n" , "af")


# csvwrite
def write_tocsv(data) :
    """
    Write incoming data from MQTT to csv file
    :param data:
    :return:
    """
    with open(logfile, "a") as output_file:
        writer = csv.writer(output_file, delimiter=',', lineterminator='\r')
        writer.writerow(data)

#define callback
def on_message(client, userdata, message):
    """
    Every data incoming from broker will call this function, implement all of logic here
    :param client:
    :param userdata:
    :param message:
    :return:
    """
    global listSignal
    #print "message topic=", message.topic , " - qos=", message.qos , " - flag=", message.retain
    receivedMessage = str(message.payload.decode("utf-8"))
    #print "received message = " , receivedMessage

    signal = receivedMessage.split(':')

    #print(signal)


    listSignal.append(signal)
    #print(listSignal)
    # waiting for full 1 window for 180 incoming data
    if(len(listSignal) == 180):
        saveListSignal = listSignal

        write_tocsv(listSignal)
        #convert to float
        features = np.array([], dtype=float)
        # convert to float
        for x in range(len(saveListSignal) - 1):
            # print(float(signal[x]))
            features = np.hstack((features,float(saveListSignal[x])))

        features = np.vstack((features, features))  # become 2 d array
        print("Testing incoming data ...")
        testingData(features,'ovo','ovo_voting')

        # reset the list
        listSignal = []



def on_connect(client, userdata, flags, rc):
    """
    call when mqtt connect
    :param client:
    :param userdata:
    :param flags:
    :param rc:
    :return:
    """
    print("Connected with result code "+ str(rc))
    # subscribe the topic "ekg/device1/signal"
    # subTopic = "ekg/+/signal"  # + is wildcard for all string to that level
    # subTopic = "rhythm/PPG004/ppg"
    subTopic = "ppg/signal"
    print "Subscribe topic ", subTopic
    clientMQTT.subscribe(subTopic)


"""
main program
"""
# create client object
clientMQTT = mqtt.Client("client-Server")

# set callback
clientMQTT.on_message = on_message
clientMQTT.on_connect = on_connect

# connection established
print "connecting to broker" , brokerHost
clientMQTT.connect(brokerHost,port) # connect to broker

clientMQTT.loop_forever() # loop forever

