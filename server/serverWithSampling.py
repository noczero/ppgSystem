import paho.mqtt.client as mqtt
import time
import datetime
import csv
import numpy as np
import pywt
import operator
import warnings
warnings.filterwarnings("ignore")
from train_SVM import *
from sklearn.externals import joblib
import statistics


dt = datetime.datetime.now()
#brokerHost = "hantamsurga.net"
brokerHost = "192.168.43.19"
#brokerHost = "192.168.1.2"
#port = 49877
port = 1883

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

def compute_LBP(signal, neigh=4):
    hist_u_lbp = np.zeros(np.power(2, neigh), dtype=float)

    avg_win_size = 2
    # TODO: use some type of average of the data instead the full signal...
    # Average window-5 of the signal?
    #signal_avg = average_signal(signal, avg_win_size)
    signal_avg = scipy.signal.resample(signal, len(signal) / avg_win_size)

    for i in range(neigh/2, len(signal) - neigh/2):
        pattern = np.zeros(neigh)
        ind = 0
        for n in range(-neigh/2,0) + range(1,neigh/2+1):
            if signal[i] > signal[i+n]:
                pattern[ind] = 1
            ind += 1
        # Convert pattern to id-int 0-255 (for neigh == 8)
        pattern_id = int("".join(str(c) for c in pattern.astype(int)), 2)

        hist_u_lbp[pattern_id] += 1.0

    return hist_u_lbp

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

    f_lbp = np.empty((0, 16 * num_leads))
    f_lbp_lead = np.empty([])
    f_lbp_lead = compute_LBP(signal, 4)
    print("Feature Extraction : LBP...")
    features = []
    features = np.hstack((f_wav_lead,f_lbp_lead))

    return features

def testingData(signal, multi_mode, voting_strategy):
    """
    Do testing data on SVM
    :param signal:
    :return:
    """
    # load trained SVM Model
    model_svm_path = "../model/trained/trained_model_wvlt_lbp.joblib.pkl"
    svm_model = joblib.load(model_svm_path)

    m = statistics.mean(signal)
    variance = statistics.variance(signal)
    print(m)
    print(variance)
    if (m > 10):
        print("Please put the device in your wrist...")
        clientMQTT.publish("sensor/PPG001/n", "release")
    else :
        feature = featureExtraction(signal)

        feature = np.vstack((feature,feature))

        y_predict = svm_model.predict(feature)

        # actual class 0 = normal, af = 1
        if (y_predict[1] == 1.):
            print "Predict Status : Normal"
            clientMQTT.publish("sensor/PPG001/n" , "normal")
        elif (y_predict[1] == 0.):
            print "Predict Status : AF"
            clientMQTT.publish("sensor/PPG001/n" , "af")


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
    print "message topic=", message.topic, " - qos=", message.qos, " - flag=", message.retain
    receivedMessage = str(message.payload.decode("utf-8"))
    print "received message = ", receivedMessage

    signal = receivedMessage.split(':')
    # print(signal)
    write_tocsv(signal)  # write signal to CSV

    # testing features
    features = np.array([], dtype=float)
    # convert to float
    for x in range(len(signal) - 1):
        # print(float(signal[x]))
        features = np.hstack((features, float(signal[x])))

    # check if features coming with id then don't use the id
    if (len(features) == 200):
        features = features[1:200]

    result = testingData(features, 'ovo', 'ovo_voting')  # testing the signal


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
    subTopic = "sensor/PPG001/signal"
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

