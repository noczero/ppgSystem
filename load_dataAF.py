"""
Load Data from File
Process load_ppg_file -> load_signal
"""

import os
import csv
import gc
import cPickle as pickle
import numpy as np
from features_ECG import *


# training dataset
DS_train = ['AF_01' , 'AF_03' , 'AF_05' , 'AF_07' , 'AF_09' , 'AF_11', 'AF_13' , 'AF_15' , 'AF_17' , 'AF_19' , 'N_01' , 'N_03' , 'N_05' , 'N_07' , 'N_09' , 'N_11', 'N_13' , 'N_15' , 'N_17' ,'N_02' , 'N_04' , 'N_06' , 'N_08' , 'N_10' , 'N_12' ]
DS_testing = ['AF_02' , 'AF_04' , 'AF_06' , 'AF_08' , 'AF_10' , 'AF_12', 'AF_14' , 'AF_16' , 'AF_18' , 'AF_20' ,  'N_14' , 'N_16' , 'N_18' ]
pathDB = 'dataset/'

class ppg_db:
    def __init__(self):
        #instance
        self.filename = []
        self.RAW_signal = []
        self.beat = np.empty([])
        self.class_ID = np.empty([])

def load_signal(DS, sampling ):
    """
    DS is list of filename for dataset
    Sampling is selected array maximum length
    status is decide for Normal or AF signal
    :param DS:
    :param winL:
    :param winR:
    :param status:
    :return:
    """
    my_db = ppg_db()

    file_records = list()

    # get all file in the Normal directory
    folder_name = 'Normal/'
    list_file = os.listdir(pathDB + folder_name)
    list_file.sort()

    for file in list_file:
        # check just file name
        if file[0:4] in DS:
            file_records.append(file)

    # get all file in the AF directory
    folder_name = 'AF/'
    list_file = os.listdir(pathDB + folder_name)
    list_file.sort()

    for file in list_file:
        #print(file[0:5])
        if file[0:5] in DS:
            file_records.append(file)

    # inialize variable
    class_ID = [[] for i in range(len(DS))]
    beat = [[] for i in range(len(DS))]
    valid_R = [ np.array([]) for i in range(len(DS))]

    for myFile in range(0, len(file_records)):
        print("Processing signal... " + str(myFile) + " / " + str(len(file_records)) + "...")


        # check is each file is named with Normal or AF
        # print(file_records[myFile][0]) => N or A
        if (file_records[myFile][0] == 'N'):
            filename = pathDB + 'Normal/' + file_records[myFile]
            print("file name " + str(filename))
            f = open(filename, 'rb')
            reader = csv.reader(f, delimiter=',')

            RAW_signal_N = []
            for row in reader:
                # save signal to list
                RAW_signal_N.append(float(row[0]))

            # iterate in signal
            selectedSignal = []
            for i in range(0,len(RAW_signal_N)):
                selectedSignal.append(RAW_signal_N[i])
                # sampling every 180 unit
                if( i % (sampling-1) == 0 and i > 0 ):
                    beat[myFile].append(selectedSignal)
                    class_ID[myFile].append(0) # label the beat for Normal
                    selectedSignal = []


        elif(file_records[myFile][0] == 'A'):
            filename = pathDB + 'AF/' + file_records[myFile]
            print(filename)
            f = open(filename, 'rb')
            reader = csv.reader(f, delimiter=',')

            RAW_signal_AF = []
            for row in reader:
                # save signal to list
                RAW_signal_AF.append(float(row[0]))

            # iterate in signal
            selectedSignal = []
            for i in range(0, len(RAW_signal_AF)):
                selectedSignal.append(RAW_signal_AF[i])
                # sampling every 180 unit
                if (i % (sampling - 1) == 0 and i > 0):
                    beat[myFile].append(selectedSignal)
                    class_ID[myFile].append(1)  # label the beat for AF
                    selectedSignal = []

    print("Complete Load File")
    my_db.filename = file_records
    my_db.beat = beat
    my_db.class_ID = class_ID

    return my_db

def load_ppg_db(type, features):
    """
    Load PPG and process for the features
    :param type: string train or testing
    :param features: wvlt
    :return:
    """
    path_model = 'model/'
    features_labels_name = path_model+ 'ppg_' + type + '_' + features + '.p'
    path_db_pickle_name = path_model + 'data_'

    if os.path.isfile(features_labels_name):
        print("Loading features pickle : " + features_labels_name + "...")

        f = open(features_labels_name , 'rb')
        gc.disable()
        features,labels = pickle.load(f)
        gc.enable()
        f.close
        # done
    else:
        # if features not process yet
        print("Features processing...")
        print("Loading PPG Data..." + type + "...")

        # process type
        path_db_pickle_name = path_db_pickle_name + type + '.p'
        if os.path.isfile(path_db_pickle_name):
            print("Signal already process, just loading..")
            f = open(path_db_pickle_name , 'rb')
            gc.disable()
            ppg_db = pickle.load(f)
            gc.enable()
            f.close()
        else:
            # if there is not process signal
            print("Processing signal started...")
            if type == 'train':
                ppg_db = load_signal(DS_train,200) # sampling 200
            elif type == 'testing':
                ppg_db = load_signal(DS_testing,200) # sampling 200

            print("Saving signal processed data ...")
            f = open(path_db_pickle_name, 'wb')
            pickle.dump(ppg_db, f, 2)
            f.close

        features = np.array([], dtype=float)
        labels = np.array([] , dtype=np.int32)

        # feature extraction
        print("Wavelets ...")
        f_wav = np.empty((0,25 * 1)) # suitable to length of sampling

        for p in range(len(ppg_db.beat)):
            for b in ppg_db.beat[p]:
                f_wav_lead = np.empty([])
                f_wav_lead = compute_wavelet_descriptor(b, 'db1',3)

                f_wav = np.vstack((f_wav, f_wav_lead))

        features = np.coloumn_stack((features , f_wav)) if features.size else f_wav

        print("lbp ...")
        f_lbp = np.empty((0, 16 * 1))
        for p in range(len(ppg_db.beat)):
            for b in ppg_db.beat[p]:
                f_lbp_lead = np.empty([])
                f_lbp_lead = compute_LBP(b, 4)

                f_lbp = np.vstack((f_lbp, f_lbp_lead))

        features = np.column_stack((features, f_lbp)) if features.size else f_lbp

        labels = np.array(sum(ppg_db.class_ID , [])).flatten()
        print("Done...")

        # save the file
        print("Writing pickle : " + features_labels_name + "...")
        f = open(features_labels_name, 'wb')
        pickle.dump([features, labels] , f ,2)
        f.close

    return features, labels



"""

## for training
train_db = load_signal(DS_train,180)
path_db_pickle_name = 'model/data_train_180sampling.p'
f = open(path_db_pickle_name, 'wb')
pickle.dump(train_db, f, 2)
f.close

## for testing
test_db = load_signal(DS_testing,180)
path_db_pickle_name = 'model/data_testing_180sampling.p'
f = open(path_db_pickle_name, 'wb')
pickle.dump(test_db, f, 2)
f.close
"""


