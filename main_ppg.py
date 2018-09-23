from load_dataAF import *
import sklearn
from sklearn.externals import joblib
from sklearn.preprocessing import StandardScaler
from sklearn import svm
from sklearn.metrics import classification_report, accuracy_score
import  time

type = 'testing'
db_path = 'model/trained/'
C_value = 10
use_probability = False
multi_mode = 'ovo'
features = 'wvlt_lbp'

if type == 'train':
    print("Training process...")
    [train_features , train_labels] = load_ppg_db(type,features)
    model_trained_path = db_path + 'trained_model_'+features+'.joblib.pkl'

    if os.path.isfile(model_trained_path):
        # load the trained model
        print "Load traine" \
              "'d model..."
        train_model = joblib.load(model_trained_path)
    else:
        print "Training.."
        class_weights = {}
        for c in range(2): # c from 0 to 1
            class_weights.update({c:len(train_labels) / float(np.count_nonzero(train_labels == c))})

        svm_model = svm.SVC(C=C_value, kernel='rbf', degree=2, gamma='auto',
                            coef0=0.0, shrinking=True, probability=use_probability, tol=0.001,
                            cache_size=200, class_weight=class_weights, verbose=False,
                            max_iter=-1, random_state=None)
        #train process
        start = time.time() # time
        svm_model.fit(train_features , train_labels)
        end = time.time()

        print("Trained completed!\n\t" + model_trained_path + "\n \
                       \tTime required: " + str(format(end - start, '.2f')) + " sec")

        # Export model: save/write trained model
        joblib.dump(svm_model, model_trained_path)

elif(type == 'testing'):
    print("Testing process...")
    [testing_features, testing_labels] = load_ppg_db(type, features)
    model_trained_path = db_path + 'trained_model_'+features+'.joblib.pkl'

    if os.path.isfile(model_trained_path):
        # load the trained model
        print "Load trained model..."
        train_model = joblib.load(model_trained_path)

        y_predict = train_model.predict(testing_features)
        target_names = ['Normal', 'AFIB']

        print(classification_report(testing_labels, y_predict, target_names=target_names))
        print(accuracy_score(testing_labels, y_predict))
    else:
        print "Please, training the model"

