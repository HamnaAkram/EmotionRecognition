import glob
import math
import os
import random
import matplotlib.pyplot as plt

import cv2
import dlib
import numpy as np
from sklearn.svm import SVC

emotions = ["anger", "contempt", "disgust", "fear", "happiness", "neutral", "sadness", "surprise"]  # Emotion list
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")  # Or set this to whatever you named the downloaded file
clf = SVC(kernel='linear', probability=True,tol=1e-3)  # , verbose = True) #Set the classifier as a support vector machines with polynomial kernel

data = {}  # Make dictionary for all values

data['landmarks_vectorised'] = []

test_img = []  # Make dictionary for all values

#####################################################################################################################################
def get_files(emotion):  #Define function to get file list, randomly shuffle it and split 80/20
    os.chdir('/home/hamna/PycharmProjects/data_preprocessing/dataset/%s/' % emotion)
    files = glob.glob("*")
    random.shuffle(files)
    training = files[:int(len(files) * 0.8)]  # get first 80% of file list
    prediction = files[-int(len(files) * 0.2):]  # get last 20% of file list
    return training, prediction

###############################################################################################################################


def get_landmarks(image):
    detections = detector(image, 1)
    for k, d in enumerate(detections):  # For all detected face instances individually
        shape = predictor(image, d)  # Draw Facial Landmarks with the predictor class
        xlist = []
        ylist = []
        for i in range(1, 68):  # Store X and Y coordinates in two lists
            xlist.append(float(shape.part(i).x))
            ylist.append(float(shape.part(i).y))

        xmean = np.mean(xlist)
        ymean = np.mean(ylist)
        xcentral = [(x - xmean) for x in xlist]
        ycentral = [(y - ymean) for y in ylist]

        landmarks_vectorised = []
        for x, y, w, z in zip(xcentral, ycentral, xlist, ylist):
            landmarks_vectorised.append(w)
            landmarks_vectorised.append(z)
            meannp = np.asarray((ymean, xmean))
            coornp = np.asarray((z, w))
            dist = np.linalg.norm(coornp - meannp)
            landmarks_vectorised.append(dist)
            landmarks_vectorised.append((math.atan2(y, x) * 360) / (2 * math.pi))

        data['landmarks_vectorised'] = landmarks_vectorised
    if len(detections) < 1:
        data['landmarks_vestorised'] = "error"


###############################################################################################################################


def make_sets():
    training_data = []
    training_labels = []
    prediction_data = []
    prediction_labels = []
    for emotion in emotions:
        print(" working on %s" % emotion)
        training, prediction = get_files(emotion)
        # Append data to training and prediction list, and generate labels 0-7
        for item in training:
            image = cv2.imread(item)  # open image
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # convert to grayscale
            clahe_image = clahe.apply(gray)
            get_landmarks(clahe_image)
            if data['landmarks_vectorised'] == "error":
                print("no face detected on this one")
            else:
                training_data.append(data['landmarks_vectorised'])  # append image array to training data list
                training_labels.append(emotions.index(emotion))

        for item in prediction:
            image = cv2.imread(item)
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            clahe_image = clahe.apply(gray)
            test_img.append(clahe_image)
            get_landmarks(clahe_image)
            if data['landmarks_vectorised'] == "error":
                print("no face detected on this one")
            else:
                prediction_data.append(data['landmarks_vectorised'])
                prediction_labels.append(emotions.index(emotion))

    return training_data, training_labels, prediction_data, prediction_labels


###############################################################################################################################


def train_predict_system():
    print("******************************************Welcome to Emotion Recognition System*****************************************")
    print('###########################Initialising System#########################################################')
    accur_lin = []

    for i in range(0, 10):
        print("Making set No. %s" % i)  # Make sets by random sampling 80/20%
        training_data, training_labels, prediction_data, prediction_labels = make_sets()

        npar_train = np.array(training_data)  # Turn the training set into a numpy array for the classifier
        print("training SVM linear for set %s" % i)  # train SVM
        clf.fit(npar_train, training_labels)

        print("getting accuracies of set %s" % i)  # Use score() function to get accuracy
        npar_pred = np.array(prediction_data)
        pred_lin = clf.score(npar_pred, prediction_labels)
        print("Mean value of linear SVM: ", pred_lin)
        accur_lin.append(pred_lin)
        # Store accuracy in a list

    print("Mean value of complete runs: %s" % np.mean(accur_lin))# FGet mean accuracy of the 10 runs
    res = clf.predict_proba(npar_pred)
    result = np.round(res,3)

    ##for row in result:
        ##print('         '.join([str(elem) for elem in row]))




    display_data(result)

###############################################################################################################################

def display_data(probability):

        for i in range(0,70):
            plt.figure(i)
            plt.subplot(1, 2, 1)
            plt.imshow(test_img[i])

            plt.subplot(122)
            objects = emotions
            y_pos = np.arange(len(objects))
            performance = probability[i]

            plt.bar(y_pos, performance, align='center', alpha=0.5)
            plt.xticks(y_pos, objects)
            plt.ylabel('Probability')
            plt.title('Expected Emotions')

            plt.show()

        


###############################################################################################################################




train_predict_system()