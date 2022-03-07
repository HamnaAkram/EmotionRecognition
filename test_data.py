
tdata = {}  # Make dictionary for all values

tdata['landmarks_vectorised'] = []


###############################################################################################################################


def get_testfiles(emotion):  # Define function to get file list, randomly shuffle it and split 80/20
    os.chdir('/home/hamna/PycharmProjects/EmotionRecognition/testdata/%s/' % emotion)
    files = glob.glob("*")
    random.shuffle(files)
    test_data = files
    return test_data


###############################################################################################################################


def make_test_set():
    test_data = []
    test_labels = []
    for emotion in emotions:
        print(" working on %s" % emotion)
        test = get_testfiles(emotion)
        # Append data to training and prediction list, and generate labels 0-7

        for item in test:
            image = cv2.imread(item)
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            clahe_image = clahe.apply(gray)
            get_landmarks(clahe_image)
            if tdata['landmarks_vectorised'] == "error":
                print("no face detected on this one")
            else:
                test_data.append(tdata['landmarks_vectorised'])
                test_labels.append(emotions.index(emotion))

    return test_data, test_labels


###############################################################################################################################