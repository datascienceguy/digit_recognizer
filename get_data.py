import numpy as np


def read_in_data(use_sample_data = False):

    train_file = "data/sample/train.csv" if use_sample_data else "data/train.csv"
    test_file = "data/sample/test.csv" if use_sample_data else "data/test.csv"


    # load the CSV file as a numpy matrix (the first row is header)
    # Column 0 is the digit label, columns 1:783 are pixel values (for a 28 x 28) image
    train_data = np.loadtxt(train_file, delimiter=",", skiprows=1)
    features_train = train_data[:, 1:]  # select columns 1 through end to get all pixels
    labels_train = train_data[:, 0]   # column 0 is the label of the digit

    # load the CSV file as a numpy matrix
    test_data = np.loadtxt(test_file, delimiter=",", skiprows=1)
    if(use_sample_data):
        features_test = test_data[:, 1:]  # select columns 0 through end to get all pixels
        labels_test = test_data[:, 0]  # select columns 0 for the labels (if using sample data)
    else:
        features_test = test_data[:, 0:]  # select columns 0 through end to get all pixels

    return features_train, labels_train, features_test, labels_test
