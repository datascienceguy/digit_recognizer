from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.decomposition import PCA
import numpy as np
from get_data import read_in_data

use_sample_data = True
features_train, labels_train, features_test, labels_test = read_in_data(use_sample_data)

# Create the SVM classifier
# clf = SVC(kernel='rbf', C=100)
clf = SVC()

# Fit the classifier on training data
clf.fit(features_train, labels_train)

# Make predictions on test data
pred = clf.predict(features_test)

accuracyBefore = accuracy_score(labels_test, pred)
print 'Accuracy before: ' + str(accuracyBefore) + '\n'

num_components = [35]
# num_components = [10, 30, 35, 40, 50, 100, 150, 200, 250]
for component in num_components:
    pca = PCA(n_components=component, whiten=True)
    pca.fit(features_train)
    pca_features_train = pca.transform(features_train)
    pca_features_test = pca.transform(features_test)

    # Refit after using PCA on features
    clf.fit(pca_features_train, labels_train)

    #re-predict
    pred = clf.predict(pca_features_test)
    accuracyPCA = accuracy_score(labels_test, pred)
    print 'Accuracy with '+str(component)+' num components: ' + str(accuracyPCA) + '\n'

# # Output:
# Accuracy before: 0.142
# PCA accuracies with PCA
# Accuracy with 10 num components: 0.912
# Accuracy with 30 num components: 0.958
# Accuracy with 50 num components: 0.96
# Accuracy with 100 num components: 0.948
# Accuracy with 150 num components: 0.942
# Accuracy with 200 num components: 0.93
# Accuracy with 250 num components: 0.914
