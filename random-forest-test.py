from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA
from get_data import read_in_data

print 'Reading in the data...\n'
use_sample_data = True
features_train, labels_train, features_test, labels_test = read_in_data(use_sample_data)

print 'Running PCA...\n'
pca = PCA(n_components=100, whiten=True)
pca.fit(features_train)
pca_features_train = pca.transform(features_train)
pca_features_test = pca.transform(features_test)

'Print running random forest...\n'
num_estimators = [100, 150]
max_features = [10, 20, 30, 35, 40, 50, 75, 100]
for num_estimator in num_estimators:
    for max_feature in max_features:

        clf = RandomForestClassifier(n_estimators=num_estimator, max_features=max_feature)
        clf.fit(pca_features_train, labels_train)

        predictions = clf.predict(pca_features_test)

        accuracy = accuracy_score(labels_test, predictions)
        print 'Accuracy with ' + str(num_estimator) + ' estimators and ' \
            + str(max_feature) + ' max features: ' + str(accuracy) + '\n'
