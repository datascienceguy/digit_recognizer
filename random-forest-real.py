from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from get_data import read_in_data

use_sample_data = False
features_train, labels_train, features_test, labels_test = read_in_data(use_sample_data)


num_estimators = [100, 150]
max_features = [10, 20, 30, 35, 40, 50]
for num_estimator in num_estimators:
    for max_feature in max_features:
        clf = RandomForestClassifier(n_estimators=num_estimator, max_features=max_feature)
        clf.fit(features_train, labels_train)

        predictions = clf.predict(features_test)

        accuracy = accuracy_score(labels_test, predictions)
        print 'Accuracy with ' + str(num_estimator) + ' estimators and ' \
            + str(max_feature) + ' max features: ' + str(accuracy) + '\n'
