import A
from sklearn.feature_extraction import DictVectorizer

import nltk
from sklearn import svm

# You might change the window size
window_size = 10


# B.1.a,b,c,d
def extract_features(data, language):
    """"

    takes the instances of ONE lexelt and creates the feature space (with feature values)


    :param language:
    :param data: list of instances for a given lexelt with the following structure:
        {
			[(instance_id, left_context, head, right_context, sense_id), ...]
        }
    :return: features: A dictionary with the following structure
             { instance_id: {f1:count, f2:count,...}
            ...
            }
            labels: A dictionary with the following structure
            { instance_id : sense_id }
    """

    features = {}
    labels = {}

    # implement your code here

    my_word_context_list = []
    for (instance_id, left, head, right, sense_id) in data:
        words_left = nltk.word_tokenize(left)[-window_size:]
        words_right = nltk.word_tokenize(right)[:window_size]
        my_word_context_list += words_left + words_right

    my_word_context_set = list(set(my_word_context_list))

    # prepare features for each instance
    for (instance_id, left, head, right, sense_id) in data:

        wd_context = left + right
        for word in my_word_context_set:
            if instance_id not in features:
                features[instance_id] = {}
            features[instance_id][word] = wd_context.count(word)

    # prepare labels for output

    for instance in data:
        labels[instance[0]] = instance[4]
        print instance[4]


    #for (instance_id, e2, e3, e4, sense_id) in data:
    #    labels[instance_id] = sense_id
    #    print instance_id, sense_id

    return features, labels


# implemented for you
def vectorize(train_features, test_features):
    '''
    convert set of features to vector representation
    :param train_features: A dictionary with the following structure
             { instance_id: {f1:count, f2:count,...}
            ...
            }
    :param test_features: A dictionary with the following structure
             { instance_id: {f1:count, f2:count,...}
            ...
            }
    :return: X_train: A dictionary with the following structure
             { instance_id: [f1_count,f2_count, ...]}
            ...
            }
            X_test: A dictionary with the following structure
             { instance_id: [f1_count,f2_count, ...]}
            ...
            }
    '''
    X_train = {}
    X_test = {}

    vec = DictVectorizer()
    vec.fit(train_features.values())
    for instance_id in train_features:
        X_train[instance_id] = vec.transform(train_features[instance_id]).toarray()[0]

    for instance_id in test_features:
        X_test[instance_id] = vec.transform(test_features[instance_id]).toarray()[0]

    return X_train, X_test


# B.1.e
def feature_selection(X_train, X_test, y_train):
    '''
    Try to select best features using good feature selection methods (chi-square or PMI)
    or simply you can return train, test if you want to select all features
    :param X_train: A dictionary with the following structure
             { instance_id: [f1_count,f2_count, ...]}
            ...
            }
    :param X_test: A dictionary with the following structure
             { instance_id: [f1_count,f2_count, ...]}
            ...
            }
    :param y_train: A dictionary with the following structure
            { instance_id : sense_id }
    :return:
    '''

    # implement your code here

    # return X_train_new, X_test_new
    # or return all feature (no feature selection):
    return X_train, X_test


# B.2
def classify(X_train, X_test, y_train):
    '''
    Train the best classifier on (X_train, and y_train) then predict X_test labels

    :param X_train: A dictionary with the following structure
            { instance_id: [w_1 count, w_2 count, ...],
            ...
            }

    :param X_test: A dictionary with the following structure
            { instance_id: [w_1 count, w_2 count, ...],
            ...
            }

    :param y_train: A dictionary with the following structure
            { instance_id : sense_id }

    :return: results: a list of tuples (instance_id, label) where labels are predicted by the best classifier
    '''

    results = []

    # implement your code here

    svm_clf = svm.LinearSVC()

    X_train_vals = X_train.values()
    X_test_vals = X_test.values()

    y_train_vals = y_train.values()



    # train the classifier
    svm_clf.fit(X_train_vals, y_train_vals)

    # classify the test (dev) data
    svm_pred_results = svm_clf.predict(X_test_vals)

    # reformat results into list
    my_counter = 0
    for test_instance in X_test:
        results.append((test_instance, svm_pred_results[my_counter]))
        my_counter += 1

    return results


# run part B
def run(train, test, language, answer):
    results = {}

    for lexelt in train:
        train_features, y_train = extract_features(train[lexelt], language)
        test_features, _ = extract_features(test[lexelt], language)

        X_train, X_test = vectorize(train_features, test_features)
        X_train_new, X_test_new = feature_selection(X_train, X_test, y_train)
        results[lexelt] = classify(X_train_new, X_test_new, y_train)

    A.print_results(results, answer)
