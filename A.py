import main
from main import replace_accented
from sklearn import svm
from sklearn import neighbors
import nltk

# don't change the window size
window_size = 10


# A.1
def build_s(data):
    '''
    Compute the context vector for each lexelt
    :param data: dic with the following structure:
        {
			lexelt: [(instance_id, left_context, head, right_context, sense_id), ...],
			...
        }
    :return: dic s with the following structure:
        {
			lexelt: [w1,w2,w3, ...],
			...
        }


    comments RW: collect the words in the context of each lexelt

    '''
    s = {}

    # implement your code here
    for one_lexelt in data:

        my_word_context_bag = []
        for one_instance in data[one_lexelt]:
            words_left = nltk.word_tokenize(one_instance[1])[-window_size:]
            words_right = nltk.word_tokenize(one_instance[3])[0:window_size]

            my_word_context_bag += words_left + words_right
            my_word_context_set = set(my_word_context_bag)

        s[one_lexelt] = list(my_word_context_set)

    return s


# A.1
def vectorize(data, s):
    '''
    :param data: list of instances for a given lexelt with the following structure:
        {
			[(instance_id, left_context, head, right_context, sense_id), ...]
        }
    :param s: list of words (features) for a given lexelt: [w1,w2,w3, ...]
    :return: vectors: A dictionary with the following structure
            { instance_id: [w_1 count, w_2 count, ...],
            ...
            }
            labels: A dictionary with the following structure
            { instance_id : sense_id }

    '''
    vectors = {}
    labels = {}

    # implement your code here
    for one_instance in data:

        words_left = nltk.word_tokenize(one_instance[1])[-window_size:]
        words_right = nltk.word_tokenize(one_instance[3])[0:window_size]
        my_word_context_bag = words_left + words_right

        feature_vec = []
        for tmp_feature in s:
            feature_vec.append(my_word_context_bag.count(tmp_feature))

        vectors[one_instance[0]] = feature_vec
        labels[one_instance[0]] = one_instance[4]

    return vectors, labels


# A.2
def classify(X_train, X_test, y_train):
    '''
    Train two classifiers on (X_train, and y_train) then predict X_test labels

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

    :return: svm_results: a list of tuples (instance_id, label) where labels are predicted by LinearSVC
             knn_results: a list of tuples (instance_id, label) where labels are predicted by KNeighborsClassifier
    '''

    svm_results = []
    knn_results = []

    svm_clf = svm.LinearSVC()
    knn_clf = neighbors.KNeighborsClassifier()

    # implement your code here

    # prepare data for scikit learn objects
    X_train_vals = X_train.values()
    X_test_vals = X_test.values()

    y_train_vals = y_train.values()

    # create classifiers based on labeled training data
    svm_clf.fit(X_train_vals, y_train_vals)
    knn_clf.fit(X_train_vals, y_train_vals)

    # use test data set for prediction
    svm_pred_results = svm_clf.predict(X_test_vals)
    knn_pred_results = knn_clf.predict(X_test_vals)

    # Format results into list of tuples
    my_counter = 0
    for test_instance in X_test:
        svm_results.append((test_instance, svm_pred_results[my_counter]))
        knn_results.append((test_instance, knn_pred_results[my_counter]))
        my_counter += 1

    return svm_results, knn_results


# A.3, A.4 output
def print_results(results, output_file):
    '''

    :param results: A dictionary with key = lexelt and value = a list of tuples (instance_id, label)
    :param output_file: file to write output

    '''

    # implement your code here

    # don't forget to remove the accent of characters using main.replace_accented(input_str)
    # you should sort results on instance_id before printing

    all_lines = []
    for lex_item in results:

        lex_item_clean = main.replace_accented(lex_item)
        for my_tupel in results[lex_item]:

            instance_id_clean = main.replace_accented(my_tupel[0])
            sense_id = my_tupel[1]

            tmp_tupel_new = (lex_item_clean, instance_id_clean, sense_id)
            all_lines.append(tmp_tupel_new)

    # sort in two steps using stable-feature
    all_lines_sorted = sorted(all_lines, key=lambda actual_line: [actual_line[0], actual_line[1]])

    print all_lines_sorted


    # write file to disk
    fid = open(output_file, 'w')

    for item, inst, sense in all_lines_sorted:
        tmp_line = item + ' ' + inst + ' ' + sense + '\n'
        fid.write(tmp_line)

    fid.close()


# run part A
def run(train, test, language, knn_file, svm_file):
    s = build_s(train)
    svm_results = {}
    knn_results = {}
    for lexelt in s:
        X_train, y_train = vectorize(train[lexelt], s[lexelt])
        X_test, _ = vectorize(test[lexelt], s[lexelt])
        svm_results[lexelt], knn_results[lexelt] = classify(X_train, X_test, y_train)

    print_results(svm_results, svm_file)
    print_results(knn_results, knn_file)
