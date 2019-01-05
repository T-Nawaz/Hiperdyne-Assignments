import numpy as np
from numpy.linalg import norm
# from scipy.spatial import distance

def knn(vector_training, vector_test, K, dst_measure):
    """
    :param vector_training: the entire training dataset
    :param vector_test: the entire test dataset
    :param K: number of neighbors to consider
    :param dst_measure: which distance measurement algorithm to use
    :return: nothing. prints accuracy directly
    """
    predictions = []

    if dst_measure == 3:
        tf_idf_vector_train = tf_idf(tf_vector=tf(vector_training), idf_vector=idf(vector_training))
        tf_idf_vector_test = tf_idf(tf_vector=tf(vector_test), idf_vector=idf(vector_test))

        for test_data in tf_idf_vector_test:

            neighbors, true_class = get_max_distance_neighbors(tf_idf_vector_train, test_data, K)
            result = get_response(neighbors)
            predictions.append([result, true_class])

        acc = accuracy(predictions)
        print('Classification acc: ', repr(acc) + '%')
    else:
        for test_data in vector_test:
            neighbors, true_class = get_min_distance_neighbors(vector_training, test_data, K, dst_measure)
            result = get_response(neighbors)
            predictions.append([result, true_class])
        acc = accuracy(predictions)
        print('Classification acc: ', repr(acc) + '%')


def get_max_distance_neighbors(training_data, test_data, K):
    ""
    distance_list = []

    for line in training_data:
        distance = cosine_distance(line[0], test_data[0])
        # format: distance, class
        distance_list.append([distance, line[1]])

    true_class = test_data[1]
    distance_list = sorted(distance_list,reverse=True)

    neighbors = []
    for x in range(K):
        neighbors.append(distance_list[x])
    return neighbors, true_class

def get_min_distance_neighbors(training_data, test_data, K, dst_measure):
    """
    :param training_data: the entire training dataset
    :param test_data: one selected test data
    :param K: number of neighbors to consider
    :param dst_measure: which distance measurement algorithm to use
    :return:
    """
    distance_list = []
    for line in training_data:
        if dst_measure == 1:
            distance = euclidian_distance(line[0], test_data[0])
        if dst_measure == 2:
            distance = hamming_distance(line[0], test_data[0])
        # format: distance, class
        distance_list.append([distance, line[1]])

    true_class = test_data[1]
    distance_list = sorted(distance_list)
    # print(distance_list)

    neighbors = []
    for x in range(K):
        neighbors.append(distance_list[x])
    return neighbors, true_class


def get_response(neighbors):
    votes = {}
    for x in range(len(neighbors)):
        response = neighbors[x][1]
        if response in votes:
            votes[response] += 1
        else:
            votes[response] = 1
    votes = sorted(votes.items())
    return votes[0][0]


def accuracy(predictions):
    correct = 0
    for x in range(len(predictions)):
        if predictions[x][0] == predictions[x][1]:
            correct += 1
    return (correct / float(len(predictions))) * 100


def euclidian_distance(a, b):
    return np.sqrt(np.sum((a - b) ** 2))

def hamming_distance(a,b):
    # a = np.array(a);  b = np.array(b)
    a_ni_b = a != b
    # return distance.hamming(a,b)
    return np.average(a_ni_b)


def cosine_distance(a,b):

    return np.dot(a,b)/(norm(a) * norm(b))

def tf(vector):
    """
    TF or Term Frequency is the ratio of number of times the word appears in a document compared to the total number of words in that document.
    """
    for line in vector:
        line[0] = np.true_divide(line[0], np.sum(line[0]))
    return vector

def idf(vector):
    import math
    """
    Inverse Data Frequency or idf is used to calculate the weight of rare words across all documents in the corpus.
    The words that occur rarely in the corpus have a high IDF score.
    """
    N = len(vector)
    words = len(vector[0])

    for x in range(words):
        df=0
        for line in vector:
            if line[0][x]>0:
                df += 1
        for line in vector:
            if df == 0:
                df = 0.01
            line[0][x] = math.log2(N/ float(df))
    return vector


def tf_idf(tf_vector, idf_vector):
    for x in range(len(tf_vector)):
        tf_vector[x][0] = tf_vector[x][0] * idf_vector[x][0]
    return tf_vector