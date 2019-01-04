import numpy as np
from scipy.spatial import distance

def knn(vector_training, vector_test, K, dst_measure):
    """
    :param vector_training: the entire training dataset
    :param vector_test: the entire test dataset
    :param K: number of neighbors to consider
    :param dst_measure: which distance measurement algorithm to use
    :return:
    """
    predictions = []
    for test_data in vector_test:
        neighbors, true_class = get_min_neighbors(vector_training, test_data, K, dst_measure)
        result = get_response(neighbors)
        predictions.append([result, true_class])
    acc = accuracy(predictions)
    print('Classification acc: ', repr(acc) + '%')


def get_min_neighbors(training_data, test_data, K, dst_measure):
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
    a_ne_b = a != b
    # return distance.hamming(a,b)
    return np.average(a_ne_b)

