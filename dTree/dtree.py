import csv
import random
import numpy as np
from math import log
from helpers import peek
from sklearn.metrics import confusion_matrix

MAX_DEPTH = 50
PRUNING_THRESHOLD = 200


def get_data_set():
    with open('car.data') as file:
        "Read the file and store it in a list"
        data_set = list(csv.reader(file))
        # for row in data_set: print(row)

        random.shuffle(data_set)  # ;print("new");    peek(data_set)
        random.shuffle(data_set)  # ;print("new");    peek(data_set)



        train_test_split = int(len(data_set) * .8)  # ;print('train_test_split: ', train_test_split)
        train_data_set = np.array(data_set[:train_test_split])
        test_data_set = np.array(data_set[train_test_split + 1:])

        # print('train_data_set: ', len(train_data_set)); print('test_data_set: ', len(test_data_set))
        # peek(train_data_set);peek(test_data_set)
        return train_data_set, test_data_set


def get_data_count(data):
    "Return the unique values and their frequencies as a dictionary"
    data_values, data_frequency = np.unique(data, return_counts=True)
    return dict(zip(data_values, data_frequency))


def build_tree(train_data_set, attributes):
    root = choose_best_attribute(train_data_set, attributes)
    # print('root-> ',root)
    ID3(root, train_data_set, attributes, 1)  # here depth = 1
    return root


def make_terminal(examples):
    outcomes = [row[-1] for row in examples]
    return max(set(outcomes), key=outcomes.count)


def ID3(node, train_data_set, attributes, depth):
    # print('depth-> ', depth)
    best_attribute = node['attr']

    # print(best_attribute)

    '''Deleted each used attribute'''
    # print("Attributes in: ", attributes)
    i = np.where(attributes == best_attribute)
    attributes = np.delete(attributes, i)
    print("Attributes out: ", attributes)

    'y checks if the last/class col has all same values'
    y = np.all(train_data_set == train_data_set[0, :], axis=0)
    # print(y[-1])

    branches = list(node['branch'].keys())
    for branch in branches:
        sub_data_set = []
        for row in train_data_set:
            if row[best_attribute] == branch:
                sub_data_set.append(row)
        sub_data_set = np.array(sub_data_set)

        'IF all class same'
        if y[-1]:
            node['branch'][branch] = make_terminal(train_data_set)
            return

        'IF empty attribute or exceed max depth'
        if attributes.size == 0 or depth > MAX_DEPTH:
            node['branch'][branch] = make_terminal(sub_data_set)
            # print("->", node['branch'][branch])
            return

        if len(sub_data_set) <= PRUNING_THRESHOLD:
            node['branch'][branch] = make_terminal(sub_data_set)
        else:
            node['branch'][branch] = choose_best_attribute(sub_data_set, attributes)
            ID3(node['branch'][branch], sub_data_set, attributes, depth + 1)
    # print(node)
    # print(node['branch'])
    # node['branch'] = make_terminal(train_data_set)
    # print(node)
    # print(node['branch'])


def choose_best_attribute(train_data_set, attributes):
    max_gain = best_attribute = -1
    branch_dict = {}
    for A in attributes:
        # attribute_values = train_data_set[:, A]
        # unique_values = get_data_count(train_data_set[:, -1])
        # unique_values_keys = list(unique_values.keys())
        gain = information_gain(train_data_set, A)
        if gain > max_gain:
            max_gain = gain
            best_attribute = A
            "Get only the names of the unique values"
            unique_values_keys = get_data_count(train_data_set[:, A]).keys()

            "Create and empty dictionary whose elements wil become the next branch or leaf"
            branch_dict = dict.fromkeys(unique_values_keys)

        '''Debugging'''
        # print('Testing attr: ', A , ' attr vals: ', attribute_values)
        # print('unique: ', unique_values)
        # print('keys: ',unique_values_keys)
    print('Feature-> ', best_attribute, ' gain= ', max_gain)
    # print(branch_dict)
    return {'attr': best_attribute, 'branch': branch_dict}


def information_gain(train_data_set, A):
    parent_entropy = calculate_entropy(train_data_set)

    unique_values = get_data_count(train_data_set[:, A])
    unique_values_keys = list(unique_values.keys())

    child_entropy = 0.0
    for key in unique_values_keys:
        sub_data_set = []

        for example in train_data_set:
            if example[A] == key:
                sub_data_set.append(example)

        sub_data_set = np.array(sub_data_set)
        # print(sub_data_set)
        child_entropy = child_entropy - (len(sub_data_set) / len(train_data_set)) * calculate_entropy(sub_data_set)
    '''Return Gain'''
    return parent_entropy + child_entropy


def calculate_entropy(data_set):
    unique_values = get_data_count(data_set[:, -1])
    unique_values_keys = list(unique_values.keys())
    total_data = len(data_set)

    '''Debugging'''
    # print(unique_values);print(unique_values_keys);print(total_data)

    entropy = 0.0
    for i in unique_values_keys:
        p = float(unique_values[i] / total_data)
        entropy = entropy - p * log(p, 2)
        # print('val: ', unique_values[i], ' p: ',p, 'entropy: ', entropy)

    return entropy


def evaluation(predictions, true_classes):
    counter = 0
    for x in range(len(predictions)):
        if predictions[x] == true_classes[x]:
            counter += 1
    return (counter / len(predictions)) * 100


def get_predictions(node, row):
    for x in list(node['branch'].keys()):
        if row[node['attr']] == x:
            # print('attr match. example: ', row[node['attr']], ' tree: ', x)
            if isinstance(node['branch'][x], dict):
                # print('more dict')
                return get_predictions(node['branch'][x], row)
            else:
                # print('found ans: ', node['branch'][x])
                return node['branch'][x]


def main():
    train_data_set, test_data_set = get_data_set()
    attributes = np.arange(0, len(train_data_set[0]) - 1, 1)
    print(len(train_data_set[0]))
    print(attributes)
    # peek(train_data_set[:, len(test_data_set[0])-1]);    print(type(test_data_set))

    '''Small test cases
    Remove before submission'''
    # with open('car.data') as file:
    #     train_data_set = np.array(list(csv.reader(file)))
    # attributes = np.arange(0, len(train_data_set[0]) - 1, 1)
    # print(len(train_data_set))
    # uniques = (get_data_count(train_data_set[:, - 1]))
    # print(uniques)
    # calculate_entropy(train_data_set)
    # for x in train_data_set:
    #     print(x[0])
    # print(information_gain(train_data_set,1))
    # build_tree(train_data_set, attributes)
    # y = np.all(train_data_set == train_data_set[0,:], axis=0)
    # print(y[-1])
    '''End of test case'''

    root = build_tree(train_data_set, attributes)
    print(root)
    predictions = []
    true_classes = []

    for row in test_data_set:
        predicted = get_predictions(root, row)
        predictions.append(predicted)
        true_classes.append(row[-1])
        # print('real class: ', row[-1])
    mean_accuracy = evaluation(predictions, true_classes)
    print('\n\tMean acc = ', mean_accuracy)

    '''Precision and Recall'''
    # print(predictions.__sizeof__())
    # print(true_classes.__sizeof__())
    # confusion_Matrix = confusion_matrix(true_classes, predictions)
    # print(confusion_Matrix)
    #
    # True_Positive = np.diag(confusion_Matrix)
    # False_Positive = np.sum(confusion_Matrix, axis=0) - True_Positive
    # False_Negative = np.sum(confusion_Matrix, axis=1) - True_Positive
    #
    # print(True_Positive.dtype)
    # print(False_Positive.dtype)
    # print(False_Negative.dtype)
    #
    # precision = True_Positive / (True_Positive + False_Positive)
    # recall = True_Positive / (True_Positive + False_Negative)


main()
