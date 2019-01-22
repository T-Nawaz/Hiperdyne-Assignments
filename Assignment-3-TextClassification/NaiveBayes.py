import numpy as np
import math


def sigmoid(x):
    return 1 / (1 + math.exp(-x))


def naive_bayes(vector_training, vector_test, topic_list, V):
    num_topics = len(topic_list)
    num_class_data = int(np.floor(len(vector_training) / num_topics))

    vector_class = []
    for x in range(num_topics):
        class_name = vector_training[x * num_class_data][1]
        summation = np.sum(vector_training[x * num_class_data: ((x + 1) * num_class_data - 1)], axis=0)
        # print(x*num_class_data ,'->', (x+1)*num_class_data - 1,'->',class_name)
        vector_class.append(summation[0])

    total_word_count = 0
    for x in range(num_topics):
        total_word_count = total_word_count + np.sum(vector_class[x])

    total_word_in_class = np.array(vector_class).sum(axis=0)
    print(total_word_in_class)

    alpha = 1
    # for alpha in np.arange(0.02, 1.02, 1 / 50):
    if alpha == 1:
        print('Alpha-> ', alpha)
        predictions = []
        x = 0
        for test_doc in vector_test:
            x += 1
            probability_list = []

            for class_index in range(num_topics):
                # print('class-> ', topic_list[class_index])
                # total_word_in_class = np.sum(vector_class[class_index])
                class_prior = math.log(1 / num_topics)

                # print('class_prior -> ',class_prior)
                word_prob = 1.0

                for word in range(len(vector_class[class_index])):

                    # print(test_doc[0][word])
                    if test_doc[0][word] > 0:
                        numerator = vector_class[class_index][word] + alpha
                    else:
                        numerator = alpha
                    denominator = (total_word_in_class[word] + alpha * V)

                    # print('numerator-> ', numerator, 'denominator-> ', denominator)
                    #
                    word_prob = (word_prob + math.log(numerator / denominator))
                    # print(posterior)

                # print('posterior -> ', posterior)

                probability = class_prior + word_prob

                probability_list.append([probability])
            if x % 50 == 0:
                print(probability_list)
            predictions.append([topic_list[probability_list.index(max(probability_list))], test_doc[1]])
            # format -> (predicted, real)
            # print(probability_list)
            # print(probability_list.index(max(probability_list)))
            # print(topic_list[probability_list.index(max(probability_list))])

        print(predictions)
        acc = accuracy(predictions)
        print('Classification acc: ', repr(acc) + '%')


def accuracy(predictions):
    correct = 0
    for x in range(len(predictions)):
        if predictions[x][0] == predictions[x][1]:
            correct += 1
    return (correct / float(len(predictions))) * 100
