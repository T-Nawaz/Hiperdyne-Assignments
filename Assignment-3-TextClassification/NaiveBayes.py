import numpy as np
import math

def sigmoid(x):
  return 1 / (1 + math.exp(-x))

def naive_bayes(vector_training,vector_test,topic_list,V):
    # a = np.array([1,2,3,4])
    # b = np.array([1,2,3,4])
    # xs = []
    # xs.append([a,"yo"])
    # xs.append([b,"yo"])
    #
    # xs.append([a,"yo"])
    # xs.append([b,"yo"])
    #
    # print(xs)
    # s = np.sum(xs[:4],axis=0)
    # print(s)
    # print(s[0])
    num_topics = len(topic_list)
    num_class_data = int(np.floor(len(vector_training) / num_topics))
    print(num_class_data)

    vector_class = []
    for x in range(num_topics):
        class_name = vector_training[x*num_class_data][1]
        sum = np.sum(vector_training[x*num_class_data : ((x+1)*num_class_data - 1)] , axis=0)
        # print(x*num_class_data ,'->', (x+1)*num_class_data - 1,'->',class_name)
        vector_class.append([sum[0], class_name])

    # print(vector_class)
    # print(np.sum(vector_class[0][0]))
    # print(np.sum(vector_class[1][0]))
    # print(np.sum(vector_class[2][0]))
    # print(V)

    total_word_count = 0
    for x in range(num_topics):
        total_word_count = total_word_count + np.sum(vector_class[x][0])
    # print(total_word_count)

    # for x in np.arange(0.02,1.02, 1/50):
    #     print(x)

    alpha = 1
    for alpha in np.arange(0.02, 1.02, 1 / 50):
    # if alpha == 1:
        print('Alpha-> ', alpha)
        predictions = []
        for test_doc in vector_test:

            probability_list = []

            for class_index in range(num_topics):
                # print('class-> ', topic_list[class_index])
                total_word_in_class = np.sum(vector_class[class_index][0])
                class_prior = (total_word_in_class / total_word_count)

                # print('class_prior -> ',class_prior)
                word_prob = 1.0

                for word in range(len(vector_class[class_index][0])):

                    # print(test_doc[0][word])
                    numerator = (test_doc[0][word] + alpha)
                    denominator = (total_word_in_class + alpha * V)

                    # print('numerator-> ', numerator, 'denominator-> ', denominator)
                    #
                    word_prob = (word_prob * (numerator/denominator))
                    # print(posterior)

                # print('posterior -> ', posterior)

                probability = class_prior * word_prob
                probability_list.append([probability])

            predictions.append([topic_list[probability_list.index(max(probability_list))] , test_doc[1]])



            # print(probability_list)
            # print(probability_list.index(max(probability_list)))
            # print(topic_list[probability_list.index(max(probability_list))])

            # print(predictions)

        acc = accuracy(predictions)
        print('Classification acc: ', repr(acc) + '%')

def accuracy(predictions):
    correct = 0
    for x in range(len(predictions)):
        if predictions[x][0] == predictions[x][1]:
            correct += 1
    return (correct / float(len(predictions))) * 100


















