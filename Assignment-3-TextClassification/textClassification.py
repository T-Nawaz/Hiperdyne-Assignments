from datetime import datetime
startTime = datetime.now()
import XMLReader as xr
import KNN as knn


def main():
    topic_list = []
    with open('topics.txt') as file:
        for line in file:
            topic_list.append(line.strip('\n'))

    print(topic_list)

    tokenized_training_data = xr.tokenize("Training", topic_list, 200)
    tokenized_test_data = xr.tokenize("Test", topic_list, 50)

    wordmap = xr.create_wordmap(tokenized_training_data)
    vector_training = xr.create_vector(tokenized_training_data, wordmap)
    vector_test = xr.create_vector(tokenized_test_data, wordmap)

    for dst_measure in range(1,4):
        print()
        if dst_measure == 1:
            print("Euclidian distance")
        if dst_measure == 2:
            print("Hamming distance")
        if dst_measure == 3:
            print("Cosine Similarity")

        for K in range(1, 6, 2):
            print('For K = ', K)
            knn.knn(vector_training, vector_test, K, dst_measure)

    # test = knn.idf(vector_test)
    # print(test[0])

main()

print(datetime.now() - startTime)
