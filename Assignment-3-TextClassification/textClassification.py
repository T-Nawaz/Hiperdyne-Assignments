from datetime import datetime
startTime = datetime.now()
import XMLReader as xr


def main():
    topic_list = []
    with open('topics.txt') as file:
        for line in file:
            topic_list.append(line.strip('\n'))

    print(topic_list)

    tokenized_training_data = xr.tokenize("Training",topic_list,200)
    tokenized_test_data = xr.tokenize("Test",topic_list,50)

    wordmap = xr.create_wordmap(tokenized_training_data)
    vector_training = xr.create_vector(tokenized_training_data,wordmap)
    vector_test = xr.create_vector(tokenized_test_data,wordmap)

    print(len(tokenized_training_data))
    print(vector_training[0][0])
    print(vector_test[0][0])

main()

print(datetime.now() - startTime)