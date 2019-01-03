from xml.dom import minidom
import re
import nltk
from nltk.corpus import stopwords

custom_stops = []
with open('stopwords.txt') as file:
    for line in file:
        custom_stops.append(line.strip('\n'))


def remove_tags(text):
    tag_re = re.compile(r'<[^>]+>')
    return tag_re.sub('', text)


def tokenize(data_set, topic_list, MAX_ROWS):
    """

    :param data_set: a String variable. Points to the Training or Test folder
    :param topic_list: a list of topics to be used
    :param MAX_ROWS: maximum number of rows of the XML file to be considered
    :return: a list of tokenized words and labels
    """
    tokenized_word_list = []
    stops = set(stopwords.words('english'))

    for topic in topic_list:

        file = "Dataset/" + data_set + "/" + topic + ".xml"
        mydoc = minidom.parse(file)
        rows = mydoc.getElementsByTagName('row')

        count = 0
        for row in rows:

            if count == MAX_ROWS:
                break

            string = remove_tags(row.attributes['Body'].value).lower()
            string = re.sub('|\n|\t|\?', '', string)
            string = re.sub('--|/', ' ', string)
            string = re.sub(r'http\S+|www\S+|\'|\(|\)', '', string)
            #handle empty string
            if not string:
                continue

            words = string.split(" ")

            tokenized_line = []
            for word in words:
                if word != '' not in stops:
                    if len(word) < 12 :
                        if word not in custom_stops:
                            tokenized_line.append(word.strip(',').strip('.'))

            tokenized_word_list.append([tokenized_line,topic])
            count += 1

    return tokenized_word_list

def create_wordmap(tokenized_word_list):
    index = 0
    wordmap = {}
    for line in tokenized_word_list:
        for word in line[0]:
            print(word)
            if word not in wordmap:
                wordmap[word] = index
                index += 1
    return wordmap