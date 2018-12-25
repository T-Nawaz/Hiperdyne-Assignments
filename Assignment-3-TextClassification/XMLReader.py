from xml.dom import minidom
import re
from datetime import datetime
startTime = datetime.now()


MAX_ROWS = 200
train_file = "Dataset/Training/Anime.xml"
test_file = "Dataset/Test/Anime.xml"

TAG_RE = re.compile(r'<[^>]+>')

def remove_tags(text):
    return TAG_RE.sub('', text)

# parse an xml file by name
mydoc = minidom.parse(train_file)

items = mydoc.getElementsByTagName('row')
print('rows->>',len(items))
index=0
wordmap = {}

count=0

for item in items:
    count=count+1
    if count==MAX_ROWS:
        break
    # print('\nitem-> ' , item)
    string = remove_tags(item.attributes['Body'].value)
    # print('string-> ' ,string)
    words = string.split(" ")
    for word in words:
        # print(word)
        if word not in wordmap:
            wordmap[word] = index
            index = index + 1

print(len(wordmap))

vector = [0]*len(wordmap)

testdoc = minidom.parse(train_file)

testItems = mydoc.getElementsByTagName('row')

for x in testItems:
    string = remove_tags(x.attributes['Body'].value)
    words = string.split(" ")
    for w in words:
        if w in wordmap.keys():
            vector[wordmap[w]] = vector[wordmap[w]] + 1
print(vector)


# string = remove_tags(testItems[0].attributes['Body'].value)
# print(string)
# words = string.split(" ")
# for w in words:
#     if w in wordmap.keys():
#         vector[wordmap[w]]=vector[wordmap[w]]+1
#         if vector[wordmap[w]]>=2:
#             print(w)
#
# print(vector)

print(datetime.now() - startTime)