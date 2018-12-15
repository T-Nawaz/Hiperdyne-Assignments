from statistics import mean
fileName = "data.txt"


def readAllData(fileName):
    dataTupleList = []

    with open(fileName, "r") as fp:
        for line in fp.readlines():
            # print(line)
            line = line[:len(line) - 1]
            tmp = line.split("\t")
            # print(tmp[0])
            # print(tmp[1])
            dataTupleList.append((tmp[0], tmp[1]))
    dataTupleList.pop(0)
    return dataTupleList


def computeAverageForClasses(dataList):
    dictMean = {}
    classList = set()

    for sample in dataList:
        classList.add(sample[1])

    for className in classList:
        classTotal = []
        for sample in dataList:
            if (sample[1] == className):
                classTotal.append(float(sample[0]))
        classMean = mean(classTotal)
        dictMean.update({className:classMean})
    return dictMean,classList

def countMisclassified(dataList,dictMean,classList):
    newFile = open("Misclassified2.txt", "w")
    misclassified = 0

    for sample in dataList:
        error  = False

        for className in classList:
            if ( abs(float(sample[0]) - dictMean[sample[1]]) > abs(float(sample[0]) - dictMean[className]) ):
                misclassified += 1
                error = True
        if error:
            line = str(sample) + "\n"
            newFile.write(line)
    newFile.close()
    print("Total misclassified", misclassified)



dataList = readAllData(fileName)
dictMean,classList = computeAverageForClasses(dataList)
print(dictMean)
countMisclassified(dataList,dictMean,classList)
