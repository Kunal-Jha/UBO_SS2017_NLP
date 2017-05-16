import nltk
from sklearn.model_selection import train_test_split

content = []
positiveWords = []
negativeWords = []


def getvalues(content):
    myresult = []
    for sentence in content:
        wordcount = 0
        positivecount = 0
        negativecount = 0

        for word in nltk.word_tokenize(sentence):
            wordcount += 1
            if word in positiveWords:
                positivecount += 1
            elif word in negativeWords:
                negativecount += 1
        pos = positivecount / float(wordcount)
        neg = negativecount / float(wordcount)
        if pos >= neg:
            myresult.append(1)
        elif neg > pos:
            myresult.append(0)
    return myresult


with open("postive words.txt") as f:
    positiveWords = f.read().splitlines()

with open("negative words.txt") as f:
    negativeWords = f.read().splitlines()

with open("training.txt") as f:
    for line in f:
        content.append((line.rstrip().split('\t')))

train, test = train_test_split(content, test_size = 0.15)

file = open("output_baseline.txt",'w')
file.write("Training data \n")
resultTrain= []
traindata= []
for ele in train:
    resultTrain.append(int(ele[0]))
    traindata.append(ele[1])

mytrainresult = getvalues(traindata)
file.write(str(sum(1 for x,y in zip(resultTrain,mytrainresult) if x == y) / float(len(resultTrain))))

resultTest=[]
testdata=[]
for ele in train:
    resultTest.append(int(ele[0]))
    testdata.append(ele[1])

mytestresult = getvalues(testdata)
file.write("\n Testing data \n")
file.write(str(sum(1 for x,y in zip(resultTest,mytestresult) if x == y) / float(len(resultTest))))
file.close()

