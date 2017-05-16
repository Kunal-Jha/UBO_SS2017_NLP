import nltk

content = []
positiveWords = []
negativeWords = []
result = []


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
        content.append((line.rstrip().split('\t')[1]))
        result.append(int(line.rstrip().split('\t')[0]))

myresult = getvalues(content)

file = open("output_baseline.txt",'w')
file.write("Training data \n")
file.write(str(sum(1 for x,y in zip(result,myresult) if x == y) / float(len(result))))
file.close()

# with open("testing.txt") as f:
#     for line in f:
#        content.append((line.rstrip().split('\t')[1]))

# getvalues(content, "Testing")
