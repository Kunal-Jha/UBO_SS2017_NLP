import nltk
from sklearn.model_selection import train_test_split
from nltk.classify.util import apply_features,accuracy

content = []
positiveWords = []
negativeWords = []
with open("postive words.txt") as f:
    positiveWords = f.read().splitlines()

with open("negative words.txt") as f:
    negativeWords = f.read().splitlines()

def get_word_features(tweets):
        all_words = []
        for (words, sentiment) in tweets:
          all_words.extend(words)
        wordlist = nltk.FreqDist(all_words)
        word_features = []
        for feature in wordlist:
            if wordlist[feature] > 10000:
                word_features.append(feature)
        print(len(word_features))
        return word_features

def extract_features(document):
        document_words = set(document)
        features = {}
        for word in word_features:
            features['contains(%s)' % word] = (word in document_words)
        return features



def getData(train):
    positives, negatives, traindata = [], [], []
    for ele in train:
        if int(ele[0]) == 0:
            negatives.append((ele[1],'neg'))
        elif int(ele[0]) == 1:
            positives.append((ele[1],'pos'))

    for (words, sentiment) in positives + negatives:
        words_filtered = [e.lower() for e in words.split() if len(e) >= 3]
        traindata.append((words_filtered, sentiment))

    return traindata

with open("training.txt") as f:
    for line in f:
        content.append((line.rstrip().split('\t')))

train, test = train_test_split(content, test_size = 0.15)

traindata= getData(train)
testdata = getData(test)

word_features = get_word_features(traindata+testdata)

print("Word Features Generated")

training_set = nltk.classify.apply_features(extract_features, traindata)
test_set = nltk.classify.apply_features(extract_features, testdata)
print("Training and Test Sets Created")

classifier = nltk.NaiveBayesClassifier.train(training_set)
print("Model generated")
accuracy = nltk.classify.accuracy(classifier,test_set)

print(accuracy)
