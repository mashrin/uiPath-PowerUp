import nltk
from nltk.corpus import stopwords
from nltk.tokenize import wordpunct_tokenize
from os import listdir
from os.path import isfile, join
nltk.data.path = ['nltk_data']
stopwords = set(stopwords.words('english'))

spamPath = 'data/spam/'
easyHamPath = 'data/easy_ham/'

def getWords(message): 
    allWords = set(wordpunct_tokenize(message.replace('=\\n', '').lower()))
    msgWords = [word for word in allWords if word not in stopwords and len(word) > 2]
    return msgWords
    
def getMailFromFile(file_name):
    message = ''
    with open(file_name, 'r') as inputFile:        
        for line in inputFile:
            if line == '\n':
                for line in inputFile:
                    message += line                    
    return message

    
def createTrainingSet(path):
    train = {}
    mailDir = [inputFile for inputFile in listdir(path) if isfile(join(path, inputFile))]
    cmdsCount = 0
    totalFileCount = len(mailDir)    
    for mail_name in mailDir:
        if mail_name == 'cmds':
            cmdsCount += 1
            continue
        message = getMailFromFile(path + mail_name)
        terms = getWords(message)
        for term in terms:
            if term in train:
                train[term] = train[term] + 1
            else:
                train[term] = 1
    totalFileCount -= cmdsCount
    for term in train.keys():
        train[term] = float(train[term]) / totalFileCount                            
    return train

spamTrain = createTrainingSet(spamPath)
hamTrain = createTrainingSet(easyHamPath)
