from TrainingSetsUtil import *

def inferMsg(message, train, prior = 0.5, c = 3.7e-4):
    msgTerms = getWords(message)    
    msgProbability = 1  
    for term in msgTerms:        
        if term in train:
            msgProbability *= train[term]
        else:
            msgProbability *= c        
    return msgProbability * prior

inputMsg = raw_input('Enter the message to be classified:')
# 0.2 and 0.8 - ratio of samples for spam and ham
spamProbability = inferMsg(inputMsg, spamTrain, 0.2)
hamProbability = inferMsg(inputMsg, hamTrain, 0.8)
if spamProbability > hamProbability:
    print 'Classified as SPAM.'
else:
    print 'Classified as HAM.'