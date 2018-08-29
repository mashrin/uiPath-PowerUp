import pandas as pd
import numpy as np
import tensorflow as tf
import re
from nltk.corpus import stopwords
import time
from tensorflow.python.layers.core import Dense
print('TensorFlow Version: {}'.format(tf.__version__))

msg = pd.read_csv("Reviews.csv")

msg = msg.dropna()
msg = msg.drop(['Id','ProductId','UserId','ProfileName','HelpfulnessNumerator','HelpfulnessDenominator',
                        'Score','Time'], 1)
msg = msg.reset_index(drop=True)
msg.head()
for i in range(5):
    print("Review #",i+1)
    print(msg.Summary[i])
    print(msg.Text[i])
    print()
contractions = { 
"ain't": "am not",
"aren't": "are not",
"can't": "cannot",
"can't've": "cannot have",
"'cause": "because",
"could've": "could have",
"couldn't": "could not",
"couldn't've": "could not have",
"didn't": "did not",
"doesn't": "does not",
"don't": "do not",
"hadn't": "had not",
"hadn't've": "had not have",
"hasn't": "has not",
"haven't": "have not",
"he'd": "he would",
"he'd've": "he would have",
"he'll": "he will",
"he's": "he is",
"how'd": "how did",
"how'll": "how will",
"how's": "how is",
"i'd": "i would",
"i'll": "i will",
"i'm": "i am",
"i've": "i have",
"isn't": "is not",
"it'd": "it would",
"it'll": "it will",
"it's": "it is",
"let's": "let us",
"ma'am": "madam",
"mayn't": "may not",
"might've": "might have",
"mightn't": "might not",
"must've": "must have",
"mustn't": "must not",
"needn't": "need not",
"oughtn't": "ought not",
"shan't": "shall not",
"sha'n't": "shall not",
"she'd": "she would",
"she'll": "she will",
"she's": "she is",
"should've": "should have",
"shouldn't": "should not",
"that'd": "that would",
"that's": "that is",
"there'd": "there had",
"there's": "there is",
"they'd": "they would",
"they'll": "they will",
"they're": "they are",
"they've": "they have",
"wasn't": "was not",
"we'd": "we would",
"we'll": "we will",
"we're": "we are",
"we've": "we have",
"weren't": "were not",
"what'll": "what will",
"what're": "what are",
"what's": "what is",
"what've": "what have",
"where'd": "where did",
"where's": "where is",
"who'll": "who will",
"who's": "who is",
"won't": "will not",
"wouldn't": "would not",
"you'd": "you would",
"you'll": "you will",
"you're": "you are"
}
def cleanText(text, removeStopwords = True):
    text = text.lower()
    if True:
        text = text.split()
        newText = []
        for w in text:
            if w in contractions:
                newText.append(contractions[w])
            else:
                newText.append(w)
        text = " ".join(newText)
    text = re.sub(r'https?:\/\/.*[\r\n]*', '', text, flags=re.MULTILINE)
    text = re.sub(r'\<a href', ' ', text)
    text = re.sub(r'&amp;', '', text) 
    text = re.sub(r'[_"\-;%()|+&=*%.,!?:#$@\[\]/]', ' ', text)
    text = re.sub(r'<br />', ' ', text)
    text = re.sub(r'\'', ' ', text)
    if removeStopwords:
        text = text.split()
        stops = set(stopwords.words("english"))
        text = [w for w in text if not w in stops]
        text = " ".join(text)
    return text
import nltk
nltk.download('stopwords')
cleanSummaries = []
for summ in msg.Summary:
    cleanSummaries.append(cleanText(summ, removeStopwords=False))
print("Summaries are complete.")

cleanTexts = []
for text in msg.Text:
    cleanTexts.append(cleanText(text))
print("Texts are complete.")
for i in range(5):
    print("Clean Review #",i+1)
    print(cleanSummaries[i])
    print(cleanTexts[i])
    print()

def countWords(count_dict, text):
    '''Count the number of occurrences of each word in a set of text'''
    for sentence in text:
        for word in sentence.split():
            if word not in count_dict:
                count_dict[word] = 1
            else:
                count_dict[word] += 1

wordCounts = {}

countWords(wordCounts, cleanSummaries)
countWords(wordCounts, cleanTexts)
            
print("Size of Vocabulary:", len(wordCounts))

embeddingsIndex = {}
with open('/home/mashrin/Documents/Text-Summarization-with-Amazon-Reviews-master/numberbatch-en-17.02.txt', encoding='utf-8') as f:
    for line in f:
        values = line.split(' ')
        word = values[0]
        embedding = np.asarray(values[1:], dtype='float32')
        embeddingsIndex[word] = embedding

print('Word embeddings:', len(embeddingsIndex))

missingWords = 0
threshold = 20

for word, count in wordCounts.items():
    if count > threshold:
        if word not in embeddingsIndex:
            missingWords += 1
            
missingRatio = round(missingWords/len(wordCounts),4)*100
            
print("Number of words missing from CN:", missingWords)
print("Percent of words that are missing from vocabulary: {}%".format(missingRatio))

vocabToInt = {} 

value = 0
for word, count in wordCounts.items():
    if count >= threshold or word in embeddingsIndex:
        vocabToInt[word] = value
        value += 1
codes = ["<UNK>","<PAD>","<EOS>","<GO>"]   
for code in codes:
    vocabToInt[code] = len(vocabToInt)
intToVocab = {}
for word, value in vocabToInt.items():
    intToVocab[value] = word

usageRatio = round(len(vocabToInt) / len(wordCounts),4)*100

print("Total number of unique words:", len(wordCounts))
print("Number of words we will use:", len(vocabToInt))
print("Percent of words we will use: {}%".format(usageRatio))
dimension = 300
nbWords = len(vocabToInt)
weMatrix = np.zeros((nbWords, dimension), dtype=np.float32)
for word, i in vocabToInt.items():
    if word in embeddingsIndex:
        weMatrix[i] = embeddingsIndex[word]
    else:
        newEmb = np.array(np.random.uniform(-1.0, 1.0, dimension))
        embeddingsIndex[word] = newEmb
        weMatrix[i] = newEmb
print(len(weMatrix))

def integerConversion(text, countWord, countUnk, eos=False):

    ints = []
    for sentence in text:
        integerSentence = []
        for word in sentence.split():
            countWord += 1
            if word in vocabToInt:
                integerSentence.append(vocabToInt[word])
            else:
                integerSentence.append(vocabToInt["<UNK>"])
                countUnk += 1
        if eos:
            integerSentence.append(vocabToInt["<EOS>"])
        ints.append(integerSentence)
    return ints, countWord, countUnk
countWord = 0
countUnk = 0

summInt, countWord, countUnk = integerConversion(cleanSummaries, countWord, countUnk)
textInt, countWord, countUnk = integerConversion(cleanTexts, countWord, countUnk, eos=True)

unk_percent = round(countUnk/countWord,4)*100

print("Total number of words in headlines:", countWord)
print("Total number of UNKs in headlines:", countUnk)
print("Percent of words that are UNK: {}%".format(unk_percent))

def lengthCreate(text):
    '''Create a data frame of the sentence lengths from a text'''
    lengths = []
    for sentence in text:
        lengths.append(len(sentence))
    return pd.DataFrame(lengths, columns=['counts'])
lenSummay = lengthCreate(summInt)
lenText = lengthCreate(textInt)

print("Summaries:")
print(lenSummay.describe())
print()
print("Texts:")
print(lenText.describe())
print(np.percentile(lenText.counts, 90))
print(np.percentile(lenText.counts, 95))
print(np.percentile(lenText.counts, 99))
print(np.percentile(lenSummay.counts, 90))
print(np.percentile(lenSummay.counts, 95))
print(np.percentile(lenSummay.counts, 99))
def counterUnk(sentence):
    '''Counts the number of time UNK appears in a sentence.'''
    countUnk = 0
    for word in sentence:
        if word == vocabToInt["<UNK>"]:
            countUnk += 1
    return countUnk
summarySort = []
sortText = []
maxLenText = 84
maxSummText = 13
minLen = 2
limUnkText = 1
limUnkSumm = 0

for length in range(min(lenText.counts), maxLenText): 
    for count, words in enumerate(summInt):
        if (len(summInt[count]) >= minLen and
            len(summInt[count]) <= maxSummText and
            len(textInt[count]) >= minLen and
            counterUnk(summInt[count]) <= limUnkSumm and
            counterUnk(textInt[count]) <= limUnkText and
            length == len(textInt[count])
           ):
            summarySort.append(summInt[count])
            sortText.append(textInt[count])

print(len(summarySort))
print(len(sortText))

def inpModel():
    
    inData = tf.placeholder(tf.int32, [None, None], name='input')
    targets = tf.placeholder(tf.int32, [None, None], name='targets')
    lr = tf.placeholder(tf.float32, name='learning_rate')
    probKeep = tf.placeholder(tf.float32, name='probKeep')
    lenSumm = tf.placeholder(tf.int32, (None,), name='lenSumm')
    maxSummText = tf.reduce_max(lenSumm, name='max_dec_len')
    lenText = tf.placeholder(tf.int32, (None,), name='lenText')

    return inData, targets, lr, probKeep, lenSumm, maxSummText, lenText

def encInpProcess(target_data, vocabToInt, bSize):
    ending = tf.strided_slice(target_data, [0, 0], [bSize, -1], [1, 1])
    dec_input = tf.concat([tf.fill([bSize, 1], vocabToInt['<GO>']), ending], 1)

    return dec_input

def encLayer(sizeRnn, sequence_length, layerNum, rnn_inputs, probKeep):
    '''Create the encoding layer'''
    
    for layer in range(layerNum):
        with tf.variable_scope('encoder_{}'.format(layer)):
            cell_fw = tf.contrib.rnn.LSTMCell(sizeRnn,
                                              initializer=tf.random_uniform_initializer(-0.1, 0.1, seed=2))
            cell_fw = tf.contrib.rnn.DropoutWrapper(cell_fw, 
                                                    input_probKeep = probKeep)

            cell_bw = tf.contrib.rnn.LSTMCell(sizeRnn,
                                              initializer=tf.random_uniform_initializer(-0.1, 0.1, seed=2))
            cell_bw = tf.contrib.rnn.DropoutWrapper(cell_bw, 
                                                    input_probKeep = probKeep)

            enc_output, enc_state = tf.nn.bidirectional_dynamic_rnn(cell_fw, 
                                                                    cell_bw, 
                                                                    rnn_inputs,
                                                                    sequence_length,
                                                                    dtype=tf.float32)

    enc_output = tf.concat(enc_output,2)
    
    return enc_output, enc_state

def trainDecodeLayer(dec_embed_input, lenSumm, dec_cell, initial_state, output_layer, 
                            vocab_size, maxSummText):
    '''Create the training logits'''
    
    training_helper = tf.contrib.seq2seq.TrainingHelper(inputs=dec_embed_input,
                                                        sequence_length=lenSumm,
                                                        time_major=False)

    training_decoder = tf.contrib.seq2seq.BasicDecoder(dec_cell,
                                                       training_helper,
                                                       initial_state,
                                                       output_layer) 

    training_logits, *_ = tf.contrib.seq2seq.dynamic_decode(training_decoder,
                                                           output_time_major=False,
                                                           impute_finished=True,
                                                           maximum_iterations=maxSummText)
    return training_logits


def interfaceDecodeLayer(embeddings, start_token, end_token, dec_cell, initial_state, output_layer,
                             maxSummText, bSize):
    '''Create the inference logits'''
    
    start_tokens = tf.tile(tf.constant([start_token], dtype=tf.int32), [bSize], name='start_tokens')
    
    inference_helper = tf.contrib.seq2seq.GreedyEmbeddingHelper(embeddings,
                                                                start_tokens,
                                                                end_token)
                
    inference_decoder = tf.contrib.seq2seq.BasicDecoder(dec_cell,
                                                        inference_helper,
                                                        initial_state,
                                                        output_layer)
                
    inference_logits, *_ = tf.contrib.seq2seq.dynamic_decode(inference_decoder,
                                                            output_time_major=False,
                                                            impute_finished=True,
                                                            maximum_iterations=maxSummText)
    
    return inference_logits


def decodeLayer(dec_embed_input, embeddings, enc_output, enc_state, vocab_size, lenText, lenSumm, 
                   maxSummText, sizeRnn, vocabToInt, probKeep, bSize, layerNum):
    '''Create the decoding cell and attention for the training and inference decoding layers'''
    
    for layer in range(layerNum):
        with tf.variable_scope('decoder_{}'.format(layer)):
            lstm = tf.contrib.rnn.LSTMCell(sizeRnn,
                                           initializer=tf.random_uniform_initializer(-0.1, 0.1, seed=2))
            dec_cell = tf.contrib.rnn.DropoutWrapper(lstm, 
                                                     input_probKeep = probKeep)
    
    output_layer = Dense(vocab_size,
                         kernel_initializer = tf.truncated_normal_initializer(mean = 0.0, stddev=0.1))
    
    attn_mech = tf.contrib.seq2seq.BahdanauAttention(sizeRnn,
                                                  enc_output,
                                                  lenText,
                                                  normalize=False,
                                                  name='BahdanauAttention')

    dec_cell = tf.contrib.seq2seq.AttentionWrapper(dec_cell,
                                                          attn_mech,
                                                          sizeRnn)
            
    initial_state = dec_cell.zero_state(bSize, tf.float32)
    with tf.variable_scope("decode"):
        training_logits = trainDecodeLayer(dec_embed_input, 
                                                  lenSumm, 
                                                  dec_cell, 
                                                  initial_state,
                                                  output_layer,
                                                  vocab_size, 
                                                  maxSummText)
    with tf.variable_scope("decode", reuse=True):
        inference_logits = interfaceDecodeLayer(embeddings,  
                                                    vocabToInt['<GO>'], 
                                                    vocabToInt['<EOS>'],
                                                    dec_cell, 
                                                    initial_state, 
                                                    output_layer,
                                                    maxSummText,
                                                    bSize)

    return training_logits, inference_logits

def seqModel(inData, target_data, probKeep, lenText, lenSumm, maxSummText, 
                  vocab_size, sizeRnn, layerNum, vocabToInt, bSize):
    embeddings = weMatrix
    
    enc_embed_input = tf.nn.embedding_lookup(embeddings, inData)
    enc_output, enc_state = encLayer(sizeRnn, lenText, layerNum, enc_embed_input, probKeep)
    
    dec_input = encInpProcess(target_data, vocabToInt, bSize)
    dec_embed_input = tf.nn.embedding_lookup(embeddings, dec_input)
    
    training_logits, inference_logits  = decodeLayer(dec_embed_input, 
                                                        embeddings,
                                                        enc_output,
                                                        enc_state, 
                                                        vocab_size, 
                                                        lenText, 
                                                        lenSumm, 
                                                        maxSummText,
                                                        sizeRnn, 
                                                        vocabToInt, 
                                                        probKeep, 
                                                        bSize,
                                                        layerNum)
    
    return training_logits, inference_logits

def padSentence(sentence_batch):
    senMax = max([len(sentence) for sentence in sentence_batch])
    return [sentence + [vocabToInt['<PAD>']] * (senMax - len(sentence)) for sentence in sentence_batch]

def getBatch(summaries, texts, bSize):
    for batchi in range(0, len(texts)//bSize):
        starti = batchi * bSize
        summaries_batch = summaries[starti:starti + bSize]
        texts_batch = texts[starti:starti + bSize]
        padsummbatch = np.array(padSentence(summaries_batch))
        padtextbatch = np.array(padSentence(texts_batch))
        
        padsummlengths = []
        for summary in padsummbatch:
            padsummlengths.append(len(summary))
        
        padtextlengths = []
        for text in padtextbatch:
            padtextlengths.append(len(text))
        
        yield padsummbatch, padtextbatch, padsummlengths, padtextlengths

epochs = 100
bSize = 64
sizeRnn = 256
layerNum = 2
learning_rate = 0.005
probKeepability = 0.75
train_graph = tf.Graph()

with train_graph.as_default():
    

    inData, targets, lr, probKeep, lenSumm, maxSummText, lenText = inpModel()


    training_logits, inference_logits = seqModel(tf.reverse(inData, [-1]),
                                                      targets, 
                                                      probKeep,   
                                                      lenText,
                                                      lenSumm,
                                                      maxSummText,
                                                      len(vocabToInt)+1,
                                                      sizeRnn, 
                                                      layerNum, 
                                                      vocabToInt,
                                                      bSize)
    
 
    training_logits = tf.identity(training_logits.rnn_output, 'logits')
    inference_logits = tf.identity(inference_logits.sample_id, name='predictions')
    

    masks = tf.sequence_mask(lenSumm, maxSummText, dtype=tf.float32, name='masks')

    with tf.name_scope("optimization"):
 
        cost = tf.contrib.seq2seq.sequence_loss(
            training_logits,
            targets,
            masks)

 
        optimizer = tf.train.AdamOptimizer(learning_rate)


        gradients = optimizer.compute_gradients(cost)
        capped_gradients = [(tf.clip_by_value(grad, -5., 5.), var) for grad, var in gradients if grad is not None]
        train_op = optimizer.apply_gradients(capped_gradients)
print("Graph is built.")


start = 200000
end = start + 50000
summarySort_short = summarySort[start:end]
sortText_short = sortText[start:end]
print("The shortest text length:", len(sortText_short[0]))
print("The longest text length:",len(sortText_short[-1]))

learning_rate_decay = 0.95
min_learning_rate = 0.0005
display_step = 20 
stop_early = 0 
stop = 3 
per_epoch = 3 
update_check = (len(sortText_short)//bSize//per_epoch)-1

update_loss = 0 
batch_loss = 0
summary_update_loss = [] 

checkpoint = "./best_model.ckpt" 
with tf.Session(graph=train_graph) as sess:
    sess.run(tf.global_variables_initializer())

    for epoch_i in range(1, epochs+1):
        update_loss = 0
        batch_loss = 0
        for batchi, (summaries_batch, texts_batch, summaries_lengths, texts_lengths) in enumerate(
                getBatch(summarySort_short, sortText_short, bSize)):
            start_time = time.time()
            _, loss = sess.run(
                [train_op, cost],
                {inData: texts_batch,
                 targets: summaries_batch,
                 lr: learning_rate,
                 lenSumm: summaries_lengths,
                 lenText: texts_lengths,
                 probKeep: probKeepability})

            batch_loss += loss
            update_loss += loss
            end_time = time.time()
            batch_time = end_time - start_time

            if batchi % display_step == 0 and batchi > 0:
                print('Epoch {:>3}/{} Batch {:>4}/{} - Loss: {:>6.3f}, Seconds: {:>4.2f}'
                      .format(epoch_i,
                              epochs, 
                              batchi, 
                              len(sortText_short) // bSize, 
                              batch_loss / display_step, 
                              batch_time*display_step))
                batch_loss = 0

            if batchi % update_check == 0 and batchi > 0:
                print("Average loss for this update:", round(update_loss/update_check,3))
                summary_update_loss.append(update_loss)
  
                if update_loss <= min(summary_update_loss):
                    print('New Record!') 
                    stop_early = 0
                    saver = tf.train.Saver() 
                    saver.save(sess, checkpoint)

                else:
                    print("No Improvement.")
                    stop_early += 1
                    if stop_early == stop:
                        break
                update_loss = 0
            
                 
                 
        learning_rate *= learning_rate_decay
        if learning_rate < min_learning_rate:
            learning_rate = min_learning_rate
        
        if stop_early == stop:
            print("Stopping Training.")
            break



def seqText(text):
    '''Prepare the text for the model'''
    
    text = cleanText(text)
    return [vocabToInt.get(word, vocabToInt['<UNK>']) for word in text.split()]


random = np.random.randint(0,len(cleanTexts))
input_sentence = cleanTexts[random]
text = seqText(cleanTexts[random])


loaded_graph = tf.Graph()
with tf.Session(graph=loaded_graph) as sess:
    loader = tf.train.import_meta_graph(checkpoint + '.meta')
    loader.restore(sess, checkpoint)

    inData = loaded_graph.get_tensor_by_name('input:0')
    logits = loaded_graph.get_tensor_by_name('predictions:0')
    lenText = loaded_graph.get_tensor_by_name('lenText:0')
    lenSumm = loaded_graph.get_tensor_by_name('lenSumm:0')
    probKeep = loaded_graph.get_tensor_by_name('probKeep:0')
    
    
    answer_logits = sess.run(logits, {inData: [text]*bSize, 
                                      lenSumm: [np.random.randint(5,8)], 
                                      lenText: [len(text)]*bSize,
                                      probKeep: 1.0})[0] 


pad = vocabToInt["<PAD>"] 

print('Original Text:', input_sentence)

print('\nText')
print('  Word Ids:    {}'.format([i for i in text]))
print('  Input Words: {}'.format(" ".join([intToVocab[i] for i in text])))

print('\nSummary')
print('  Word Ids:       {}'.format([i for i in answer_logits if i != pad]))
print('  Response Words: {}'.format(" ".join([intToVocab[i] for i in answer_logits if i != pad])))


