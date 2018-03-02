import tensorflow as tf
import numpy as np
import os, pickle, sys
from model import ModelBuilder

#load dictionary
with open(os.path.join('processeddata', 'dictionary.txt'), 'rb') as f:
    wordIndices = pickle.load(f)
dictSize = len(wordIndices) + 1#padding entry with index 0 is not listed in wordIndices

#load data
with open(os.path.join('processeddata', 'test.txt'), 'rb') as f:
    testingData = pickle.load(f)

batch_size = 1
question_dim = 256
obj_dim = 256

#parse which RN to use
modelToUse = 1
layerCount = 0#only relevant for RN VIII
if len(sys.argv) >= 2:
    modelToUse = int(sys.argv[1])
    if len(sys.argv) >= 3:
        layerCount = int(sys.argv[2])

sess = tf.Session()

#utility functions
def getIndices(dataset, epochs):#generate indices over all epochs
    for epoch in range(epochs):
        for index in np.random.permutation(len(dataset)):
            yield index

def getBatchIndices(dataset, epochs):#generate batches of indices
    batchIndices = []
    for index in getIndices(dataset, epochs):
        batchIndices.append(index)
        if len(batchIndices) >= batch_size:
            yield batchIndices
            batchIndices = []

def getBatches(dataset, epochs):#generate batches of data
    for batchIndices in getBatchIndices(dataset, epochs):
        samples = [dataset[i] for i in batchIndices]
        contextLengths = [len(context) for context, question, answer in samples]
        maxContextLen = max(contextLengths)
        #Debug print: 
        # if maxContextLen >= 100 and batch_size == 1:
        #     print('Warning, large context detected: ' + str(maxContextLen) + ' skipping...')
        #     continue
        contextSentenceLengths = [[len(sentence) for sentence in context] for context, question, answer in samples]
        maxContextSentenceLen = max([max(sentenceLengthInContext) for sentenceLengthInContext in contextSentenceLengths])
        questionLengths = [len(question) for context, question, answer in samples]
        maxQuestionLen = max(questionLengths)
        #build tensors from data and apply padding
        emptySentence = [0]*maxContextSentenceLen#empty sentence for batch context padding
        contextInput = [[sentence + [0]*(maxContextSentenceLen - len(sentence)) for sentence in context] + [emptySentence]*(maxContextLen - len(context)) for context, question, answer in samples]
        contextSentenceLengths = [sentenceLengths + [1]*(maxContextLen - len(sentenceLengths)) for sentenceLengths in contextSentenceLengths]#apply padding for tensorflow tensor - padding with 1 instead of 0 so sequence-end-selectors dont fail with bufferunderrun
        questionInput = [question + [0]*(maxQuestionLen - len(question)) for context, question, answer in samples]
        answerInput = [answer for context, question, answer in samples]
        yield contextInput, contextLengths, contextSentenceLengths, questionInput, questionLengths, answerInput

#build the whole model and run it
modelBuilder = ModelBuilder(batch_size, question_dim, obj_dim, dictSize)

(inputContext, inputContextLengths, inputContextSentenceLengths, inputQuestion, inputQuestionLengths, objects, question) = modelBuilder.buildWordProcessorLSTMs()

if modelToUse == 1:
    print("Using model I")
    rnOutput = modelBuilder.buildRN_I(objects, question)
elif modelToUse == 2:
    print("Using model II")
    rnOutput = modelBuilder.buildRN_II(objects, question)
elif modelToUse == 3:
    print("Using model III")
    rnOutput = modelBuilder.buildRN_III(objects, question)
elif modelToUse == 4:
    print("Using model IV")
    rnOutput = modelBuilder.buildRN_IV(objects, question)
elif modelToUse == 5:
    print("Using model V")
    rnOutput = modelBuilder.buildRN_V(objects, question)
elif modelToUse == 6:
    print("Using model VI")
    rnOutput = modelBuilder.buildRN_VI(objects, question)
elif modelToUse == 7:
    print("Using model VII")
    rnOutput = modelBuilder.buildRN_VII_jl(objects, question)
elif modelToUse == 8 and layerCount >= 0:
    print("Using model VIII with " + str(layerCount) + " layers")
    rnOutput = modelBuilder.buildRN_VIII_jl(objects, question, layerCount)
else:
    print("Invalid model number specified: " + str(modelToUse))
    sys.exit(0)

(answer, answerGates) = modelBuilder.buildAnswerModel(rnOutput)

(inputAnswer, loss, optimizer_op) = modelBuilder.buildOptimizer(answer, answerGates)

with tf.name_scope('testing'):
    #correct = tf.reduce_min(tf.cast(tf.equal(inputAnswer, tf.round(answer)), dtype=tf.float32), axis=1)#bad results since the max entries often don't achieve 0.5 so rounding doesnt work
    correct = tf.cast(tf.equal(tf.argmax(inputAnswer, axis=1), tf.argmax(answer, axis=1)), dtype=tf.float32)#this is incorrect for multi-answer questions but gives better answers than rounding on single-answer questions -> TODO: find good solution for multi-answer questions
    #idea for better implementation of "correct"-variable: take argmax of answer1, answer2, answer3 each, also round answerGates and then calculate "answer" similar as in "buildModel()" and finally check tf.equal
    accuracy = tf.reduce_mean(correct)

saver = tf.train.Saver()

def runTest():
    total_acc = []
    for task_name in testingData:
        print("testing " + task_name)
        acc = []
        for i, (contextInput, contextLengths, contextSentenceLengths, questionInput, questionLengths, answerInput) in enumerate(getBatches(testingData[task_name], 1)):
            #print("validation batch " + str(i))
            feed_dict={inputContext: contextInput, inputContextLengths: contextLengths, inputContextSentenceLengths: contextSentenceLengths, inputQuestion: questionInput, inputQuestionLengths: questionLengths, inputAnswer: answerInput}
            batchAcc = sess.run(accuracy, feed_dict)
            acc.append(batchAcc)
            # print("label")
            # print(answerInput)
            # print("prediction")
            # print(sess.run(answer, feed_dict))
            # print("correct")
            # print(sess.run(correct, feed_dict))
            # print("gates")
            # print(sess.run(answerGates, feed_dict))
        taskAcc = sum(acc) / len(acc)
        print("task accuracy " + str(taskAcc))
        total_acc.append(taskAcc)
    total_acc_val = sum(total_acc) / len(total_acc)
    print("total accuracy " + str(total_acc_val))

weightsDir = os.path.join('weights', '_'.join(sys.argv[1:]))
#weightsPath = os.path.join(weightsDir, 'model.ckpt')

lastCheckpoint = tf.train.latest_checkpoint(weightsDir)
if lastCheckpoint is None:
    print('Missing weights! Exiting...')
    sys.exit(0)

saver.restore(sess, lastCheckpoint)
print('Weights restored.')

runTest()

print('Finished')