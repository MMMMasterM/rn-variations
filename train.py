import tensorflow as tf
import numpy as np
import os, pickle, sys
from itertools import islice
#from model import buildRN_I, buildRN_II, buildRN_III, buildRN_IV, buildRN_V, buildRN_VI, buildRN_VII_jl, buildRN_VII_jk, buildRN_VIII_jl, buildRN_VIII_jk, buildWordProcessorLSTMs, buildAnswerModel, buildOptimizer
from model import ModelBuilder

logDir = os.path.join('log', '_'.join(sys.argv[1:]))
try:
    os.stat(logDir)
except:
    os.mkdir(logDir)

#load dictionary
with open(os.path.join('processeddata', 'dictionary.txt'), 'rb') as f:
    wordIndices = pickle.load(f)
dictSize = len(wordIndices) + 1#padding entry with index 0 is not listed in wordIndices

#load data
with open(os.path.join('processeddata', 'train.txt'), 'rb') as f:
    trainingData = pickle.load(f)

with open(os.path.join('processeddata', 'valid.txt'), 'rb') as f:
    validationData = pickle.load(f)

#training parameters
batch_size = 1#32
epoch_count = 20

question_dim = 256
obj_dim = 256

#parse which RN to use
modelToUse = 1
layerCount = 0#only relevant for RN VIII
if len(sys.argv) >= 2:
    modelToUse = int(sys.argv[1])
    if len(sys.argv) >= 3:
        layerCount = int(sys.argv[2])

#determine appropriate batch size for chosen model
if modelToUse == 1 or modelToUse == 7 or modelToUse == 8:
    print("Using batch_size=8 for models with quadratic complexity")
    batch_size = 8
else:
    print("Using batch_size=1 for models with cubic complexity")
    batch_size = 1

sess = tf.Session()

# testTensorA = tf.constant([[[1, 2, 3, 4, 5, 6, 7], [8, 9, 10, 11, 12, 13, 14]]])
# testTensorB = tf.constant([[[21, 22, 23, 24, 25, 26, 27], [28, 29, 30, 31, 32, 33, 34]]])
# testTensorC = tf.constant([[[1, 2, 3, 4, 5, 6, 7], [8, 9, 10, 11, 12, 13, 14]], [[21, 22, 23, 24, 25, 26, 27], [28, 29, 30, 31, 32, 33, 34]]])
# #testObjCount = tf.constant(2)
# resTensorB = tf.reshape(testTensorA, shape=(7, 2))

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
        if maxContextLen >= 100 and batch_size == 1:
            print('Warning, large context detected: ' + str(maxContextLen) + ' skipping...')
            continue
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

(answer, answerGates, answerForCorrectness) = modelBuilder.buildAnswerModel(rnOutput)

(inputAnswer, loss, optimizer_op, global_step_tensor) = modelBuilder.buildOptimizer(answer, answerGates)

with tf.name_scope('validation'):
    #correct = tf.reduce_min(tf.cast(tf.equal(inputAnswer, tf.round(answer)), dtype=tf.float32), axis=1)#bad results since the max entries often don't achieve 0.5 so rounding doesnt work
    #correct = tf.cast(tf.equal(tf.argmax(inputAnswer, axis=1), tf.argmax(answer, axis=1)), dtype=tf.float32)#this is incorrect for multi-answer questions but gives better answers than rounding on single-answer questions -> TODO: find good solution for multi-answer questions
    #idea for better implementation of "correct"-variable: take argmax of answer1, answer2, answer3 each, also round answerGates and then calculate "answer" similar as in "buildModel()" and finally check tf.equal
    correct = tf.cast(tf.reduce_all(tf.equal(answerForCorrectness, inputAnswer), axis=1), dtype=tf.float32)
    accuracy = tf.reduce_mean(correct)
    total_acc_placeholder = tf.placeholder(tf.float32, shape=())
    task_acc_placeholders = {}
    for task_name in validationData:
        task_acc_placeholders[task_name] = tf.placeholder(tf.float32, shape=())
    training_acc_placeholder = tf.placeholder(tf.float32, shape=())

#prepare tensorboard summaries
loss_summary = tf.summary.scalar('loss', loss)
total_acc_summary = tf.summary.scalar('total_acc', total_acc_placeholder)
training_acc_summary = tf.summary.scalar('training_acc', training_acc_placeholder)
task_acc_summaries = {}
for task_name in validationData:
    task_acc_summaries[task_name] = tf.summary.scalar('task_acc_' + task_name, task_acc_placeholders[task_name])
#merged_summaries = tf.summary.merge_all()
writer = tf.summary.FileWriter(logDir, sess.graph)
saver = tf.train.Saver()

def runValidation():
    global_step = int(sess.run(global_step_tensor))
    total_acc = []
    for task_name in validationData:
        print("validating " + task_name)
        acc = []
        for i, (contextInput, contextLengths, contextSentenceLengths, questionInput, questionLengths, answerInput) in enumerate(getBatches(validationData[task_name], 1)):
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
        summary = sess.run(task_acc_summaries[task_name], feed_dict={task_acc_placeholders[task_name]: taskAcc})
        writer.add_summary(summary, global_step=global_step)
        print("task accuracy " + str(taskAcc))
        total_acc.append(taskAcc)
    total_acc_val = sum(total_acc) / len(total_acc)
    summary = sess.run(total_acc_summary, feed_dict={total_acc_placeholder: total_acc_val})
    writer.add_summary(summary, global_step=global_step)
    print("total accuracy " + str(total_acc_val))

def train():
    acc = []
    first_global_step = int(sess.run(global_step_tensor))#continue from restored global_step
    if first_global_step > 0:
        print('Skipping forward to global_step ' + str(first_global_step))
    for i, (contextInput, contextLengths, contextSentenceLengths, questionInput, questionLengths, answerInput) in islice(enumerate(getBatches(trainingData, epoch_count)), first_global_step, None):
        feed_dict={inputContext: contextInput, inputContextLengths: contextLengths, inputContextSentenceLengths: contextSentenceLengths, inputQuestion: questionInput, inputQuestionLengths: questionLengths, inputAnswer: answerInput}
        #print(sess.run(tf.shape(objects), feed_dict=feed_dict))#debug
        sess.run(optimizer_op, feed_dict=feed_dict)
        summary, lossVal, batchAcc = sess.run([loss_summary, loss, accuracy], feed_dict=feed_dict)
        acc.append(batchAcc)
        writer.add_summary(summary, global_step=i)
        if (i % 50 == 0):
            print("batch " + str(i))
            print("loss " + str(lossVal))
        if (i % 1000 == 999):
            #measure training accuracy over the last 1000 batches and write it to summary
            trainAcc = sum(acc) / len(acc)
            acc = []
            summary = sess.run(training_acc_summary, feed_dict={training_acc_placeholder: trainAcc})
            writer.add_summary(summary, global_step=i)
            print('Training accuracy: ' + str(trainAcc))
            #save model weights
            saver.save(sess, weightsPath, global_step=global_step_tensor)
            print('Model saved.')
            #run validation
            runValidation()

try:
    os.stat('weights')
except:
    os.mkdir('weights')
weightsDir = os.path.join('weights', '_'.join(sys.argv[1:]))
try:
    os.stat(weightsDir)
except:
    os.mkdir(weightsDir)

weightsPath = os.path.join(weightsDir, 'model.ckpt')

#generate new random generator seed to store
randomSeed = np.random.randint(2**16)
np.random.seed(randomSeed)
randomSeedVar = tf.Variable(randomSeed, trainable=False, name='randomSeedVar', dtype=tf.int32)

lastCheckpoint = tf.train.latest_checkpoint(weightsDir)
if lastCheckpoint is not None:#restore weights
    saver.restore(sess, lastCheckpoint)
    #restore random generator seed for seamless continue after skipping global_step number of batches
    np.random.seed(sess.run(randomSeedVar))
    print('Weights restored.')
else:#initialize weights
    sess.run(tf.global_variables_initializer())
    print('Weights initialized.')

train()

saver.save(sess, weightsPath)
print('Training finished. Model saved.')

#print(sess.run(resTensorB))
# print(sess.run(getHeteroCombinations(testTensorC, testTensorC)))
# print(sess.run(getCombinations(testTensorC)))
# print(sess.run(getTransitiveCombine(getCombinations(testTensorC))))

# print(sess.run(getTripleCombinations(testTensorC)))