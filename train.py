import tensorflow as tf
import numpy as np
import os, pickle, sys
import argparse
from itertools import islice
#from model import buildRN_I, buildRN_II, buildRN_III, buildRN_IV, buildRN_V, buildRN_VI, buildRN_VII_jl, buildRN_VII_jk, buildRN_VIII_jl, buildRN_VIII_jk, buildWordProcessorLSTMs, buildAnswerModel, buildOptimizer
from model import ModelBuilder
import math

#load dictionary
with open(os.path.join('processeddata', 'dictionary.txt'), 'rb') as f:
    wordIndices = pickle.load(f)
dictSize = len(wordIndices) + 1#padding entry with index 0 is not listed in wordIndices

#load data
with open(os.path.join('processeddata', 'train.txt'), 'rb') as f:
    trainingData = pickle.load(f)

with open(os.path.join('processeddata', 'valid.txt'), 'rb') as f:
    validationData = pickle.load(f)

enabledWeightSaveRestore = True

#training parameters
batch_size = 1#32
epoch_count = 60#only two epochs to determine good min/max 

parser = argparse.ArgumentParser()
parser.add_argument('modelToUse', metavar='modelToUse', type=int, nargs='?', default=1)
parser.add_argument('--layers', type=int, default=0)
parser.add_argument('--optimizer', default='adam')
parser.add_argument('--clr', action='store_true')#Cyclical learning rate
parser.add_argument('--learningRate', type=float, default=1e-5)
parser.add_argument('--questionAwareContext', action='store_true')
parser.add_argument('--f_layers', type=int, default=3)
parser.add_argument('--f_inner_layers', type=int, default=3)
parser.add_argument('--g_layers', type=int, default=3)
parser.add_argument('--h_layers', type=int, default=3)
parser.add_argument('--appendPosVec', action='store_true')
parser.add_argument('--obj_dim', type=int, default=256)
parser.add_argument('--question_dim', type=int, default=256)
parser.add_argument('--batch_size', type=int, default=32)#number of samples per training step - combines ceil(args.batch_size / batch_size) many batches into macro-batches containing ceil(args.batch_size / batch_size) * batch_size many samples in total
parser.add_argument('--batchNorm', action='store_true')
parser.add_argument('--layerNorm', action='store_true')
args = parser.parse_args()

if args.batchNorm:
    args.layerNorm = False#do not allow batchNorm and layerNorm at the same time - what would happen though?

macro_batch_size = args.batch_size#effective batch size - accumulates multiple batches' gradients and then performs a training step

question_dim = args.question_dim
obj_dim = args.obj_dim

#parse which RN to use
modelToUse = args.modelToUse
layerCount = args.layers#0#only relevant for RN VIII
# if len(sys.argv) >= 2:
#     modelToUse = int(sys.argv[1])
#     if len(sys.argv) >= 3:
#         layerCount = int(sys.argv[2])
if args.optimizer != 'adam' and args.optimizer != 'nesterov':
    print('Optimizer must be one of [adam, nesterov]')
    exit()

paramString = str(modelToUse) + '_' + str(layerCount) + '_' + args.optimizer + '_' + str(args.clr) + '_' + str(args.learningRate) + '_' + str(args.questionAwareContext) + '_' + str(args.h_layers) + '_' + str(args.g_layers) + '_' + str(args.f_inner_layers) + '_' + str(args.f_layers) + '_' + str(args.appendPosVec) + '_' + str(args.obj_dim) + '_' + str(args.question_dim) + '_' + str(macro_batch_size) + '_' + str(args.batchNorm) + '_' + str(args.layerNorm)
logDir = os.path.join('log', paramString)
try:
    os.stat(logDir)
except:
    os.mkdir(logDir)

#determine appropriate batch size for chosen model
if modelToUse == 1 or modelToUse == 7 or modelToUse == 8:
    print("Using batch_size=8 for models with quadratic complexity")
    batch_size = 8
else:
    print("Using batch_size=1 for models with cubic complexity")
    batch_size = 1

if args.batchNorm:
    print("Forcing batch_size=1 for batchnorm due to padding")
    batch_size = 1

clr_stepsize = math.floor(2 * len(trainingData) / batch_size)#as per recommendation in the CLR paper (https://arxiv.org/pdf/1506.01186.pdf), 2-10 times the number of iterations in an epoch

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
        if maxContextLen >= 100: # and batch_size == 1:
            print('Warning, large context detected: ' + str(maxContextLen) + ' skipping...')
            continue
        contextSentenceLengths = [[len(sentence) for sentence in context] for context, question, answer in samples]
        maxContextSentenceLen = max([max(sentenceLengthInContext) for sentenceLengthInContext in contextSentenceLengths])
        questionLengths = [len(question) for context, question, answer in samples]
        maxQuestionLen = max(questionLengths)
        #build tensors from data and apply padding
        emptySentence = [0]*maxContextSentenceLen#empty sentence for batch context padding
        contextInput = sum([[sentence + [0]*(maxContextSentenceLen - len(sentence)) for sentence in context] for context, question, answer in samples], [])#concatenated
        #contextInput = [[sentence + [0]*(maxContextSentenceLen - len(sentence)) for sentence in context] + [emptySentence]*(maxContextLen - len(context)) for context, question, answer in samples]
        contextSentenceLengths = sum(contextSentenceLengths, [])#concatenated
        #contextSentenceLengths = [sentenceLengths + [1]*(maxContextLen - len(sentenceLengths)) for sentenceLengths in contextSentenceLengths]#apply padding for tensorflow tensor - padding with 1 instead of 0 so sequence-end-selectors dont fail with bufferunderrun
        questionInput = [question + [0]*(maxQuestionLen - len(question)) for context, question, answer in samples]
        answerInput = [answer for context, question, answer in samples]
        yield contextInput, contextLengths, contextSentenceLengths, questionInput, questionLengths, answerInput

#build the whole model and run it
modelBuilder = ModelBuilder(batch_size, macro_batch_size, question_dim, obj_dim, dictSize, args.questionAwareContext, args.f_layers, args.f_inner_layers, args.g_layers, args.h_layers, args.appendPosVec, args.batchNorm, args.layerNorm)

(inputContext, inputContextLengths, inputContextSentenceLengths, inputQuestion, inputQuestionLengths, objects, question) = modelBuilder.buildWordProcessorLSTMs()

if modelToUse == 1:
    print("Using model I")
    (rnOutput, isTraining) = modelBuilder.buildRN_I(objects, question)
elif modelToUse == 2:
    print("Using model II")
    (rnOutput, isTraining) = modelBuilder.buildRN_II(objects, question)
elif modelToUse == 3:
    print("Using model III")
    (rnOutput, isTraining) = modelBuilder.buildRN_III(objects, question)
elif modelToUse == 4:
    print("Using model IV")
    (rnOutput, isTraining) = modelBuilder.buildRN_IV(objects, question)
elif modelToUse == 5:
    print("Using model V")
    (rnOutput, isTraining) = modelBuilder.buildRN_V(objects, question)
elif modelToUse == 6:
    print("Using model VI")
    (rnOutput, isTraining) = modelBuilder.buildRN_VI(objects, question)
elif modelToUse == 7:
    print("Using model VII")
    (rnOutput, isTraining) = modelBuilder.buildRN_VII_jl(objects, question)
elif modelToUse == 8 and layerCount >= 0:
    print("Using model VIII with " + str(layerCount) + " layers")
    (rnOutput, isTraining) = modelBuilder.buildRN_VIII_jl(objects, inputContextLengths, question, layerCount)
else:
    print("Invalid model number specified: " + str(modelToUse))
    sys.exit(0)

#(answer, answerGates, answerForCorrectness) = modelBuilder.buildAnswerModel(rnOutput)
(answer, answerForCorrectness) = modelBuilder.buildAnswerModel(rnOutput)

#(inputAnswer, loss, optimizer_op, global_step_tensor, gradientsNorm, learningRate) = modelBuilder.buildOptimizer(answer, args.optimizer)#, answerGates)
(inputAnswer, loss, accum_ops, zero_ops, train_step, global_step_tensor, gradientsNorm, learningRate) = modelBuilder.buildOptimizer(answer, args.optimizer)#, answerGates)

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
gradients_norm_summary = tf.summary.scalar('gradients_norm', gradientsNorm)
training_summary = tf.summary.merge([loss_summary, gradients_norm_summary])
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
            feed_dict={inputContext: contextInput, inputContextLengths: contextLengths, inputContextSentenceLengths: contextSentenceLengths, inputQuestion: questionInput, inputQuestionLengths: questionLengths, inputAnswer: answerInput, isTraining: False}
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

batches_per_macro_batch = math.ceil(macro_batch_size / batch_size)

def train():
    acc = []
    first_global_step = int(sess.run(global_step_tensor))#continue from restored global_step
    if first_global_step > 0:
        print('Skipping forward to global_step ' + str(first_global_step))
    for i, (contextInput, contextLengths, contextSentenceLengths, questionInput, questionLengths, answerInput) in islice(enumerate(getBatches(trainingData, epoch_count)), first_global_step, None):
        if i % batches_per_macro_batch == 0:
            sess.run(zero_ops)
        if args.clr:
            x = 1 - abs((i % (2*clr_stepsize)) / clr_stepsize - 1)#periodic triangle function, starting with 0
            minLR = 0.00002
            maxLR = 0.0001
            lr = minLR + (maxLR - minLR) * x
        else:
            lr = args.learningRate
        feed_dict={inputContext: contextInput, inputContextLengths: contextLengths, inputContextSentenceLengths: contextSentenceLengths, inputQuestion: questionInput, inputQuestionLengths: questionLengths, inputAnswer: answerInput, isTraining: True}
        #print(sess.run(tf.shape(objects), feed_dict=feed_dict))#debug
        sess.run(accum_ops, feed_dict=feed_dict)
        summary, lossVal, batchAcc = sess.run([training_summary, loss, accuracy], feed_dict=feed_dict)
        acc.append(batchAcc)
        writer.add_summary(summary, global_step=i)
        if i % batches_per_macro_batch == batches_per_macro_batch-1:
            sess.run(train_step, feed_dict={learningRate: lr})
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
            if enabledWeightSaveRestore:
                saver.save(sess, weightsPath, global_step=global_step_tensor)
                print('Model saved.')
            #run validation
            runValidation()

if enabledWeightSaveRestore:
    try:
        os.stat('weights')
    except:
        os.mkdir('weights')
    weightsDir = os.path.join('weights', paramString)
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
        #np.random.seed(sess.run(randomSeedVar))#TODO: fix "uninitialized variable" error on this line
        print('Weights restored.')
    else:#initialize weights
        sess.run(tf.global_variables_initializer())
        print('Weights initialized.')
else:
    sess.run(tf.global_variables_initializer())
    print('Weights initialized.')

total_parameters = 0
for variable in tf.trainable_variables():
    # shape is an array of tf.Dimension
    shape = variable.get_shape()
    print(shape)
    print(len(shape))
    variable_parameters = 1
    for dim in shape:
        print(dim)
        variable_parameters *= dim.value
    print(variable_parameters)
    total_parameters += variable_parameters
print('Total weight count:')
print(total_parameters)

train()

if enabledWeightSaveRestore:
    saver.save(sess, weightsPath)
    print('Training finished. Model saved.')
else:
    print('Training finished.')
