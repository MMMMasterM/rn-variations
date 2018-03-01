import tensorflow as tf
import numpy as np
import os, pickle, sys

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
epoch_count = 10

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

testTensorA = tf.constant([[[1, 2, 3, 4, 5, 6, 7], [8, 9, 10, 11, 12, 13, 14]]])
testTensorB = tf.constant([[[21, 22, 23, 24, 25, 26, 27], [28, 29, 30, 31, 32, 33, 34]]])
testTensorC = tf.constant([[[1, 2, 3, 4, 5, 6, 7], [8, 9, 10, 11, 12, 13, 14]], [[21, 22, 23, 24, 25, 26, 27], [28, 29, 30, 31, 32, 33, 34]]])
#testObjCount = tf.constant(2)
resTensorB = tf.reshape(testTensorA, shape=(7, 2))

#generate all combinations of two objects: first from A, second from B
def getHeteroCombinations(tensorA, tensorB):#input shape=(batch_size, obj_count, obj_dim)
    inputShapeA = tf.shape(tensorA)#[0] is batch_size, [1] is obj_count_A
    inputShapeB = tf.shape(tensorB)#[0] is batch_size, [1] is obj_count_B
    tensorRepA = tf.tile(tensorA, [1, 1, inputShapeB[1]])
    tensorA = tf.reshape(tensorRepA, shape=(inputShapeA[0], inputShapeA[1], inputShapeB[1], -1))

    tensorRepB = tf.tile(tensorB, [1, inputShapeA[1], 1])
    tensorB = tf.reshape(tensorRepB, shape=(inputShapeB[0], inputShapeA[1], inputShapeB[1], -1))
    return tf.concat([tensorA, tensorB], 3)#output shape=(batch_size, obj_count_A, obj_count_B, obj_dim_A + obj_dim_B)

#generate all combinations of two objects
def getCombinations(inputTensor):#input shape=(batch_size, obj_count, obj_dim)
    #implementation options to consider for best speed:
    # a)tile A, tile B
    # b)tile A -> transpose A into B
    # c)tile B -> transpose B into A

    #implementation b):
    inputShape = tf.shape(inputTensor)#[0] is batch_size, [1] is obj_count
    tensorRep = tf.tile(inputTensor, [1, inputShape[1], 1])
    tensorA = tf.reshape(tensorRep, shape=(inputShape[0], inputShape[1], inputShape[1], -1))
    tensorB = tf.transpose(tensorA, perm=[0, 2, 1, 3])

    #alternative formula for tensorB (for implementation a) and c))
    #tensorRepB = tf.tile(inputTensor, [1, 1, inputShape[1]])
    #tensorB = tf.reshape(tensorRepB, shape=(inputShape[0], inputShape[1], inputShape[1], -1))
    ##tensorA = tf.transpose(tensorB, perm=[0, 2, 1, 3])#for implementation c)
    return tf.concat([tensorB, tensorA], 3)#output shape=(batch_size, obj_count, obj_count, 2*obj_dim)

#generate all combinations (obj_ij, obj_jk)
def getTransitiveCombine(inputTensor):#input shape=(batch_size, obj_count, obj_count, processed_obj_dim)
    inputShape = tf.shape(inputTensor)#[0] is batch_size, [1] is obj_count

    tensorRepA = tf.tile(inputTensor, [1, 1, 1, inputShape[1]])
    tensorA = tf.reshape(tensorRepA, shape=(inputShape[0], inputShape[1], inputShape[1], inputShape[1], -1))

    tensorRepB = tf.tile(inputTensor, [1, inputShape[1], 1, 1])
    tensorB = tf.reshape(tensorRepB, shape=(inputShape[0], inputShape[1], inputShape[1], inputShape[1], -1))

    return tf.concat([tensorA, tensorB], 4)#output shape=(batch_size, obj_count, obj_count, obj_count, 2*processed_obj_dim)

#generate all combinations (o_i, o_j, o_k)
def getTripleCombinations(inputTensor):#input shape=(batch_size, obj_count, obj_dim)
    with tf.name_scope('triple_combinations'):
        inputShape = tf.shape(inputTensor)#[0] is batch_size, [1] is obj_count
        tensorRepA = tf.tile(inputTensor, [1, 1, inputShape[1]*inputShape[1]])
        tensorA = tf.reshape(tensorRepA, shape=(inputShape[0], inputShape[1], inputShape[1], inputShape[1], -1))
        tensorRepB = tf.tile(inputTensor, [1, inputShape[1], inputShape[1]])
        tensorB = tf.reshape(tensorRepB, shape=(inputShape[0], inputShape[1], inputShape[1], inputShape[1], -1))
        tensorRepC = tf.tile(inputTensor, [1, inputShape[1]*inputShape[1], 1])
        tensorC = tf.reshape(tensorRepC, shape=(inputShape[0], inputShape[1], inputShape[1], inputShape[1], -1))
        result = tf.concat([tensorA, tensorB, tensorC], 4)#output shape=(batch_size, obj_count, obj_count, obj_count, 3*obj_dim)

    return result

def buildRN_I(objects, question):#objects shape=(batch_size, obj_count, obj_dim), question shape=(batch_size, question_dim)
    with tf.name_scope('RN_I'):
        #model parameters
        g_dim = 256
        f_dim = 256

        def build_g(objPairs, question):#objPairs shape=(batch_size*obj_count*obj_count, 2*obj_dim)
            layerInput = tf.concat([objPairs, question], 1)
            return tf.layers.dense(layerInput, g_dim)#TODO: introduce constant for units count and add more layers

        def build_f(gSum, question):#gSum shape=(batch_size, g_dim)
            layerInput = gSum
            return tf.layers.dense(layerInput, f_dim)#TODO: introduce constant for units count and add more layers

        inputShape = tf.shape(objects)#[0] is batch_size, [1] is obj_count, [2] = obj_dim
        #questionShape = tf.shape(question)#[0] is batch_size, [1] is question_dim
        objPairs2D = getCombinations(objects)
        objPairs = tf.reshape(objPairs2D, shape=(-1, 2*obj_dim))
        questionRep = tf.tile(question, [1, inputShape[1]*inputShape[1]])
        questionRep = tf.reshape(questionRep, shape=(-1, question_dim))
        gResult = build_g(objPairs, questionRep)
        gResult1D = tf.reshape(gResult, shape=(batch_size, -1, g_dim))#shape=(batch_size, obj_count*obj_count, g_dim)
        gSum = tf.reduce_sum(gResult1D, axis=1)
        result = build_f(gSum, question)

    return result

def buildRN_II(objects, question):#objects shape=(batch_size, obj_count, obj_dim), question shape=(batch_size, question_dim)
    with tf.name_scope('RN_II'):
        #model parameters
        g_dim = 256
        f_dim = 256

        def build_g(objTriples, question):#objPairs shape=(batch_size*obj_count*obj_count*obj_count, 3*obj_dim)
            layerInput = tf.concat([objTriples, question], 1)
            return tf.layers.dense(layerInput, g_dim)#TODO: introduce constant for units count and add more layers

        def build_f(gSum, question):#gSum shape=(batch_size, g_dim)
            layerInput = gSum
            return tf.layers.dense(layerInput, f_dim)#TODO: introduce constant for units count and add more layers

        inputShape = tf.shape(objects)#[0] is batch_size, [1] is obj_count, [2] = obj_dim
        #questionShape = tf.shape(question)#[0] is batch_size, [1] is question_dim
        objTriples3D = getTripleCombinations(objects)
        objTriples = tf.reshape(objTriples3D, shape=(-1, 3*obj_dim))
        questionRep = tf.tile(question, [1, inputShape[1]*inputShape[1]*inputShape[1]])
        questionRep = tf.reshape(questionRep, shape=(-1, question_dim))
        gResult = build_g(objTriples, questionRep)
        gResult1D = tf.reshape(gResult, shape=(batch_size, -1, g_dim))#shape=(batch_size, obj_count*obj_count*obj_count, g_dim)
        gSum = tf.reduce_sum(gResult1D, axis=1)
        result = build_f(gSum, question)

    return result

def buildRN_III(objects, question):#objects shape=(batch_size, obj_count, obj_dim), question shape=(batch_size, question_dim)
    with tf.name_scope('RN_III'):
        #model parameters
        h_dim = 256
        g_dim = 256
        f_dim = 256

        def build_h(objPairs, question):#objPairs shape=(batch_size*obj_count*obj_count, 2*obj_dim)
            layerInput = tf.concat([objPairs, question], 1)
            return tf.layers.dense(layerInput, h_dim)#TODO: introduce constant for units count and add more layers

        def build_g(objPairs, question):#objPairs shape=(batch_size*obj_count*obj_count*obj_count, h_dim+obj_dim)
            layerInput = tf.concat([objPairs, question], 1)
            return tf.layers.dense(layerInput, g_dim)#TODO: introduce constant for units count and add more layers

        def build_f(gSum, question):#gSum shape=(batch_size, g_dim)
            layerInput = gSum
            return tf.layers.dense(layerInput, f_dim)#TODO: introduce constant for units count and add more layers

        inputShape = tf.shape(objects)#[0] is batch_size, [1] is obj_count, [2] = obj_dim
        #questionShape = tf.shape(question)#[0] is batch_size, [1] is question_dim
        objPairs2D = getCombinations(objects)
        objPairs = tf.reshape(objPairs2D, shape=(-1, 2*obj_dim))
        questionRep = tf.tile(question, [1, inputShape[1]*inputShape[1]])
        questionRep = tf.reshape(questionRep, shape=(-1, question_dim))
        hResult = build_h(objPairs, questionRep)
        hResult1D = tf.reshape(hResult, shape=(batch_size, inputShape[1]*inputShape[1], h_dim))
        #intermedShape = tf.shape(hResult1D)#[0] is batch_size, [1] is obj_count*obj_count, [2] is h_dim
        intermedObjPairs2D = getHeteroCombinations(hResult1D, objects)
        intermedObjPairs = tf.reshape(intermedObjPairs2D, shape=(-1, h_dim+obj_dim))
        questionRep3 = tf.tile(question, [1, inputShape[1]*inputShape[1]*inputShape[1]])
        questionRep3 = tf.reshape(questionRep3, shape=(-1, question_dim))
        gResult = build_g(intermedObjPairs, questionRep3)
        gResult1D = tf.reshape(gResult, shape=(batch_size, inputShape[1]*inputShape[1]*inputShape[1], g_dim))#shape=(batch_size, obj_count*obj_count*obj_count, g_dim)
        gSum = tf.reduce_sum(gResult1D, axis=1)
        result = build_f(gSum, question)

    return result

def buildRN_IV(objects, question):#objects shape=(batch_size, obj_count, obj_dim), question shape=(batch_size, question_dim)
    with tf.name_scope('RN_IV'):
        #model parameters
        h_dim = 256
        g_dim = 256
        f_dim = 256

        def build_h(objPairs, question):#objPairs shape=(batch_size*obj_count*obj_count, 2*obj_dim)
            layerInput = tf.concat([objPairs, question], 1)
            return tf.layers.dense(layerInput, h_dim)#TODO: introduce constant for units count and add more layers

        def build_g(objPairs, question):#objPairs shape=(batch_size*obj_count*obj_count*obj_count, 2*h_dim)
            layerInput = tf.concat([objPairs, question], 1)
            return tf.layers.dense(layerInput, g_dim)#TODO: introduce constant for units count and add more layers

        def build_f(gSum, question):#gSum shape=(batch_size, g_dim)
            layerInput = gSum
            return tf.layers.dense(layerInput, f_dim)#TODO: introduce constant for units count and add more layers

        inputShape = tf.shape(objects)#[0] is batch_size, [1] is obj_count, [2] = obj_dim
        #questionShape = tf.shape(question)#[0] is batch_size, [1] is question_dim
        objPairs2D = getCombinations(objects)
        objPairs = tf.reshape(objPairs2D, shape=(-1, 2*obj_dim))
        questionRep = tf.tile(question, [1, inputShape[1]*inputShape[1]])
        questionRep = tf.reshape(questionRep, shape=(-1, question_dim))
        hResult = build_h(objPairs, questionRep)
        hResult2D = tf.reshape(hResult, shape=(batch_size, inputShape[1], inputShape[1], h_dim))
        intermedObjPairs3D = getTransitiveCombine(hResult2D)
        #intermedShape = tf.shape(intermedObjPairs3D)#[0] is batch_size, [1] is obj_count, [2] is obj_count, [3] is obj_count, [4] is 2*h_dim
        intermedObjPairs = tf.reshape(intermedObjPairs3D, shape=(-1, 2*h_dim))
        questionRep3 = tf.tile(question, [1, inputShape[1]*inputShape[1]*inputShape[1]])
        questionRep3 = tf.reshape(questionRep3, shape=(-1, question_dim))
        gResult = build_g(intermedObjPairs, questionRep3)
        gResult1D = tf.reshape(gResult, shape=(batch_size, inputShape[1]*inputShape[1]*inputShape[1], g_dim))#shape=(batch_size, obj_count*obj_count*obj_count, g_dim)
        gSum = tf.reduce_sum(gResult1D, axis=1)
        result = build_f(gSum, question)

    return result

def buildRN_V(objects, question):#objects shape=(batch_size, obj_count, obj_dim), question shape=(batch_size, question_dim)
    with tf.name_scope('RN_V'):
        #model parameters
        h_dim = 256
        g_dim = 256
        f_inner_dim = 256
        f_dim = 256

        def build_h(objPairs, question):#objPairs shape=(batch_size*obj_count*obj_count, 2*obj_dim)
            layerInput = tf.concat([objPairs, question], 1)
            return tf.layers.dense(layerInput, h_dim)#TODO: introduce constant for units count and add more layers

        def build_g(objPairs, question):#objPairs shape=(batch_size*obj_count*obj_count*obj_count, 2*h_dim)
            layerInput = tf.concat([objPairs, question], 1)
            return tf.layers.dense(layerInput, g_dim)#TODO: introduce constant for units count and add more layers

        def build_f_inner(gSum):#gSum2D shape=(batch_size*obj_count*obj_count, g_dim)
            return tf.layers.dense(gSum, f_inner_dim)#TODO: introduce constant for units count and add more layers

        def build_f(gSum, question):#gSum shape=(batch_size, g_dim)
            layerInput = gSum
            return tf.layers.dense(layerInput, f_dim)#TODO: introduce constant for units count and add more layers

        inputShape = tf.shape(objects)#[0] is batch_size, [1] is obj_count, [2] = obj_dim
        #questionShape = tf.shape(question)#[0] is batch_size, [1] is question_dim
        objPairs2D = getCombinations(objects)
        objPairs = tf.reshape(objPairs2D, shape=(-1, 2*obj_dim))
        questionRep = tf.tile(question, [1, inputShape[1]*inputShape[1]])
        questionRep = tf.reshape(questionRep, shape=(-1, question_dim))
        hResult = build_h(objPairs, questionRep)
        hResult2D = tf.reshape(hResult, shape=(batch_size, inputShape[1], inputShape[1], -1))
        intermedObjPairs3D = getTransitiveCombine(hResult2D)
        #intermedShape = tf.shape(intermedObjPairs3D)#[0] is batch_size, [1] is obj_count, [2] is obj_count, [3] is obj_count, [4] is 2*h_dim
        intermedObjPairs = tf.reshape(intermedObjPairs3D, shape=(-1, 2*h_dim))
        questionRep3 = tf.tile(question, [1, inputShape[1]*inputShape[1]*inputShape[1]])
        questionRep3 = tf.reshape(questionRep3, shape=(-1, question_dim))
        gResult = build_g(intermedObjPairs, questionRep3)
        #gOutShape = tf.shape(gResult)#[0] is batch_size*obj_count*obj_count*obj_count, [1] is g_dim
        gResult3D = tf.reshape(gResult, shape=(batch_size, inputShape[1], inputShape[1], inputShape[1], g_dim))
        gSum2D = tf.reduce_sum(gResult3D, axis=2)
        gSum = tf.reshape(gSum2D, shape=(-1, g_dim))
        fInnerResult = build_f_inner(gSum)
        fInnerResult1D = tf.reshape(fInnerResult, shape=(batch_size, inputShape[1]*inputShape[1], f_inner_dim))
        fInnerSum = tf.reduce_sum(fInnerResult1D, axis=1)
        result = build_f(fInnerSum, question)

    return result

def buildRN_VI(objects, question):#objects shape=(batch_size, obj_count, obj_dim), question shape=(batch_size, question_dim)
    with tf.name_scope('RN_VI'):
        #model parameters
        h_dim = 256
        g_dim = 256
        f_inner_dim = 256
        f_dim = 256

        def build_h(objPairs, question):#objPairs shape=(batch_size*obj_count*obj_count, 2*obj_dim)
            layerInput = tf.concat([objPairs, question], 1)
            return tf.layers.dense(layerInput, h_dim)#TODO: introduce constant for units count and add more layers

        def build_g(objPairs, question):#objPairs shape=(batch_size*obj_count*obj_count*obj_count, 2*h_dim)
            layerInput = tf.concat([objPairs, question], 1)
            return tf.layers.dense(layerInput, g_dim)#TODO: introduce constant for units count and add more layers

        def build_f_inner(gSum):#gSum2D shape=(batch_size*obj_count, g_dim)
            return tf.layers.dense(gSum, f_inner_dim)#TODO: introduce constant for units count and add more layers

        def build_f(gSum, question):#gSum shape=(batch_size, g_dim)
            layerInput = gSum
            return tf.layers.dense(layerInput, f_dim)#TODO: introduce constant for units count and add more layers

        inputShape = tf.shape(objects)#[0] is batch_size, [1] is obj_count, [2] = obj_dim
        #questionShape = tf.shape(question)#[0] is batch_size, [1] is question_dim
        objPairs2D = getCombinations(objects)
        objPairs = tf.reshape(objPairs2D, shape=(-1, 2*obj_dim))
        questionRep = tf.tile(question, [1, inputShape[1]*inputShape[1]])
        questionRep = tf.reshape(questionRep, shape=(-1, question_dim))
        hResult = build_h(objPairs, questionRep)
        hResult2D = tf.reshape(hResult, shape=(batch_size, inputShape[1], inputShape[1], -1))
        intermedObjPairs3D = getTransitiveCombine(hResult2D)
        intermedShape = tf.shape(intermedObjPairs3D)#[0] is batch_size, [1] is obj_count, [2] is obj_count, [3] is obj_count, [4] is 2*h_dim
        intermedObjPairs = tf.reshape(intermedObjPairs3D, shape=(-1, 2*h_dim))
        questionRep3 = tf.tile(question, [1, inputShape[1]*inputShape[1]*inputShape[1]])
        questionRep3 = tf.reshape(questionRep3, shape=(-1, question_dim))
        gResult = build_g(intermedObjPairs, questionRep3)
        #gOutShape = tf.shape(gResult)#[0] is batch_size*obj_count*obj_count*obj_count, [1] is g_dim
        gResult3D = tf.reshape(gResult, shape=(batch_size, inputShape[1], inputShape[1], inputShape[1], g_dim))
        gSum1D = tf.reduce_sum(gResult3D, axis=[1,3])
        gSum = tf.reshape(gSum1D, shape=(-1, g_dim))
        fInnerResult = build_f_inner(gSum)
        fInnerResult1D = tf.reshape(fInnerResult, shape=(batch_size, inputShape[1], f_inner_dim))
        fInnerSum = tf.reduce_sum(fInnerResult1D, axis=1)
        result = build_f(fInnerSum, question)

    return result

def buildRN_VII_jl(objects, question):#objects shape=(batch_size, obj_count, obj_dim), question shape=(batch_size, question_dim)
    with tf.name_scope('RN_VII_jl'):
        #model parameters
        h_dim = 256
        g_dim = 256
        f_dim = 256

        def build_g(objPairs, question):#objPairs shape=(batch_size*obj_count*obj_count, 2*obj_dim)
            layerInput = tf.concat([objPairs, question], 1)
            return tf.layers.dense(layerInput, g_dim)#TODO: introduce constant for units count and add more layers

        def build_h(objPairs, question):#objPairs shape=(batch_size*obj_count*obj_count, 2*g_dim)
            layerInput = tf.concat([objPairs, question], 1)
            return tf.layers.dense(layerInput, h_dim)#TODO: introduce constant for units count and add more layers

        def build_f(hSum, question):#gSum shape=(batch_size, g_dim)
            layerInput = hSum
            return tf.layers.dense(layerInput, f_dim)#TODO: introduce constant for units count and add more layers

        inputShape = tf.shape(objects)#[0] is batch_size, [1] is obj_count, [2] = obj_dim
        objPairs2D = getCombinations(objects)
        objPairs = tf.reshape(objPairs2D, shape=(-1, 2*obj_dim))
        questionRep = tf.tile(question, [1, inputShape[1]*inputShape[1]])
        questionRep = tf.reshape(questionRep, shape=(-1, question_dim))
        gResult = build_g(objPairs, questionRep)
        gResult2D = tf.reshape(gResult, shape=(batch_size, inputShape[1], inputShape[1], g_dim))
        gSumJ = tf.reduce_sum(gResult2D, axis=2)
        #gSumL = gSumJ # correct but unused variable in optimized version
        #intermedShape = tf.shape(gResult2D)#[0] is batch_size, [1] is obj_count, [2] is obj_count, [3] is g_dim
        #intermedObjPairs2D = getHeteroCombinations(gSumJ, gSumL)#naive, unoptimized - optimize using knowledge that gSumL=gSumJ instead:
        intermedObjPairs2D = getCombinations(gSumJ)#optimized version
        intermedObjPairs = tf.reshape(intermedObjPairs2D, shape=(-1, 2*g_dim))
        hResult = build_h(intermedObjPairs, questionRep)
        hResult1D = tf.reshape(hResult, shape=(batch_size, inputShape[1]*inputShape[1], h_dim))
        hSum = tf.reduce_sum(hResult1D, axis=1)
        result = build_f(hSum, question)

    return result

def buildRN_VII_jk(objects, question):#objects shape=(batch_size, obj_count, obj_dim), question shape=(batch_size, question_dim)
    with tf.name_scope('RN_VII_jk'):
        #model parameters
        h_dim = 256
        g_dim = 256
        f_dim = 256

        def build_g(objPairs, question):#objPairs shape=(batch_size*obj_count*obj_count, 2*obj_dim)
            layerInput = tf.concat([objPairs, question], 1)
            return tf.layers.dense(layerInput, g_dim)#TODO: introduce constant for units count and add more layers

        def build_h(objPairs, question):#objPairs shape=(batch_size*obj_count*obj_count, 2*g_dim)
            layerInput = tf.concat([objPairs, question], 1)
            return tf.layers.dense(layerInput, h_dim)#TODO: introduce constant for units count and add more layers

        def build_f(hSum, question):#gSum shape=(batch_size, h_dim)
            layerInput = hSum
            return tf.layers.dense(layerInput, f_dim)#TODO: introduce constant for units count and add more layers

        inputShape = tf.shape(objects)#[0] is batch_size, [1] is obj_count, [2] = obj_dim
        objPairs2D = getCombinations(objects)
        objPairs = tf.reshape(objPairs2D, shape=(-1, 2*obj_dim))
        questionRep = tf.tile(question, [1, inputShape[1]*inputShape[1]])
        questionRep = tf.reshape(questionRep, shape=(-1, question_dim))
        gResult = build_g(objPairs, questionRep)
        gResult2D = tf.reshape(gResult, shape=(batch_size, inputShape[1], inputShape[1], g_dim))
        gSumJ = tf.reduce_sum(gResult2D, axis=2)
        gSumK = tf.reduce_sum(gResult2D, axis=1)
        #intermedShape = tf.shape(gResult2D)#[0] is batch_size, [1] is obj_count, [2] is obj_count, [3] is g_dim
        intermedObjPairs2D = getHeteroCombinations(gSumJ, gSumK)
        intermedObjPairs = tf.reshape(intermedObjPairs2D, shape=(-1, 2*g_dim))
        hResult = build_h(intermedObjPairs, questionRep)
        hResult1D = tf.reshape(hResult, shape=(batch_size, inputShape[1]*inputShape[1], h_dim))
        hSum = tf.reduce_sum(hResult1D, axis=1)
        result = build_f(hSum, question)

    return result

def buildRN_VIII_jl(objects, question, m):#objects shape=(batch_size, obj_count, obj_dim), question shape=(batch_size, question_dim), m=RN layer count
    with tf.name_scope('RN_VIII_jl'):
        #model parameters
        h_dim = 256
        g_dim = 256
        f_dim = 256

        def build_g(objPairs, question):#objPairs shape=(batch_size*obj_count*obj_count, 2*obj_dim)
            layerInput = tf.concat([objPairs, question], 1)
            return tf.layers.dense(layerInput, g_dim)#TODO: introduce constant for units count and add more layers

        def build_h(objPairs, question):#objPairs shape=(batch_size*obj_count*obj_count, 2*g_dim)
            layerInput = tf.concat([objPairs, question], 1)
            return tf.layers.dense(layerInput, h_dim)#TODO: introduce constant for units count and add more layers

        def build_f(hSum, question):#gSum shape=(batch_size, h_dim)
            layerInput = hSum
            return tf.layers.dense(layerInput, f_dim)#TODO: introduce constant for units count and add more layers

        inputShape = tf.shape(objects)#[0] is batch_size, [1] is obj_count, [2] = obj_dim
        objPairs2D = getCombinations(objects)
        objPairs = tf.reshape(objPairs2D, shape=(-1, 2*obj_dim))
        questionRep = tf.tile(question, [1, inputShape[1]*inputShape[1]])
        questionRep = tf.reshape(questionRep, shape=(-1, question_dim))
        gResult = build_g(objPairs, questionRep)

        prevResult = gResult#shape=(batch_size*obj_count*obj_count, g_dim)
        prevResultDim = g_dim
        for curLayer in range(m):
            prevResult2D = tf.reshape(gResult, shape=(batch_size, inputShape[1], inputShape[1], prevResultDim))
            sumJ = tf.reduce_sum(prevResult2D, axis=2)
            #sumL = sumJ # correct but unused variable in optimized version
            #intermedShape = tf.shape(prevResult2D)#[0] is batch_size, [1] is obj_count, [2] is obj_count, [3] is g_dim (for curLayer=0) or h_dim (for curLayer>0)
            #intermedObjPairs2D = getHeteroCombinations(sumJ, sumL)#naive, unoptimized - optimize using knowledge that sumL=sumJ instead:
            intermedObjPairs2D = getCombinations(sumJ)#optimized version
            intermedObjPairs = tf.reshape(intermedObjPairs2D, shape=(-1, 2*prevResultDim))
            hResult = build_h(intermedObjPairs, questionRep)
            prevResult = hResult
            prevResultDim = h_dim
        hResult1D = tf.reshape(prevResult, shape=(batch_size, inputShape[1]*inputShape[1], prevResultDim))
        hSum = tf.reduce_sum(hResult1D, axis=1)
        result = build_f(hSum, question)
    
    return result

def buildRN_VIII_jk(objects, question, m):#objects shape=(batch_size, obj_count, obj_dim), question shape=(batch_size, question_dim), m=RN layer count
    with tf.name_scope('RN_VIII_jk'):
        #model parameters
        h_dim = 256
        g_dim = 256
        f_dim = 256

        def build_g(objPairs, question):#objPairs shape=(batch_size*obj_count*obj_count, 2*obj_dim)
            layerInput = tf.concat([objPairs, question], 1)
            return tf.layers.dense(layerInput, g_dim)#TODO: introduce constant for units count and add more layers

        def build_h(objPairs, question):#objPairs shape=(batch_size*obj_count*obj_count, 2*g_dim)
            layerInput = tf.concat([objPairs, question], 1)
            return tf.layers.dense(layerInput, h_dim)#TODO: introduce constant for units count and add more layers

        def build_f(hSum, question):#gSum shape=(batch_size, h_dim)
            layerInput = hSum
            return tf.layers.dense(layerInput, f_dim)#TODO: introduce constant for units count and add more layers

        inputShape = tf.shape(objects)#[0] is batch_size, [1] is obj_count, [2] = obj_dim
        objPairs2D = getCombinations(objects)
        objPairs = tf.reshape(objPairs2D, shape=(-1, 2*obj_dim))
        questionRep = tf.tile(question, [1, inputShape[1]*inputShape[1]])
        questionRep = tf.reshape(questionRep, shape=(-1, question_dim))
        gResult = build_g(objPairs, questionRep)

        prevResult = gResult#shape=(batch_size*obj_count*obj_count, g_dim)
        prevResultDim = g_dim
        for curLayer in range(m):
            prevResult2D = tf.reshape(gResult, shape=(batch_size, inputShape[1], inputShape[1], prevResultDim))
            sumJ = tf.reduce_sum(prevResult2D, axis=2)
            sumK = tf.reduce_sum(prevResult2D, axis=1)
            #intermedShape = tf.shape(prevResult2D)#[0] is batch_size, [1] is obj_count, [2] is obj_count, [3] is g_dim (for curLayer=0) or h_dim (for curLayer>0)
            intermedObjPairs2D = getHeteroCombinations(sumJ, sumK)
            intermedObjPairs = tf.reshape(intermedObjPairs2D, shape=(-1, 2*prevResultDim))
            hResult = build_h(intermedObjPairs, questionRep)
            prevResult = hResult
            prevResultDim = h_dim
        hResult1D = tf.reshape(prevResult, shape=(batch_size, inputShape[1]*inputShape[1], prevResultDim))
        hSum = tf.reduce_sum(hResult1D, axis=1)
        result = build_f(hSum, question)

    return result

#process questions and context sentences through LSTMs and use the final LSTM states as input question and objects for the Relation Networks
def buildWordProcessorLSTMs():
    with tf.name_scope('wordProcessorLSTMs'):
        #model parameters
        embeddingDimension = 128
        qLstmHiddenUnits = question_dim
        sLstmHiddenUnits = obj_dim

        inputContext = tf.placeholder(tf.int32, shape=(batch_size, None, None))
        inputContextLengths = tf.placeholder(tf.int32, shape=(batch_size,))#Number of sentences in each context
        inputContextSentenceLengths = tf.placeholder(tf.int32, shape=(batch_size, None))#Number of words in each sentence
        inputQuestion = tf.placeholder(tf.int32, shape=(batch_size, None))
        inputQuestionLengths = tf.placeholder(tf.int32, shape=(batch_size,))

        #convert word indices to embedded representations (using learnable embeddings rather than one-hot vectors here)
        wordEmbedding = tf.Variable(tf.random_uniform(shape=[dictSize, embeddingDimension], minval=-1, maxval=1, seed=7))
        embeddedQuestion = tf.nn.embedding_lookup(wordEmbedding, inputQuestion)

        #setup question LSTM
        questionLSTMcell = tf.nn.rnn_cell.LSTMCell(num_units=qLstmHiddenUnits)
        questionLSTMoutputs, _ = tf.nn.dynamic_rnn(questionLSTMcell, embeddedQuestion, dtype=tf.float32, scope="questionLSTM")#shape=(batch_size, seq_len, qLstmHiddenUnits)
        #extract final states at the end of each sample's sequence
        inputQuestionMaxLength = tf.reduce_max(inputQuestionLengths)
        questionLSTMoutputs = tf.reshape(questionLSTMoutputs, shape=(-1, qLstmHiddenUnits))
        qSeqEndSelector = tf.range(batch_size) * inputQuestionMaxLength + (inputQuestionLengths - 1)
        questionLSTMoutputs = tf.gather(questionLSTMoutputs, qSeqEndSelector)#shape=(batch_size, qLstmHiddenUnits)

        #setup sentence LSTM
        inputContextMaxLength = tf.reduce_max(inputContextLengths)
        inputContextSentenceMaxLength = tf.reduce_max(inputContextSentenceLengths)
        inputSentences = tf.reshape(inputContext, shape=(batch_size*inputContextMaxLength, inputContextSentenceMaxLength))
        embeddedSentences = tf.nn.embedding_lookup(wordEmbedding, inputSentences)#shape=(batch_size*contextMaxLength, seq_len, embeddingDimension)
        #do we want to broadcast the question to the sentence LSTM here? or rather leave it entirely to the relation network
        #START VARIANTS
        # a) variant WITH broadcasting the questionLSTMoutputs to all sentences and timesteps (words/tokens):
        broadcastedQuestionLSTMoutputs = tf.expand_dims(questionLSTMoutputs, axis=1)#add time axis
        broadcastedQuestionLSTMoutputs = tf.tile(broadcastedQuestionLSTMoutputs, [inputContextMaxLength, inputContextSentenceMaxLength, 1])#repeat along time axis
        sentenceLSTMcell = tf.nn.rnn_cell.LSTMCell(num_units=sLstmHiddenUnits)
        sentenceLSTMoutputs, _ = tf.nn.dynamic_rnn(sentenceLSTMcell, tf.concat([embeddedSentences, broadcastedQuestionLSTMoutputs], axis=2), dtype=tf.float32, scope="sentenceLSTM")#shape=(batch_size*contextMaxLength, seq_len, sLstmHiddenUnits)
        # b) variant WITHOUT broadcasting the questionLSTMoutputs to all sentences and timesteps (words/tokens):
        # sentenceLSTMcell = tf.nn.rnn_cell.LSTMCell(num_units=sLstmHiddenUnits)
        # sentenceLSTMoutputs, _ = tf.nn.dynamic_rnn(sentenceLSTMcell, embeddedSentences, dtype=tf.float32, scope="sentenceLSTM")#shape=(batch_size*contextMaxLength, seq_len, sLstmHiddenUnits)
        #END VARIANTS
        #extract final states at the end of each sentence's sequence
        sentenceLSTMoutputs = tf.reshape(sentenceLSTMoutputs, shape=(-1, sLstmHiddenUnits))
        inputContextSentenceLengths1D = tf.reshape(inputContextSentenceLengths, shape=(-1,))
        sSeqEndSelector = tf.range(batch_size*inputContextMaxLength) * inputContextSentenceMaxLength + (inputContextSentenceLengths1D - 1)
        sentenceLSTMoutputs = tf.gather(sentenceLSTMoutputs, sSeqEndSelector)#shape=(batch_size*contextMaxLength, sLstmHiddenUnits)
        sentenceLSTMoutputs = tf.reshape(sentenceLSTMoutputs, shape=(batch_size, inputContextMaxLength, sLstmHiddenUnits))#these are the objects for the relation network input

        #TODO: optimization: dont apply the LSTMs to the padding-sentences (empty sentences added to make all contexts in a batch the same sentence-count)

    #return inputContext, inputContextLengths, inputQuestion, inputQuestionLengths, answer, answerGates
    return inputContext, inputContextLengths, inputContextSentenceLengths, inputQuestion, inputQuestionLengths, sentenceLSTMoutputs, questionLSTMoutputs

def buildAnswerModel(prevNetworkOutput):
    with tf.name_scope('answerModel'):
        #tf.nn.softmax removed because it's applied afterwards by the built-in loss function softmax_cross_entropy_with_logits
        #TODO: make a second output WITH softmax for validation/testing - not necessary while correctness is determined by argmax though due to monotonicity of softmax
        answer1 = tf.contrib.layers.fully_connected(prevNetworkOutput, dictSize, activation_fn=tf.nn.relu)#tf.nn.softmax)#shape=(batch_size, dictSize)
        answer2 = tf.contrib.layers.fully_connected(prevNetworkOutput, dictSize, activation_fn=tf.nn.relu)#tf.nn.softmax)
        answer3 = tf.contrib.layers.fully_connected(prevNetworkOutput, dictSize, activation_fn=tf.nn.relu)#tf.nn.softmax)
        answerGates = tf.contrib.layers.fully_connected(prevNetworkOutput, 3, activation_fn=tf.sigmoid)#shape=(batch_size, 3)
        answerStack = tf.stack([answer1, answer2, answer3], axis=1)#stack shape=(batch_size, 3, dictSize)
        answer = tf.reduce_sum(tf.multiply(answerStack, tf.expand_dims(answerGates, axis=2)), axis=1)

    return answer, answerGates

def buildOptimizer(answer, answerGates):
    with tf.name_scope('optimizer'):
        inputAnswer = tf.placeholder(tf.float32, shape=(batch_size, dictSize))#label
        #loss = tf.losses.mean_squared_error(labels=inputAnswer, predictions=answer) * dictSize - tf.reduce_mean(tf.square(answerGates - 0.5)) + 0.25#regularization term to enforce gate values close to 0 or 1
        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=answer, labels=inputAnswer)) - tf.reduce_mean(tf.square(answerGates - 0.5)) + 0.25#regularization term to enforce gate values close to 0 or 1
        #softmax_cross_entropy_with_logits is not suitable for outputs that are not probability distributions (which might be a problem for multi-answer questions) - still gives surprisingly good results for a first attempt
        optimizer = tf.train.AdamOptimizer(1e-5)
        gradients, variables = zip(*optimizer.compute_gradients(loss))
        gradients, _ = tf.clip_by_global_norm(gradients, 5.0)
        optimizer_op = optimizer.apply_gradients(zip(gradients, variables))
        #optimizer_op = tf.train.AdamOptimizer(1e-5).minimize(loss)

    return inputAnswer, loss, optimizer_op

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

(inputContext, inputContextLengths, inputContextSentenceLengths, inputQuestion, inputQuestionLengths, objects, question) = buildWordProcessorLSTMs()

if modelToUse == 1:
    print("Using model I")
    rnOutput = buildRN_I(objects, question)
elif modelToUse == 2:
    print("Using model II")
    rnOutput = buildRN_II(objects, question)
elif modelToUse == 3:
    print("Using model III")
    rnOutput = buildRN_III(objects, question)
elif modelToUse == 4:
    print("Using model IV")
    rnOutput = buildRN_IV(objects, question)
elif modelToUse == 5:
    print("Using model V")
    rnOutput = buildRN_V(objects, question)
elif modelToUse == 6:
    print("Using model VI")
    rnOutput = buildRN_VI(objects, question)
elif modelToUse == 7:
    print("Using model VII")
    rnOutput = buildRN_VII_jl(objects, question)
elif modelToUse == 8 and layerCount >= 0:
    print("Using model VIII with " + str(layerCount) + " layers")
    rnOutput = buildRN_VIII_jl(objects, question, layerCount)
else:
    print("Invalid model number specified: " + str(modelToUse))
    sys.exit(0)

(answer, answerGates) = buildAnswerModel(rnOutput)

(inputAnswer, loss, optimizer_op) = buildOptimizer(answer, answerGates)

with tf.name_scope('validation'):
    #correct = tf.reduce_min(tf.cast(tf.equal(inputAnswer, tf.round(answer)), dtype=tf.float32), axis=1)#bad results since the max entries often don't achieve 0.5 so rounding doesnt work
    correct = tf.cast(tf.equal(tf.argmax(inputAnswer, axis=1), tf.argmax(answer, axis=1)), dtype=tf.float32)#this is incorrect for multi-answer questions but gives better answers than rounding on single-answer questions -> TODO: find good solution for multi-answer questions
    #idea for better implementation of "correct"-variable: take argmax of answer1, answer2, answer3 each, also round answerGates and then calculate "answer" similar as in "buildModel()" and finally check tf.equal
    accuracy = tf.reduce_mean(correct)
    total_acc_placeholder = tf.placeholder(tf.float32, shape=())
    task_acc_placeholders = {}
    for task_name in validationData:
        task_acc_placeholders[task_name] = tf.placeholder(tf.float32, shape=())

#prepare tensorboard summaries
loss_summary = tf.summary.scalar('loss', loss)
total_acc_summary = tf.summary.scalar('total_acc', total_acc_placeholder)
task_acc_summaries = {}
for task_name in validationData:
    task_acc_summaries[task_name] = tf.summary.scalar('task_acc_' + task_name, task_acc_placeholders[task_name])
#merged_summaries = tf.summary.merge_all()
writer = tf.summary.FileWriter(logDir, sess.graph)
saver = tf.train.Saver()

def runValidation():
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
        writer.add_summary(summary)
        print("task accuracy " + str(taskAcc))
        total_acc.append(taskAcc)
    total_acc_val = sum(total_acc) / len(total_acc)
    summary = sess.run(total_acc_summary, feed_dict={total_acc_placeholder: total_acc_val})
    writer.add_summary(summary)
    print("total accuracy " + str(total_acc_val))

def train():
    for i, (contextInput, contextLengths, contextSentenceLengths, questionInput, questionLengths, answerInput) in enumerate(getBatches(trainingData, epoch_count)):
        feed_dict={inputContext: contextInput, inputContextLengths: contextLengths, inputContextSentenceLengths: contextSentenceLengths, inputQuestion: questionInput, inputQuestionLengths: questionLengths, inputAnswer: answerInput}
        #print(sess.run(tf.shape(objects), feed_dict=feed_dict))#debug
        sess.run(optimizer_op, feed_dict=feed_dict)
        summary, lossVal = sess.run([loss_summary, loss], feed_dict=feed_dict)
        writer.add_summary(summary)
        if (i % 50 == 0):
            print("batch " + str(i))
            print("loss " + str(lossVal))
        if (i % 1000 == 999):
            saver.save(sess, weightsPath)
            print('Model saved.')
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

if os.path.isfile(weightsPath):#restore weights
    saver.restore(sess, weightsPath)
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