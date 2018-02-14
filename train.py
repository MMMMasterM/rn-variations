import tensorflow as tf
import numpy as np

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

    tensorRepB = tf.tile(tensorB, [1, inputShapeB[1], 1])
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
    def build_g(objPairs, question):#objPairs shape=(batch_size*obj_count*obj_count, obj_dim)
        layerInput = tf.concat([objPairs, question], 1)
        return tf.layers.dense(layerInput, 256)#TODO: introduce constant for units count and add more layers

    def build_f(gSum, question):#gSum shape=(batch_size, g_dim)
        layerInput = gSum
        return tf.layers.dense(layerInput, 256)#TODO: introduce constant for units count and add more layers

    inputShape = tf.shape(objects)#[0] is batch_size, [1] is obj_count, [2] = obj_dim
    questionShape = tf.shape(question)#[0] is batch_size, [1] is question_dim
    objPairs2D = getCombinations(objects)
    objPairs = tf.reshape(objPairs2D, shape=(-1, inputShape[2]))
    questionRep = tf.tile(question, [1, inputShape[1]*inputShape[1]])
    questionRep = tf.reshape(questionRep, shape=(-1, questionShape[1]))
    gResult = build_g(objPairs, questionRep)
    gResult1D = tf.reshape(gResult, shape=(inputShape[0], inputShape[1]*inputShape[1], -1))
    gSum = tf.reduce_sum(gResult1D, axis=1)
    return build_f(gSum, question)

def buildRN_II(objects, question):#objects shape=(batch_size, obj_count, obj_dim), question shape=(batch_size, question_dim)
    def build_g(objTriples, question):#objPairs shape=(batch_size*obj_count*obj_count*obj_count, obj_dim)
        layerInput = tf.concat([objTriples, question], 1)
        return tf.layers.dense(layerInput, 256)#TODO: introduce constant for units count and add more layers

    def build_f(gSum, question):#gSum shape=(batch_size, g_dim)
        layerInput = gSum
        return tf.layers.dense(layerInput, 256)#TODO: introduce constant for units count and add more layers

    inputShape = tf.shape(objects)#[0] is batch_size, [1] is obj_count, [2] = obj_dim
    questionShape = tf.shape(question)#[0] is batch_size, [1] is question_dim
    objTriples3D = getTripleCombinations(objects)
    objTriples = tf.reshape(objTriples3D, shape=(-1, inputShape[2]))
    questionRep = tf.tile(question, [1, inputShape[1]*inputShape[1]*inputShape[1]])
    questionRep = tf.reshape(questionRep, shape=(-1, questionShape[1]))
    gResult = build_g(objTriples, questionRep)
    gResult1D = tf.reshape(gResult, shape=(inputShape[0], inputShape[1]*inputShape[1]*inputShape[1], -1))
    gSum = tf.reduce_sum(gResult1D, axis=1)
    return build_f(gSum, question)

def buildRN_III(objects, question):#objects shape=(batch_size, obj_count, obj_dim), question shape=(batch_size, question_dim)
    def build_h(objPairs, question):#objPairs shape=(batch_size*obj_count*obj_count, obj_dim)
        layerInput = tf.concat([objPairs, question], 1)
        return tf.layers.dense(layerInput, 256)#TODO: introduce constant for units count and add more layers

    def build_g(objPairs, question):#objPairs shape=(batch_size*obj_count*obj_count*obj_count, h_dim+obj_dim)
        layerInput = tf.concat([objPairs, question], 1)
        return tf.layers.dense(layerInput, 256)#TODO: introduce constant for units count and add more layers

    def build_f(gSum, question):#gSum shape=(batch_size, g_dim)
        layerInput = gSum
        return tf.layers.dense(layerInput, 256)#TODO: introduce constant for units count and add more layers

    inputShape = tf.shape(objects)#[0] is batch_size, [1] is obj_count, [2] = obj_dim
    questionShape = tf.shape(question)#[0] is batch_size, [1] is question_dim
    objPairs2D = getCombinations(objects)
    objPairs = tf.reshape(objPairs2D, shape=(-1, inputShape[2]))
    questionRep = tf.tile(question, [1, inputShape[1]*inputShape[1]])
    questionRep = tf.reshape(questionRep, shape=(-1, questionShape[1]))
    hResult = build_h(objPairs, questionRep)
    hResult1D = tf.reshape(gResult, shape=(inputShape[0], inputShape[1]*inputShape[1], -1))
    intermedShape = tf.shape(hResult1D)#[0] is batch_size, [1] is obj_count*obj_count, [2] is h_dim
    intermedObjPairs2D = getHeteroCombinations(hResult1D, objects)
    intermedObjPairs = tf.reshape(intermedObjPairs2D, shape=(-1, intermedShape[2]))
    questionRep3 = tf.tile(question, [1, inputShape[1]*inputShape[1]*inputShape[1]])
    questionRep3 = tf.reshape(questionRep3, shape=(-1, questionShape[1]))
    gResult = build_g(intermedObjPairs, questionRep3)
    gResult1D = tf.reshape(gResult, shape=(inputShape[0], inputShape[1]*inputShape[1]*inputShape[1], -1))
    gSum = tf.reduce_sum(gResult1D, axis=1)
    return build_f(gSum, question)

def buildRN_IV(objects, question):#objects shape=(batch_size, obj_count, obj_dim), question shape=(batch_size, question_dim)
    def build_h(objPairs, question):#objPairs shape=(batch_size*obj_count*obj_count, obj_dim)
        layerInput = tf.concat([objPairs, question], 1)
        return tf.layers.dense(layerInput, 256)#TODO: introduce constant for units count and add more layers

    def build_g(objPairs, question):#objPairs shape=(batch_size*obj_count*obj_count*obj_count, 2*h_dim)
        layerInput = tf.concat([objPairs, question], 1)
        return tf.layers.dense(layerInput, 256)#TODO: introduce constant for units count and add more layers

    def build_f(gSum, question):#gSum shape=(batch_size, g_dim)
        layerInput = gSum
        return tf.layers.dense(layerInput, 256)#TODO: introduce constant for units count and add more layers

    inputShape = tf.shape(objects)#[0] is batch_size, [1] is obj_count, [2] = obj_dim
    questionShape = tf.shape(question)#[0] is batch_size, [1] is question_dim
    objPairs2D = getCombinations(objects)
    objPairs = tf.reshape(objPairs2D, shape=(-1, inputShape[2]))
    questionRep = tf.tile(question, [1, inputShape[1]*inputShape[1]])
    questionRep = tf.reshape(questionRep, shape=(-1, questionShape[1]))
    hResult = build_h(objPairs, questionRep)
    hResult2D = tf.reshape(gResult, shape=(inputShape[0], inputShape[1], inputShape[1], -1))
    intermedObjPairs3D = getTransitiveCombine(hResult2D)
    intermedShape = tf.shape(intermedObjPairs3D)#[0] is batch_size, [1] is obj_count, [2] is obj_count, [3] is obj_count, [4] is 2*h_dim
    intermedObjPairs = tf.reshape(intermedObjPairs3D, shape=(-1, intermedShape[4]))
    questionRep3 = tf.tile(question, [1, inputShape[1]*inputShape[1]*inputShape[1]])
    questionRep3 = tf.reshape(questionRep3, shape=(-1, questionShape[1]))
    gResult = build_g(intermedObjPairs, questionRep3)
    gResult1D = tf.reshape(gResult, shape=(inputShape[0], inputShape[1]*inputShape[1]*inputShape[1], -1))
    gSum = tf.reduce_sum(gResult1D, axis=1)
    return build_f(gSum, question)

def buildRN_V(objects, question):#objects shape=(batch_size, obj_count, obj_dim), question shape=(batch_size, question_dim)
    def build_h(objPairs, question):#objPairs shape=(batch_size*obj_count*obj_count, obj_dim)
        layerInput = tf.concat([objPairs, question], 1)
        return tf.layers.dense(layerInput, 256)#TODO: introduce constant for units count and add more layers

    def build_g(objPairs, question):#objPairs shape=(batch_size*obj_count*obj_count*obj_count, 2*h_dim)
        layerInput = tf.concat([objPairs, question], 1)
        return tf.layers.dense(layerInput, 256)#TODO: introduce constant for units count and add more layers

    def build_f_inner(gSum):#gSum2D shape=(batch_size*obj_count*obj_count, g_dim)
        return tf.layers.dense(gSum, 256)#TODO: introduce constant for units count and add more layers

    def build_f(gSum, question):#gSum shape=(batch_size, g_dim)
        layerInput = gSum
        return tf.layers.dense(layerInput, 256)#TODO: introduce constant for units count and add more layers

    inputShape = tf.shape(objects)#[0] is batch_size, [1] is obj_count, [2] = obj_dim
    questionShape = tf.shape(question)#[0] is batch_size, [1] is question_dim
    objPairs2D = getCombinations(objects)
    objPairs = tf.reshape(objPairs2D, shape=(-1, inputShape[2]))
    questionRep = tf.tile(question, [1, inputShape[1]*inputShape[1]])
    questionRep = tf.reshape(questionRep, shape=(-1, questionShape[1]))
    hResult = build_h(objPairs, questionRep)
    hResult2D = tf.reshape(gResult, shape=(inputShape[0], inputShape[1], inputShape[1], -1))
    intermedObjPairs3D = getTransitiveCombine(hResult2D)
    intermedShape = tf.shape(intermedObjPairs3D)#[0] is batch_size, [1] is obj_count, [2] is obj_count, [3] is obj_count, [4] is 2*h_dim
    intermedObjPairs = tf.reshape(intermedObjPairs3D, shape=(-1, intermedShape[4]))
    questionRep3 = tf.tile(question, [1, inputShape[1]*inputShape[1]*inputShape[1]])
    questionRep3 = tf.reshape(questionRep3, shape=(-1, questionShape[1]))
    gResult = build_g(intermedObjPairs, questionRep3)
    gOutShape = tf.shape(gResult)#[0] is batch_size*obj_count*obj_count*obj_count, [1] is g_dim
    gResult3D = tf.reshape(gResult, shape=(inputShape[0], inputShape[1], inputShape[1], inputShape[1], -1))
    gSum2D = tf.reduce_sum(gResult3D, axis=2)
    gSum = tf.reshape(gSum2D, shape=(-1, gOutShape[1]))
    fInnerResult = build_f_inner(gSum)
    fInnerResult1D = tf.reshape(fInnerResult, shape=(inputShape[0], inputShape[1]*inputShape[1], -1))
    fInnerSum = tf.reduce_sum(fInnerResult1D, axis=1)
    return build_f(fInnerSum, question)

def buildRN_VI(objects, question):#objects shape=(batch_size, obj_count, obj_dim), question shape=(batch_size, question_dim)
    def build_h(objPairs, question):#objPairs shape=(batch_size*obj_count*obj_count, obj_dim)
        layerInput = tf.concat([objPairs, question], 1)
        return tf.layers.dense(layerInput, 256)#TODO: introduce constant for units count and add more layers

    def build_g(objPairs, question):#objPairs shape=(batch_size*obj_count*obj_count*obj_count, 2*h_dim)
        layerInput = tf.concat([objPairs, question], 1)
        return tf.layers.dense(layerInput, 256)#TODO: introduce constant for units count and add more layers

    def build_f_inner(gSum):#gSum2D shape=(batch_size*obj_count, g_dim)
        return tf.layers.dense(gSum, 256)#TODO: introduce constant for units count and add more layers

    def build_f(gSum, question):#gSum shape=(batch_size, g_dim)
        layerInput = gSum
        return tf.layers.dense(layerInput, 256)#TODO: introduce constant for units count and add more layers

    inputShape = tf.shape(objects)#[0] is batch_size, [1] is obj_count, [2] = obj_dim
    questionShape = tf.shape(question)#[0] is batch_size, [1] is question_dim
    objPairs2D = getCombinations(objects)
    objPairs = tf.reshape(objPairs2D, shape=(-1, inputShape[2]))
    questionRep = tf.tile(question, [1, inputShape[1]*inputShape[1]])
    questionRep = tf.reshape(questionRep, shape=(-1, questionShape[1]))
    hResult = build_h(objPairs, questionRep)
    hResult2D = tf.reshape(gResult, shape=(inputShape[0], inputShape[1], inputShape[1], -1))
    intermedObjPairs3D = getTransitiveCombine(hResult2D)
    intermedShape = tf.shape(intermedObjPairs3D)#[0] is batch_size, [1] is obj_count, [2] is obj_count, [3] is obj_count, [4] is 2*h_dim
    intermedObjPairs = tf.reshape(intermedObjPairs3D, shape=(-1, intermedShape[4]))
    questionRep3 = tf.tile(question, [1, inputShape[1]*inputShape[1]*inputShape[1]])
    questionRep3 = tf.reshape(questionRep3, shape=(-1, questionShape[1]))
    gResult = build_g(intermedObjPairs, questionRep3)
    gOutShape = tf.shape(gResult)#[0] is batch_size*obj_count*obj_count*obj_count, [1] is g_dim
    gResult3D = tf.reshape(gResult, shape=(inputShape[0], inputShape[1], inputShape[1], inputShape[1], -1))
    gSum1D = tf.reduce_sum(gResult3D, axis=[1,3])
    gSum = tf.reshape(gSum1D, shape=(-1, gOutShape[1]))
    fInnerResult = build_f_inner(gSum)
    fInnerResult1D = tf.reshape(fInnerResult, shape=(inputShape[0], inputShape[1], -1))
    fInnerSum = tf.reduce_sum(fInnerResult1D, axis=1)
    return build_f(fInnerSum, question)

def buildRN_VII_jl(objects, question):#objects shape=(batch_size, obj_count, obj_dim), question shape=(batch_size, question_dim)
    def build_g(objPairs, question):#objPairs shape=(batch_size*obj_count*obj_count, obj_dim)
        layerInput = tf.concat([objPairs, question], 1)
        return tf.layers.dense(layerInput, 256)#TODO: introduce constant for units count and add more layers

    def build_h(objPairs, question):#objPairs shape=(batch_size*obj_count*obj_count, obj_dim)
        layerInput = tf.concat([objPairs, question], 1)
        return tf.layers.dense(layerInput, 256)#TODO: introduce constant for units count and add more layers

    def build_f(hSum, question):#gSum shape=(batch_size, g_dim)
        layerInput = hSum
        return tf.layers.dense(layerInput, 256)#TODO: introduce constant for units count and add more layers

    inputShape = tf.shape(objects)#[0] is batch_size, [1] is obj_count, [2] = obj_dim
    objPairs2D = getCombinations(objects)
    objPairs = tf.reshape(objPairs2D, shape=(-1, inputShape[2]))
    questionRep = tf.tile(question, [1, inputShape[1]*inputShape[1]])
    questionRep = tf.reshape(questionRep, shape=(-1, questionShape[1]))
    gResult = build_g(objPairs, questionRep)
    gResult2D = tf.reshape(gResult, shape=(inputShape[0], inputShape[1], inputShape[1], -1))
    gSumJ = tf.reduce_sum(gResult2D, axis=2)
    #gSumL = gSumJ # correct but unused variable in optimized version
    intermedShape = tf.shape(gResult2D)#[0] is batch_size, [1] is obj_count, [2] is obj_count, [3] is g_dim
    #intermedObjPairs2D = getHeteroCombinations(gSumJ, gSumL)#naive, unoptimized - optimize using knowledge that gSumL=gSumJ instead:
    intermedObjPairs2D = getCombinations(gSumJ)#optimized version
    intermedObjPairs = tf.reshape(intermedObjPairs2D, shape=(-1, intermedShape[3]))
    hResult = build_h(intermedObjPairs, questionRep)
    hResult1D = tf.reshape(hResult, shape=(inputShape[0], inputShape[1]*inputShape[1], -1))
    hSum = tf.reduce_sum(hResult1D, axis=1)
    return build_f(hSum, question)

def buildRN_VII_jk(objects, question):#objects shape=(batch_size, obj_count, obj_dim), question shape=(batch_size, question_dim)
    def build_g(objPairs, question):#objPairs shape=(batch_size*obj_count*obj_count, obj_dim)
        layerInput = tf.concat([objPairs, question], 1)
        return tf.layers.dense(layerInput, 256)#TODO: introduce constant for units count and add more layers

    def build_h(objPairs, question):#objPairs shape=(batch_size*obj_count*obj_count, obj_dim)
        layerInput = tf.concat([objPairs, question], 1)
        return tf.layers.dense(layerInput, 256)#TODO: introduce constant for units count and add more layers

    def build_f(hSum, question):#gSum shape=(batch_size, g_dim)
        layerInput = hSum
        return tf.layers.dense(layerInput, 256)#TODO: introduce constant for units count and add more layers

    inputShape = tf.shape(objects)#[0] is batch_size, [1] is obj_count, [2] = obj_dim
    objPairs2D = getCombinations(objects)
    objPairs = tf.reshape(objPairs2D, shape=(-1, inputShape[2]))
    questionRep = tf.tile(question, [1, inputShape[1]*inputShape[1]])
    questionRep = tf.reshape(questionRep, shape=(-1, questionShape[1]))
    gResult = build_g(objPairs, questionRep)
    gResult2D = tf.reshape(gResult, shape=(inputShape[0], inputShape[1], inputShape[1], -1))
    gSumJ = tf.reduce_sum(gResult2D, axis=2)
    gSumK = tf.reduce_sum(gResult2D, axis=1)
    intermedShape = tf.shape(gResult2D)#[0] is batch_size, [1] is obj_count, [2] is obj_count, [3] is g_dim
    intermedObjPairs2D = getHeteroCombinations(gSumJ, gSumK)
    intermedObjPairs = tf.reshape(intermedObjPairs2D, shape=(-1, intermedShape[3]))
    hResult = build_h(intermedObjPairs, questionRep)
    hResult1D = tf.reshape(hResult, shape=(inputShape[0], inputShape[1]*inputShape[1], -1))
    hSum = tf.reduce_sum(hResult1D, axis=1)
    return build_f(hSum, question)

def buildRN_VIII_jl(objects, question, m):#objects shape=(batch_size, obj_count, obj_dim), question shape=(batch_size, question_dim), m=RN layer count
    def build_g(objPairs, question):#objPairs shape=(batch_size*obj_count*obj_count, obj_dim)
        layerInput = tf.concat([objPairs, question], 1)
        return tf.layers.dense(layerInput, 256)#TODO: introduce constant for units count and add more layers

    def build_h(objPairs, question):#objPairs shape=(batch_size*obj_count*obj_count, obj_dim)
        layerInput = tf.concat([objPairs, question], 1)
        return tf.layers.dense(layerInput, 256)#TODO: introduce constant for units count and add more layers

    def build_f(hSum, question):#gSum shape=(batch_size, g_dim)
        layerInput = hSum
        return tf.layers.dense(layerInput, 256)#TODO: introduce constant for units count and add more layers

    inputShape = tf.shape(objects)#[0] is batch_size, [1] is obj_count, [2] = obj_dim
    objPairs2D = getCombinations(objects)
    objPairs = tf.reshape(objPairs2D, shape=(-1, inputShape[2]))
    questionRep = tf.tile(question, [1, inputShape[1]*inputShape[1]])
    questionRep = tf.reshape(questionRep, shape=(-1, questionShape[1]))
    gResult = build_g(objPairs, questionRep)

    prevResult = gResult#shape=(batch_size*obj_count*obj_count, g_dim)
    for curLayer in range(m):
        prevResult2D = tf.reshape(gResult, shape=(inputShape[0], inputShape[1], inputShape[1], -1))
        sumJ = tf.reduce_sum(prevResult2D, axis=2)
        #sumL = sumJ # correct but unused variable in optimized version
        intermedShape = tf.shape(prevResult2D)#[0] is batch_size, [1] is obj_count, [2] is obj_count, [3] is g_dim (for curLayer=0) or h_dim (for curLayer>0)
        #intermedObjPairs2D = getHeteroCombinations(sumJ, sumL)#naive, unoptimized - optimize using knowledge that sumL=sumJ instead:
        intermedObjPairs2D = getCombinations(sumJ)#optimized version
        intermedObjPairs = tf.reshape(intermedObjPairs2D, shape=(-1, intermedShape[3]))
        hResult = build_h(intermedObjPairs, questionRep)
        prevResult = hResult
    hResult1D = tf.reshape(prevResult, shape=(inputShape[0], inputShape[1]*inputShape[1], -1))
    hSum = tf.reduce_sum(hResult1D, axis=1)
    return build_f(hSum, question)

def buildRN_VIII_jk(objects, question, m):#objects shape=(batch_size, obj_count, obj_dim), question shape=(batch_size, question_dim), m=RN layer count
    def build_g(objPairs, question):#objPairs shape=(batch_size*obj_count*obj_count, obj_dim)
        layerInput = tf.concat([objPairs, question], 1)
        return tf.layers.dense(layerInput, 256)#TODO: introduce constant for units count and add more layers

    def build_h(objPairs, question):#objPairs shape=(batch_size*obj_count*obj_count, obj_dim)
        layerInput = tf.concat([objPairs, question], 1)
        return tf.layers.dense(layerInput, 256)#TODO: introduce constant for units count and add more layers

    def build_f(hSum, question):#gSum shape=(batch_size, g_dim)
        layerInput = hSum
        return tf.layers.dense(layerInput, 256)#TODO: introduce constant for units count and add more layers

    inputShape = tf.shape(objects)#[0] is batch_size, [1] is obj_count, [2] = obj_dim
    objPairs2D = getCombinations(objects)
    objPairs = tf.reshape(objPairs2D, shape=(-1, inputShape[2]))
    questionRep = tf.tile(question, [1, inputShape[1]*inputShape[1]])
    questionRep = tf.reshape(questionRep, shape=(-1, questionShape[1]))
    gResult = build_g(objPairs, questionRep)

    prevResult = gResult#shape=(batch_size*obj_count*obj_count, g_dim)
    for curLayer in range(m):
        prevResult2D = tf.reshape(gResult, shape=(inputShape[0], inputShape[1], inputShape[1], -1))
        sumJ = tf.reduce_sum(prevResult2D, axis=2)
        sumK = tf.reduce_sum(prevResult2D, axis=1)
        intermedShape = tf.shape(prevResult2D)#[0] is batch_size, [1] is obj_count, [2] is obj_count, [3] is g_dim (for curLayer=0) or h_dim (for curLayer>0)
        intermedObjPairs2D = getHeteroCombinations(sumJ, sumK)
        intermedObjPairs = tf.reshape(intermedObjPairs2D, shape=(-1, intermedShape[3]))
        hResult = build_h(intermedObjPairs, questionRep)
        prevResult = hResult
    hResult1D = tf.reshape(prevResult, shape=(inputShape[0], inputShape[1]*inputShape[1], -1))
    hSum = tf.reduce_sum(hResult1D, axis=1)
    return build_f(hSum, question)


#print(sess.run(resTensorB))
print(sess.run(getHeteroCombinations(testTensorC, testTensorC)))
print(sess.run(getCombinations(testTensorC)))
print(sess.run(getTransitiveCombine(getCombinations(testTensorC))))

print(sess.run(getTripleCombinations(testTensorC)))

#writer = tf.summary.FileWriter('log', sess.graph)