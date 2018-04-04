import tensorflow as tf
import numpy as np
import os, pickle, sys
import math

#utility functions:
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


class ModelBuilder:
    def __init__(self, batch_size, macro_batch_size, question_dim, obj_dim, dictSize, questionAwareContext, f_layers, f_inner_layers, g_layers, h_layers, appendPosVec):
        self.batch_size = batch_size
        self.macro_batch_size = macro_batch_size
        self.question_dim = question_dim
        self.obj_dim = obj_dim
        self.dictSize = dictSize
        self.questionAwareContext = questionAwareContext
        self.f_layers = f_layers
        self.f_inner_layers = f_inner_layers
        self.g_layers = g_layers
        self.h_layers = h_layers
        self.appendPosVec = appendPosVec

    def buildRN_I(self, objects, question):#objects shape=(batch_size, obj_count, obj_dim), question shape=(batch_size, question_dim)
        with tf.name_scope('RN_I'):
            #model parameters
            g_dim = 256
            # g_layers = 3
            f_dim = 256
            # f_layers = 3

            def build_g(objPairs, question):#objPairs shape=(batch_size*obj_count*obj_count, 2*obj_dim)
                layerInput = tf.concat([objPairs, question], 1)
                for i in range(self.g_layers-1):
                    layerInput = tf.contrib.layers.fully_connected(layerInput, g_dim)
                return tf.contrib.layers.fully_connected(layerInput, g_dim)

            def build_f(gSum, question):#gSum shape=(batch_size, g_dim)
                layerInput = gSum
                for i in range(self.f_layers-1):
                    layerInput = tf.contrib.layers.fully_connected(layerInput, f_dim)
                return tf.contrib.layers.fully_connected(layerInput, f_dim)

            inputShape = tf.shape(objects)#[0] is batch_size, [1] is obj_count, [2] = obj_dim
            #questionShape = tf.shape(question)#[0] is batch_size, [1] is question_dim
            objPairs2D = getCombinations(objects)
            objPairs = tf.reshape(objPairs2D, shape=(-1, 2*self.obj_dim))
            questionRep = tf.tile(question, [1, inputShape[1]*inputShape[1]])
            questionRep = tf.reshape(questionRep, shape=(-1, self.question_dim))
            gResult = build_g(objPairs, questionRep)
            gResult1D = tf.reshape(gResult, shape=(self.batch_size, -1, g_dim))#shape=(batch_size, obj_count*obj_count, g_dim)
            gSum = tf.reduce_sum(gResult1D, axis=1)
            result = build_f(gSum, question)

        return result

    def buildRN_II(self, objects, question):#objects shape=(batch_size, obj_count, obj_dim), question shape=(batch_size, question_dim)
        with tf.name_scope('RN_II'):
            #model parameters
            g_dim = 256
            # g_layers = 3
            f_dim = 256
            # f_layers = 3

            def build_g(objTriples, question):#objPairs shape=(batch_size*obj_count*obj_count*obj_count, 3*obj_dim)
                layerInput = tf.concat([objTriples, question], 1)
                for i in range(self.g_layers-1):
                    layerInput = tf.contrib.layers.fully_connected(layerInput, g_dim)
                return tf.contrib.layers.fully_connected(layerInput, g_dim)

            def build_f(gSum, question):#gSum shape=(batch_size, g_dim)
                layerInput = gSum
                for i in range(self.f_layers-1):
                    layerInput = tf.contrib.layers.fully_connected(layerInput, f_dim)
                return tf.contrib.layers.fully_connected(layerInput, f_dim)

            inputShape = tf.shape(objects)#[0] is batch_size, [1] is obj_count, [2] = obj_dim
            #questionShape = tf.shape(question)#[0] is batch_size, [1] is question_dim
            objTriples3D = getTripleCombinations(objects)
            objTriples = tf.reshape(objTriples3D, shape=(-1, 3*self.obj_dim))
            questionRep = tf.tile(question, [1, inputShape[1]*inputShape[1]*inputShape[1]])
            questionRep = tf.reshape(questionRep, shape=(-1, self.question_dim))
            gResult = build_g(objTriples, questionRep)
            gResult1D = tf.reshape(gResult, shape=(self.batch_size, -1, g_dim))#shape=(batch_size, obj_count*obj_count*obj_count, g_dim)
            gSum = tf.reduce_sum(gResult1D, axis=1)
            result = build_f(gSum, question)

        return result

    def buildRN_III(self, objects, question):#objects shape=(batch_size, obj_count, obj_dim), question shape=(batch_size, question_dim)
        with tf.name_scope('RN_III'):
            #model parameters
            h_dim = 256
            # h_layers = 3
            g_dim = 256
            # g_layers = 3
            f_dim = 256
            # f_layers = 3

            def build_h(objPairs, question):#objPairs shape=(batch_size*obj_count*obj_count, 2*obj_dim)
                layerInput = tf.concat([objPairs, question], 1)
                for i in range(self.h_layers-1):
                    layerInput = tf.contrib.layers.fully_connected(layerInput, h_dim)
                return tf.contrib.layers.fully_connected(layerInput, h_dim)

            def build_g(objPairs, question):#objPairs shape=(batch_size*obj_count*obj_count*obj_count, h_dim+obj_dim)
                layerInput = tf.concat([objPairs, question], 1)
                for i in range(self.g_layers-1):
                    layerInput = tf.contrib.layers.fully_connected(layerInput, g_dim)
                return tf.contrib.layers.fully_connected(layerInput, g_dim)

            def build_f(gSum, question):#gSum shape=(batch_size, g_dim)
                layerInput = gSum
                for i in range(self.f_layers-1):
                    layerInput = tf.contrib.layers.fully_connected(layerInput, f_dim)
                return tf.contrib.layers.fully_connected(layerInput, f_dim)

            inputShape = tf.shape(objects)#[0] is batch_size, [1] is obj_count, [2] = obj_dim
            #questionShape = tf.shape(question)#[0] is batch_size, [1] is question_dim
            objPairs2D = getCombinations(objects)
            objPairs = tf.reshape(objPairs2D, shape=(-1, 2*self.obj_dim))
            questionRep = tf.tile(question, [1, inputShape[1]*inputShape[1]])
            questionRep = tf.reshape(questionRep, shape=(-1, self.question_dim))
            hResult = build_h(objPairs, questionRep)
            hResult1D = tf.reshape(hResult, shape=(self.batch_size, inputShape[1]*inputShape[1], h_dim))
            #intermedShape = tf.shape(hResult1D)#[0] is batch_size, [1] is obj_count*obj_count, [2] is h_dim
            intermedObjPairs2D = getHeteroCombinations(hResult1D, objects)
            intermedObjPairs = tf.reshape(intermedObjPairs2D, shape=(-1, h_dim+self.obj_dim))
            questionRep3 = tf.tile(question, [1, inputShape[1]*inputShape[1]*inputShape[1]])
            questionRep3 = tf.reshape(questionRep3, shape=(-1, self.question_dim))
            gResult = build_g(intermedObjPairs, questionRep3)
            gResult1D = tf.reshape(gResult, shape=(self.batch_size, inputShape[1]*inputShape[1]*inputShape[1], g_dim))#shape=(batch_size, obj_count*obj_count*obj_count, g_dim)
            gSum = tf.reduce_sum(gResult1D, axis=1)
            result = build_f(gSum, question)

        return result

    def buildRN_IV(self, objects, question):#objects shape=(batch_size, obj_count, obj_dim), question shape=(batch_size, question_dim)
        with tf.name_scope('RN_IV'):
            #model parameters
            h_dim = 256
            # h_layers = 3
            g_dim = 256
            # g_layers = 3
            f_dim = 256
            # f_layers = 3

            def build_h(objPairs, question):#objPairs shape=(batch_size*obj_count*obj_count, 2*obj_dim)
                layerInput = tf.concat([objPairs, question], 1)
                for i in range(self.h_layers-1):
                    layerInput = tf.contrib.layers.fully_connected(layerInput, h_dim)
                return tf.contrib.layers.fully_connected(layerInput, h_dim)

            def build_g(objPairs, question):#objPairs shape=(batch_size*obj_count*obj_count*obj_count, 2*h_dim)
                layerInput = tf.concat([objPairs, question], 1)
                for i in range(self.g_layers-1):
                    layerInput = tf.contrib.layers.fully_connected(layerInput, g_dim)
                return tf.contrib.layers.fully_connected(layerInput, g_dim)

            def build_f(gSum, question):#gSum shape=(batch_size, g_dim)
                layerInput = gSum
                for i in range(self.f_layers-1):
                    layerInput = tf.contrib.layers.fully_connected(layerInput, f_dim)
                return tf.contrib.layers.fully_connected(layerInput, f_dim)

            inputShape = tf.shape(objects)#[0] is batch_size, [1] is obj_count, [2] = obj_dim
            #questionShape = tf.shape(question)#[0] is batch_size, [1] is question_dim
            objPairs2D = getCombinations(objects)
            objPairs = tf.reshape(objPairs2D, shape=(-1, 2*self.obj_dim))
            questionRep = tf.tile(question, [1, inputShape[1]*inputShape[1]])
            questionRep = tf.reshape(questionRep, shape=(-1, self.question_dim))
            hResult = build_h(objPairs, questionRep)
            hResult2D = tf.reshape(hResult, shape=(self.batch_size, inputShape[1], inputShape[1], h_dim))
            intermedObjPairs3D = getTransitiveCombine(hResult2D)
            #intermedShape = tf.shape(intermedObjPairs3D)#[0] is batch_size, [1] is obj_count, [2] is obj_count, [3] is obj_count, [4] is 2*h_dim
            intermedObjPairs = tf.reshape(intermedObjPairs3D, shape=(-1, 2*h_dim))
            questionRep3 = tf.tile(question, [1, inputShape[1]*inputShape[1]*inputShape[1]])
            questionRep3 = tf.reshape(questionRep3, shape=(-1, self.question_dim))
            gResult = build_g(intermedObjPairs, questionRep3)
            gResult1D = tf.reshape(gResult, shape=(self.batch_size, inputShape[1]*inputShape[1]*inputShape[1], g_dim))#shape=(batch_size, obj_count*obj_count*obj_count, g_dim)
            gSum = tf.reduce_sum(gResult1D, axis=1)
            result = build_f(gSum, question)

        return result

    def buildRN_V(self, objects, question):#objects shape=(batch_size, obj_count, obj_dim), question shape=(batch_size, question_dim)
        with tf.name_scope('RN_V'):
            #model parameters
            h_dim = 256
            # h_layers = 3
            g_dim = 256
            # g_layers = 3
            f_inner_dim = 256
            # f_inner_layers = 3
            f_dim = 256
            # f_layers = 3

            def build_h(objPairs, question):#objPairs shape=(batch_size*obj_count*obj_count, 2*obj_dim)
                layerInput = tf.concat([objPairs, question], 1)
                for i in range(self.h_layers-1):
                    layerInput = tf.contrib.layers.fully_connected(layerInput, h_dim)
                return tf.contrib.layers.fully_connected(layerInput, h_dim)

            def build_g(objPairs, question):#objPairs shape=(batch_size*obj_count*obj_count*obj_count, 2*h_dim)
                layerInput = tf.concat([objPairs, question], 1)
                for i in range(self.g_layers-1):
                    layerInput = tf.contrib.layers.fully_connected(layerInput, g_dim)
                return tf.contrib.contrib.layers.fully_connected(layerInput, g_dim)

            def build_f_inner(gSum):#gSum2D shape=(batch_size*obj_count*obj_count, g_dim)
                layerInput = gSum
                for i in range(self.f_inner_layers-1):
                    layerInput = tf.contrib.layers.fully_connected(layerInput, f_inner_dim)
                return tf.contrib.layers.fully_connected(layerInput, f_inner_dim)

            def build_f(gSum, question):#gSum shape=(batch_size, g_dim)
                layerInput = gSum
                for i in range(self.f_layers-1):
                    layerInput = tf.contrib.layers.fully_connected(layerInput, f_dim)
                return tf.contrib.layers.fully_connected(layerInput, f_dim)

            inputShape = tf.shape(objects)#[0] is batch_size, [1] is obj_count, [2] = obj_dim
            #questionShape = tf.shape(question)#[0] is batch_size, [1] is question_dim
            objPairs2D = getCombinations(objects)
            objPairs = tf.reshape(objPairs2D, shape=(-1, 2*self.obj_dim))
            questionRep = tf.tile(question, [1, inputShape[1]*inputShape[1]])
            questionRep = tf.reshape(questionRep, shape=(-1, self.question_dim))
            hResult = build_h(objPairs, questionRep)
            hResult2D = tf.reshape(hResult, shape=(self.batch_size, inputShape[1], inputShape[1], -1))
            intermedObjPairs3D = getTransitiveCombine(hResult2D)
            #intermedShape = tf.shape(intermedObjPairs3D)#[0] is batch_size, [1] is obj_count, [2] is obj_count, [3] is obj_count, [4] is 2*h_dim
            intermedObjPairs = tf.reshape(intermedObjPairs3D, shape=(-1, 2*h_dim))
            questionRep3 = tf.tile(question, [1, inputShape[1]*inputShape[1]*inputShape[1]])
            questionRep3 = tf.reshape(questionRep3, shape=(-1, self.question_dim))
            gResult = build_g(intermedObjPairs, questionRep3)
            #gOutShape = tf.shape(gResult)#[0] is batch_size*obj_count*obj_count*obj_count, [1] is g_dim
            gResult3D = tf.reshape(gResult, shape=(self.batch_size, inputShape[1], inputShape[1], inputShape[1], g_dim))
            gSum2D = tf.reduce_sum(gResult3D, axis=2)
            gSum = tf.reshape(gSum2D, shape=(-1, g_dim))
            fInnerResult = build_f_inner(gSum)
            fInnerResult1D = tf.reshape(fInnerResult, shape=(self.batch_size, inputShape[1]*inputShape[1], f_inner_dim))
            fInnerSum = tf.reduce_sum(fInnerResult1D, axis=1)
            result = build_f(fInnerSum, question)

        return result

    def buildRN_VI(self, objects, question):#objects shape=(batch_size, obj_count, obj_dim), question shape=(batch_size, question_dim)
        with tf.name_scope('RN_VI'):
            #model parameters
            h_dim = 256
            # h_layers = 3
            g_dim = 256
            # g_layers = 3
            f_inner_dim = 256
            # f_inner_layers = 3
            f_dim = 256
            # f_layers = 3

            def build_h(objPairs, question):#objPairs shape=(batch_size*obj_count*obj_count, 2*obj_dim)
                layerInput = tf.concat([objPairs, question], 1)
                for i in range(self.h_layers-1):
                    layerInput = tf.contrib.layers.fully_connected(layerInput, h_dim)
                return tf.contrib.layers.fully_connected(layerInput, h_dim)

            def build_g(objPairs, question):#objPairs shape=(batch_size*obj_count*obj_count*obj_count, 2*h_dim)
                layerInput = tf.concat([objPairs, question], 1)
                for i in range(self.g_layers-1):
                    layerInput = tf.contrib.layers.fully_connected(layerInput, g_dim)
                return tf.contrib.layers.fully_connected(layerInput, g_dim)

            def build_f_inner(gSum):#gSum2D shape=(batch_size*obj_count, g_dim)
                layerInput = gSum
                for i in range(self.f_inner_layers-1):
                    layerInput = tf.contrib.layers.fully_connected(layerInput, f_inner_dim)
                return tf.contrib.layers.fully_connected(layerInput, f_inner_dim)

            def build_f(gSum, question):#gSum shape=(batch_size, g_dim)
                layerInput = gSum
                for i in range(self.f_layers-1):
                    layerInput = tf.contrib.layers.fully_connected(layerInput, f_dim)
                return tf.contrib.layers.fully_connected(layerInput, f_dim)

            inputShape = tf.shape(objects)#[0] is batch_size, [1] is obj_count, [2] = obj_dim
            #questionShape = tf.shape(question)#[0] is batch_size, [1] is question_dim
            objPairs2D = getCombinations(objects)
            objPairs = tf.reshape(objPairs2D, shape=(-1, 2*self.obj_dim))
            questionRep = tf.tile(question, [1, inputShape[1]*inputShape[1]])
            questionRep = tf.reshape(questionRep, shape=(-1, self.question_dim))
            hResult = build_h(objPairs, questionRep)
            hResult2D = tf.reshape(hResult, shape=(self.batch_size, inputShape[1], inputShape[1], -1))
            intermedObjPairs3D = getTransitiveCombine(hResult2D)
            intermedShape = tf.shape(intermedObjPairs3D)#[0] is batch_size, [1] is obj_count, [2] is obj_count, [3] is obj_count, [4] is 2*h_dim
            intermedObjPairs = tf.reshape(intermedObjPairs3D, shape=(-1, 2*h_dim))
            questionRep3 = tf.tile(question, [1, inputShape[1]*inputShape[1]*inputShape[1]])
            questionRep3 = tf.reshape(questionRep3, shape=(-1, self.question_dim))
            gResult = build_g(intermedObjPairs, questionRep3)
            #gOutShape = tf.shape(gResult)#[0] is batch_size*obj_count*obj_count*obj_count, [1] is g_dim
            gResult3D = tf.reshape(gResult, shape=(self.batch_size, inputShape[1], inputShape[1], inputShape[1], g_dim))
            gSum1D = tf.reduce_sum(gResult3D, axis=[1,3])
            gSum = tf.reshape(gSum1D, shape=(-1, g_dim))
            fInnerResult = build_f_inner(gSum)
            fInnerResult1D = tf.reshape(fInnerResult, shape=(self.batch_size, inputShape[1], f_inner_dim))
            fInnerSum = tf.reduce_sum(fInnerResult1D, axis=1)
            result = build_f(fInnerSum, question)

        return result

    def buildRN_VII_jl(self, objects, question):#objects shape=(batch_size, obj_count, obj_dim), question shape=(batch_size, question_dim)
        with tf.name_scope('RN_VII_jl'):
            #model parameters
            h_dim = 256
            # h_layers = 3
            g_dim = 256
            # g_layers = 3
            f_dim = 256
            # f_layers = 3

            def build_g(objPairs, question):#objPairs shape=(batch_size*obj_count*obj_count, 2*obj_dim)
                layerInput = tf.concat([objPairs, question], 1)
                for i in range(self.g_layers-1):
                    layerInput = tf.contrib.layers.fully_connected(layerInput, g_dim)
                return tf.contrib.layers.fully_connected(layerInput, g_dim)

            def build_h(objPairs, question):#objPairs shape=(batch_size*obj_count*obj_count, 2*g_dim)
                layerInput = tf.concat([objPairs, question], 1)
                for i in range(self.h_layers-1):
                    layerInput = tf.contrib.layers.fully_connected(layerInput, h_dim)
                return tf.contrib.layers.fully_connected(layerInput, h_dim)

            def build_f(hSum, question):#gSum shape=(batch_size, g_dim)
                layerInput = hSum
                for i in range(self.f_layers-1):
                    layerInput = tf.contrib.layers.fully_connected(layerInput, f_dim)
                return tf.contrib.layers.fully_connected(layerInput, f_dim)

            inputShape = tf.shape(objects)#[0] is batch_size, [1] is obj_count, [2] = obj_dim
            objPairs2D = getCombinations(objects)
            objPairs = tf.reshape(objPairs2D, shape=(-1, 2*self.obj_dim))
            questionRep = tf.tile(question, [1, inputShape[1]*inputShape[1]])
            questionRep = tf.reshape(questionRep, shape=(-1, self.question_dim))
            gResult = build_g(objPairs, questionRep)
            gResult2D = tf.reshape(gResult, shape=(self.batch_size, inputShape[1], inputShape[1], g_dim))
            gSumJ = tf.reduce_sum(gResult2D, axis=2)
            #gSumL = gSumJ # correct but unused variable in optimized version
            #intermedShape = tf.shape(gResult2D)#[0] is batch_size, [1] is obj_count, [2] is obj_count, [3] is g_dim
            #intermedObjPairs2D = getHeteroCombinations(gSumJ, gSumL)#naive, unoptimized - optimize using knowledge that gSumL=gSumJ instead:
            intermedObjPairs2D = getCombinations(gSumJ)#optimized version
            intermedObjPairs = tf.reshape(intermedObjPairs2D, shape=(-1, 2*g_dim))
            hResult = build_h(intermedObjPairs, questionRep)
            hResult1D = tf.reshape(hResult, shape=(self.batch_size, inputShape[1]*inputShape[1], h_dim))
            hSum = tf.reduce_sum(hResult1D, axis=1)
            result = build_f(hSum, question)

        return result

    def buildRN_VII_jk(self, objects, question):#objects shape=(batch_size, obj_count, obj_dim), question shape=(batch_size, question_dim)
        with tf.name_scope('RN_VII_jk'):
            #model parameters
            h_dim = 256
            # h_layers = 3
            g_dim = 256
            # g_layers = 3
            f_dim = 256
            # f_layers = 3

            def build_g(objPairs, question):#objPairs shape=(batch_size*obj_count*obj_count, 2*obj_dim)
                layerInput = tf.concat([objPairs, question], 1)
                for i in range(self.g_layers-1):
                    layerInput = tf.contrib.layers.fully_connected(layerInput, g_dim)
                return tf.contrib.layers.fully_connected(layerInput, g_dim)

            def build_h(objPairs, question):#objPairs shape=(batch_size*obj_count*obj_count, 2*g_dim)
                layerInput = tf.concat([objPairs, question], 1)
                for i in range(self.h_layers-1):
                    layerInput = tf.contrib.layers.fully_connected(layerInput, h_dim)
                return tf.contrib.layers.fully_connected(layerInput, h_dim)

            def build_f(hSum, question):#gSum shape=(batch_size, h_dim)
                layerInput = hSum
                for i in range(self.f_layers-1):
                    layerInput = tf.contrib.layers.fully_connected(layerInput, f_dim)
                return tf.contrib.layers.fully_connected(layerInput, f_dim)

            inputShape = tf.shape(objects)#[0] is batch_size, [1] is obj_count, [2] = obj_dim
            objPairs2D = getCombinations(objects)
            objPairs = tf.reshape(objPairs2D, shape=(-1, 2*self.obj_dim))
            questionRep = tf.tile(question, [1, inputShape[1]*inputShape[1]])
            questionRep = tf.reshape(questionRep, shape=(-1, self.question_dim))
            gResult = build_g(objPairs, questionRep)
            gResult2D = tf.reshape(gResult, shape=(self.batch_size, inputShape[1], inputShape[1], g_dim))
            gSumJ = tf.reduce_sum(gResult2D, axis=2)
            gSumK = tf.reduce_sum(gResult2D, axis=1)
            #intermedShape = tf.shape(gResult2D)#[0] is batch_size, [1] is obj_count, [2] is obj_count, [3] is g_dim
            intermedObjPairs2D = getHeteroCombinations(gSumJ, gSumK)
            intermedObjPairs = tf.reshape(intermedObjPairs2D, shape=(-1, 2*g_dim))
            hResult = build_h(intermedObjPairs, questionRep)
            hResult1D = tf.reshape(hResult, shape=(self.batch_size, inputShape[1]*inputShape[1], h_dim))
            hSum = tf.reduce_sum(hResult1D, axis=1)
            result = build_f(hSum, question)

        return result

    def buildRN_VIII_jl(self, objects, question, m):#objects shape=(batch_size, obj_count, obj_dim), question shape=(batch_size, question_dim), m=RN layer count
        with tf.name_scope('RN_VIII_jl'):
            #model parameters
            h_dim = 256
            # h_layers = 3
            g_dim = 256
            # g_layers = 3
            f_dim = 512
            # f_layers = 3

            def build_g(objPairs, question):#objPairs shape=(batch_size*obj_count*obj_count, 2*obj_dim)
                layerInput = tf.concat([objPairs, question], 1)
                for i in range(self.g_layers-1):
                    layerInput = tf.contrib.layers.fully_connected(layerInput, g_dim)
                return tf.contrib.layers.fully_connected(layerInput, g_dim)

            def build_h(objPairs, question):#objPairs shape=(batch_size*obj_count*obj_count, 2*g_dim)
                layerInput = tf.concat([objPairs, question], 1)
                for i in range(self.h_layers-1):
                    layerInput = tf.contrib.layers.fully_connected(layerInput, h_dim)
                return tf.contrib.layers.fully_connected(layerInput, h_dim)

            def build_f(hSum, question):#gSum shape=(batch_size, h_dim)
                layerInput = hSum
                for i in range(self.f_layers-1):
                    layerInput = tf.contrib.layers.fully_connected(layerInput, f_dim)
                return tf.contrib.layers.fully_connected(layerInput, f_dim)

            inputShape = tf.shape(objects)#[0] is batch_size, [1] is obj_count, [2] = obj_dim
            objPairs2D = getCombinations(objects)
            objPairs = tf.reshape(objPairs2D, shape=(-1, 2*self.obj_dim))
            questionRep = tf.tile(question, [1, inputShape[1]*inputShape[1]])
            questionRep = tf.reshape(questionRep, shape=(-1, self.question_dim))
            gResult = build_g(objPairs, questionRep)

            prevResult = gResult#shape=(batch_size*obj_count*obj_count, g_dim)
            prevResultDim = g_dim
            for curLayer in range(m):
                prevResult2D = tf.reshape(gResult, shape=(self.batch_size, inputShape[1], inputShape[1], prevResultDim))
                sumJ = tf.reduce_sum(prevResult2D, axis=2)
                #sumL = sumJ # correct but unused variable in optimized version
                #intermedShape = tf.shape(prevResult2D)#[0] is batch_size, [1] is obj_count, [2] is obj_count, [3] is g_dim (for curLayer=0) or h_dim (for curLayer>0)
                #intermedObjPairs2D = getHeteroCombinations(sumJ, sumL)#naive, unoptimized - optimize using knowledge that sumL=sumJ instead:
                intermedObjPairs2D = getCombinations(sumJ)#optimized version
                intermedObjPairs = tf.reshape(intermedObjPairs2D, shape=(-1, 2*prevResultDim))
                hResult = build_h(intermedObjPairs, questionRep)
                prevResult = hResult
                prevResultDim = h_dim
            hResult1D = tf.reshape(prevResult, shape=(self.batch_size, inputShape[1]*inputShape[1], prevResultDim))
            hSum = tf.reduce_sum(hResult1D, axis=1)
            result = build_f(hSum, question)
        
        return result

    def buildRN_VIII_jk(self, objects, question, m):#objects shape=(batch_size, obj_count, obj_dim), question shape=(batch_size, question_dim), m=RN layer count
        with tf.name_scope('RN_VIII_jk'):
            #model parameters
            h_dim = 256
            # h_layers = 3
            g_dim = 256
            # g_layers = 3
            f_dim = 512
            # f_layers = 3

            def build_g(objPairs, question):#objPairs shape=(batch_size*obj_count*obj_count, 2*obj_dim)
                layerInput = tf.concat([objPairs, question], 1)
                for i in range(self.g_layers-1):
                    layerInput = tf.contrib.layers.fully_connected(layerInput, g_dim)
                return tf.contrib.layers.fully_connected(layerInput, g_dim)

            def build_h(objPairs, question):#objPairs shape=(batch_size*obj_count*obj_count, 2*g_dim)
                layerInput = tf.concat([objPairs, question], 1)
                for i in range(self.h_layers-1):
                    layerInput = tf.contrib.layers.fully_connected(layerInput, h_dim)
                return tf.contrib.layers.fully_connected(layerInput, h_dim)

            def build_f(hSum, question):#gSum shape=(batch_size, h_dim)
                layerInput = hSum
                for i in range(self.f_layers-1):
                    layerInput = tf.contrib.layers.fully_connected(layerInput, f_dim)
                return tf.contrib.layers.fully_connected(layerInput, f_dim)

            inputShape = tf.shape(objects)#[0] is batch_size, [1] is obj_count, [2] = obj_dim
            objPairs2D = getCombinations(objects)
            objPairs = tf.reshape(objPairs2D, shape=(-1, 2*self.obj_dim))
            questionRep = tf.tile(question, [1, inputShape[1]*inputShape[1]])
            questionRep = tf.reshape(questionRep, shape=(-1, self.question_dim))
            gResult = build_g(objPairs, questionRep)

            prevResult = gResult#shape=(batch_size*obj_count*obj_count, g_dim)
            prevResultDim = g_dim
            for curLayer in range(m):
                prevResult2D = tf.reshape(gResult, shape=(self.batch_size, inputShape[1], inputShape[1], prevResultDim))
                sumJ = tf.reduce_sum(prevResult2D, axis=2)
                sumK = tf.reduce_sum(prevResult2D, axis=1)
                #intermedShape = tf.shape(prevResult2D)#[0] is batch_size, [1] is obj_count, [2] is obj_count, [3] is g_dim (for curLayer=0) or h_dim (for curLayer>0)
                intermedObjPairs2D = getHeteroCombinations(sumJ, sumK)
                intermedObjPairs = tf.reshape(intermedObjPairs2D, shape=(-1, 2*prevResultDim))
                hResult = build_h(intermedObjPairs, questionRep)
                prevResult = hResult
                prevResultDim = h_dim
            hResult1D = tf.reshape(prevResult, shape=(self.batch_size, inputShape[1]*inputShape[1], prevResultDim))
            hSum = tf.reduce_sum(hResult1D, axis=1)
            result = build_f(hSum, question)

        return result

    #process questions and context sentences through LSTMs and use the final LSTM states as input question and objects for the Relation Networks
    def buildWordProcessorLSTMs(self):
        with tf.name_scope('wordProcessorLSTMs'):
            #model parameters
            embeddingDimension = 256#128
            qLstmHiddenUnits = self.question_dim
            if self.appendPosVec:
                posVecDim = 32
                sLstmHiddenUnits = self.obj_dim - posVecDim
            else:
                posVecDim = self.obj_dim
                sLstmHiddenUnits = self.obj_dim

            inputContext = tf.placeholder(tf.int32, shape=(self.batch_size, None, None))
            inputContextLengths = tf.placeholder(tf.int32, shape=(self.batch_size,))#Number of sentences in each context
            inputContextSentenceLengths = tf.placeholder(tf.int32, shape=(self.batch_size, None))#Number of words in each sentence
            inputQuestion = tf.placeholder(tf.int32, shape=(self.batch_size, None))
            inputQuestionLengths = tf.placeholder(tf.int32, shape=(self.batch_size,))

            #convert word indices to embedded representations (using learnable embeddings rather than one-hot vectors here)
            wordEmbedding = tf.Variable(tf.random_uniform(shape=[self.dictSize, embeddingDimension], minval=-1, maxval=1, seed=7))
            embeddedQuestion = tf.nn.embedding_lookup(wordEmbedding, inputQuestion)

            #setup question LSTM
            questionLSTMcell = tf.nn.rnn_cell.LSTMCell(num_units=qLstmHiddenUnits)
            questionLSTMoutputs, _ = tf.nn.dynamic_rnn(questionLSTMcell, embeddedQuestion, dtype=tf.float32, scope="questionLSTM")#shape=(batch_size, seq_len, qLstmHiddenUnits)
            #extract final states at the end of each sample's sequence
            inputQuestionMaxLength = tf.reduce_max(inputQuestionLengths)
            questionLSTMoutputs = tf.reshape(questionLSTMoutputs, shape=(-1, qLstmHiddenUnits))
            qSeqEndSelector = tf.range(self.batch_size) * inputQuestionMaxLength + (inputQuestionLengths - 1)
            questionLSTMoutputs = tf.gather(questionLSTMoutputs, qSeqEndSelector)#shape=(batch_size, qLstmHiddenUnits)

            #setup sentence LSTM
            inputContextMaxLength = tf.reduce_max(inputContextLengths)
            inputContextSentenceMaxLength = tf.reduce_max(inputContextSentenceLengths)
            inputSentences = tf.reshape(inputContext, shape=(self.batch_size*inputContextMaxLength, inputContextSentenceMaxLength))
            embeddedSentences = tf.nn.embedding_lookup(wordEmbedding, inputSentences)#shape=(batch_size*contextMaxLength, seq_len, embeddingDimension)
            #do we want to broadcast the question to the sentence LSTM here? or rather leave it entirely to the relation network
            #START VARIANTS
            if self.questionAwareContext:
                # a) variant WITH broadcasting the questionLSTMoutputs to all sentences and timesteps (words/tokens):
                broadcastedQuestionLSTMoutputs = tf.expand_dims(questionLSTMoutputs, axis=1)#add time axis
                broadcastedQuestionLSTMoutputs = tf.tile(broadcastedQuestionLSTMoutputs, [inputContextMaxLength, inputContextSentenceMaxLength, 1])#repeat along time axis
                sentenceLSTMcell = tf.nn.rnn_cell.LSTMCell(num_units=sLstmHiddenUnits)
                sentenceLSTMoutputs, _ = tf.nn.dynamic_rnn(sentenceLSTMcell, tf.concat([embeddedSentences, broadcastedQuestionLSTMoutputs], axis=2), dtype=tf.float32, scope="sentenceLSTM")#shape=(batch_size*contextMaxLength, seq_len, sLstmHiddenUnits)
            else:
                # b) variant WITHOUT broadcasting the questionLSTMoutputs to all sentences and timesteps (words/tokens):
                sentenceLSTMcell = tf.nn.rnn_cell.LSTMCell(num_units=sLstmHiddenUnits)
                sentenceLSTMoutputs, _ = tf.nn.dynamic_rnn(sentenceLSTMcell, embeddedSentences, dtype=tf.float32, scope="sentenceLSTM")#shape=(batch_size*contextMaxLength, seq_len, sLstmHiddenUnits)
            #END VARIANTS
            #extract final states at the end of each sentence's sequence
            sentenceLSTMoutputs = tf.reshape(sentenceLSTMoutputs, shape=(-1, sLstmHiddenUnits))
            inputContextSentenceLengths1D = tf.reshape(inputContextSentenceLengths, shape=(-1,))
            sSeqEndSelector = tf.range(self.batch_size*inputContextMaxLength) * inputContextSentenceMaxLength + (inputContextSentenceLengths1D - 1)
            sentenceLSTMoutputs = tf.gather(sentenceLSTMoutputs, sSeqEndSelector)#shape=(batch_size*contextMaxLength, sLstmHiddenUnits)
            sentenceLSTMoutputs = tf.reshape(sentenceLSTMoutputs, shape=(self.batch_size, inputContextMaxLength, sLstmHiddenUnits))#these are the objects for the relation network input

            #tag sentences with position encodings to give information about their order of occurrance in the context (position encoding: https://arxiv.org/pdf/1503.08895.pdf)
            tagMatrixJ = tf.range(1, tf.cast(inputContextMaxLength + 1, tf.float32), dtype=tf.float32) / tf.cast(inputContextMaxLength, tf.float32)
            tagMatrixK = tf.range(1, posVecDim + 1, dtype=tf.float32) / posVecDim
            tagMatrixTerm1 = 1 - tf.tile(tf.expand_dims(tagMatrixJ, axis=1), [1, posVecDim])#shape=(inputContextMaxLength, posVecDim)
            tagMatrixTerm2 = tf.tile(tf.expand_dims(tagMatrixK, axis=0), [inputContextMaxLength, 1])#shape=(inputContextMaxLength, posVecDim)
            tagMatrixTerm3 = tagMatrixTerm1 * 2 - 1
            tagMatrix = tagMatrixTerm1 - tagMatrixTerm2 * tagMatrixTerm3#shape=(inputContextMaxLength, posVecDim)
            tagMatrix = tf.tile(tf.expand_dims(tagMatrix, axis=0), [self.batch_size, 1, 1])#shape=(batch_size, inputContextMaxLength, posVecDim)
            if self.appendPosVec:
                sentenceLSTMoutputs = tf.concat([sentenceLSTMoutputs, tagMatrix], axis=2)
            else:
                sentenceLSTMoutputs = sentenceLSTMoutputs * tagMatrix

            #TODO: optimization: dont apply the LSTMs to the padding-sentences (empty sentences added to make all contexts in a batch the same sentence-count)

        #return inputContext, inputContextLengths, inputQuestion, inputQuestionLengths, answer, answerGates
        return inputContext, inputContextLengths, inputContextSentenceLengths, inputQuestion, inputQuestionLengths, sentenceLSTMoutputs, questionLSTMoutputs

    def buildAnswerModel(self, prevNetworkOutput):
        with tf.name_scope('answerModel'):
            #tf.nn.softmax removed because it's applied afterwards by the built-in loss function softmax_cross_entropy_with_logits
            #no relu before softmax!
            # answer1 = tf.contrib.layers.fully_connected(prevNetworkOutput, self.dictSize, activation_fn=None)#, activation_fn=tf.nn.relu)#tf.nn.softmax)#shape=(batch_size, dictSize)
            # answer2 = tf.contrib.layers.fully_connected(prevNetworkOutput, self.dictSize, activation_fn=None)#, activation_fn=tf.nn.relu)#tf.nn.softmax)
            # answer3 = tf.contrib.layers.fully_connected(prevNetworkOutput, self.dictSize, activation_fn=None)#, activation_fn=tf.nn.relu)#tf.nn.softmax)
            # #TODO: either make sure prevNetworkOutput doesnt end with a relu or put a relu-free layer before answerGates
            # #answerGates = tf.contrib.layers.fully_connected(prevNetworkOutput, 3, activation_fn=tf.sigmoid)#shape=(batch_size, 3)
            # answerStack = tf.stack([answer1, answer2, answer3], axis=1)#stack shape=(batch_size, 3, dictSize)
            # #answer = tf.reduce_sum(tf.multiply(answerStack, tf.expand_dims(answerGates, axis=2)), axis=1)
            # answer = tf.reduce_mean(answerStack, axis=1)
            answer = tf.contrib.layers.fully_connected(prevNetworkOutput, self.dictSize, activation_fn=None)

            # maxIndices = tf.argmax(answerStack, axis=2)#shape=(batch_size, 3) #equivalent to tf.argmax(answerSoftmaxStack, ...) because of softmax' monotonicity
            # answersOneHot = tf.one_hot(maxIndices, self.dictSize)#shape=(batch_size, 3, dictSize)
            # binaryGates = tf.round(answerGates)#shape=(batch_size, 3)
            # answerForCorrectness = tf.reduce_sum(tf.multiply(answersOneHot, tf.expand_dims(binaryGates, axis=2)), axis=1)#shape=(batch_size, dictSize)


            answerSoftmax = tf.nn.softmax(answer) * 3
            maxIndex = tf.argmax(answerSoftmax, axis=1)
            answerForCorrectness1 = tf.one_hot(maxIndex, self.dictSize)

            answerSoftmax23 = answerSoftmax - answerForCorrectness1
            maxIndex2 = tf.argmax(answerSoftmax23, axis=1)
            answerForCorrectness2 = tf.one_hot(maxIndex2, self.dictSize)

            answerSoftmax3 = answerSoftmax23 - answerForCorrectness2
            maxIndex3 = tf.argmax(answerSoftmax3, axis=1)
            answerForCorrectness3 = tf.one_hot(maxIndex3, self.dictSize)

            answerForCorrectness = answerForCorrectness1 + answerForCorrectness2 + answerForCorrectness3

        #return answer, answerGates, answerForCorrectness
        return answer, answerForCorrectness

    def buildOptimizer(self, answer, optimizerAlg): #, answerGates):
        with tf.name_scope('optimizer'):
            global_step_tensor = tf.Variable(0, trainable=False, name='global_step')
            inputAnswer = tf.placeholder(tf.float32, shape=(self.batch_size, self.dictSize))#label
            learningRate = tf.placeholder(tf.float32, shape=())
            inputAnswerForLoss = inputAnswer / 3#since they represent the sum of 3 one-hot vectors, normalize to make them a probability distribution for the loss
            #loss = tf.losses.mean_squared_error(labels=inputAnswer, predictions=answer) * dictSize - tf.reduce_mean(tf.square(answerGates - 0.5)) + 0.25#regularization term to enforce gate values close to 0 or 1
            loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=answer, labels=inputAnswerForLoss))# - tf.reduce_mean(tf.square(answerGates - 0.5)) + 0.25#regularization term to enforce gate values close to 0 or 1
            #softmax_cross_entropy_with_logits is not suitable for outputs that are not probability distributions (which might be a problem for multi-answer questions) - still gives surprisingly good results for a first attempt
            if optimizerAlg == 'nesterov':
                optimizer = tf.train.MomentumOptimizer(learningRate, 0.9, use_nesterov=True)
                print('Using nesterov')
            else:
                optimizer = tf.train.AdamOptimizer(learningRate)
                print('Using adam')
            #optimizer = tf.train.AdamOptimizer(1e-5)
            tvs = tf.trainable_variables()
            accum_vars = [tf.Variable(tf.zeros_like(tv.initialized_value()), trainable=False) for tv in tvs]
            zero_ops = [tv.assign(tf.zeros_like(tv)) for tv in accum_vars]
            #gradient clipping
            gradients, variables = zip(*optimizer.compute_gradients(loss, tvs))
            gradientsNorm = tf.global_norm(gradients)#for logging purposes - keep this line before clipping
            gradients, _ = tf.clip_by_global_norm(gradients, 250.0)#threshold selected as average norm * 5
            #gradient accumulation
            accum_ops = [accum_vars[i].assign_add(gv) for i, gv in enumerate(gradients)]
            train_step = optimizer.apply_gradients([(accum_vars[i], tv) for i, tv in enumerate(variables)], global_step=global_step_tensor)
            #TODO: average gradient over macro_batches?
            #optimizer_op = optimizer.apply_gradients(zip(gradients, variables), global_step=global_step_tensor)
            #optimizer_op = tf.train.AdamOptimizer(1e-5).minimize(loss)#without gradient clipping

        #return inputAnswer, loss, optimizer_op, global_step_tensor, gradientsNorm, learningRate
        return inputAnswer, loss, accum_ops, zero_ops, train_step, global_step_tensor, gradientsNorm, learningRate