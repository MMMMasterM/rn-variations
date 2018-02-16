import os, re, itertools
import numpy as np
import pickle

print("Preprocessing dataset...")

dataPath = os.path.join('tasks_1-20_v1-2', 'en-valid-10k')
dataFiles = [curEntry for curEntry in os.listdir(dataPath) if os.path.isfile(os.path.join(dataPath, curEntry))]
trainPaths = {}
testPaths = {}
validPaths = {}

for filename in dataFiles:
    regMatch = re.match(r'^(qa\d+)_(train|test|valid)\.txt$', filename)
    if regMatch is not None:
        if regMatch.group(2) == 'train':
            trainPaths[regMatch.group(1)] = os.path.join(dataPath, filename)
        elif regMatch.group(2) == 'test':
            testPaths[regMatch.group(1)] = os.path.join(dataPath, filename)
        else:
            validPaths[regMatch.group(1)] = os.path.join(dataPath, filename)

# build dictionary for all the words
wordset = set({'.'})
lineFormatReg = re.compile(r'^\d+( ([a-zA-Z]+))+(\.|(\? ?\t([a-zA-Z]+,)*[a-zA-Z]+\t(\d+ )*\d+))$')
for filePath in itertools.chain(trainPaths.values(), testPaths.values(), validPaths.values()):
    with open(filePath) as f:
        for line in f:
            #sanity check succeeded -> for now commented for performance
            # if lineFormatReg.match(line) is None:#Check if the expected format really applies
            #     print("Warning: Unexpected line format")
            #     print(filePath)
            #     print(line)

            line = line.lower()
            parts = line.split('\t')
            words = parts[0].strip('? .\n').split(' ')[1:]
            wordset.update(words)
            
            if len(parts) > 1:#add words from the answer to the wordset
                words = parts[1].split(',')
                wordset.update(words)

wordIndices = {}
for i, word in enumerate(wordset):
    wordIndices[word] = i + 1#reserve 0 for zero-padding sequences

# save dictionary
with open(os.path.join('processeddata', 'dictionary.txt'), 'wb') as f:
    pickle.dump(wordIndices, f)

def linesFromFilepaths(filepathIter):
    for filepath in filepathIter:
        with open(filepath) as f:
            for line in f:
                yield line

def paragraphsFromLines(linesIter):
    paragraph = []
    for line in linesIter:
        lineNum = line.split(' ', 1)[0]
        if lineNum == '1' and len(paragraph) > 0:
            yield paragraph
            paragraph = []
        paragraph.append(line)
    if len(paragraph) > 0:
        yield paragraph

def cqaFromParagraphs(paragraphIter):
    for paragraph in paragraphIter:
        context = []
        for line in paragraph:
            line = line.lower()
            parts = line.split('\t')
            if (len(parts) == 1):#context
                words = line.strip('.\n').split(' ')[1:]
                tokens = [wordIndices[word] for word in words]
                context.append(tokens)
            else:#question/answer/hints
                #question
                words = parts[0].strip('? ').split(' ')[1:]
                question = [wordIndices[word] for word in words]
                #answer
                words = parts[1].split(',')
                tokens = [wordIndices[word] for word in words]
                answerVecs = np.zeros(shape=(len(tokens), len(wordset) + 1))
                for i, token in enumerate(tokens):
                    answerVecs[i, token] = 1
                answer = np.sum(answerVecs, axis=0)
                #add "." tokens at the end of each sentence
                terminatedContext = [sentence + [wordIndices['.']] for sentence in context]
                yield terminatedContext, question, answer

wordDict = {}
for word, num in wordIndices.items():
    wordDict[num] = word

#generate training dataset
trainingData = []
for cqa in cqaFromParagraphs(paragraphsFromLines(linesFromFilepaths(trainPaths.values()))):
    #Sanity check succeeded -> commented
    # print(cqa)
    # context, question, answer = cqa
    # for sentence in context:
    #     print([wordDict[token] for token in sentence] + ['.'])
    # print([wordDict[token] for token in question] + ['?'])
    # print('answer:')
    # for word, num in wordIndices.items():
    #     if answer[num] > 0:
    #         print(word)
    # break

    trainingData.append(cqa)

#generate test dataset
testData = {}
for taskname, filepath in testPaths.items():
    taskData = []
    for cqa in cqaFromParagraphs(paragraphsFromLines(linesFromFilepaths([filepath]))):
        taskData.append(cqa)
    testData[taskname] = taskData

#generate validation dataset
validData = {}
for taskname, filepath in validPaths.items():
    taskData = []
    for cqa in cqaFromParagraphs(paragraphsFromLines(linesFromFilepaths([filepath]))):
        taskData.append(cqa)
    validData[taskname] = taskData


#TODO: apply padding?

outDir = 'processeddata'

#create directory if it doesnt exist
try:
    os.stat(outDir)
except:
    os.mkdir(outDir)

#save training dataset
with open(os.path.join(outDir, 'train.txt'), 'wb') as f:
    pickle.dump(trainingData, f)

#save test dataset
with open(os.path.join(outDir, 'test.txt'), 'wb') as f:
    pickle.dump(testData, f)

#save validation dataset
with open(os.path.join(outDir, 'valid.txt'), 'wb') as f:
    pickle.dump(validData, f)

print("Preprocessing finished")