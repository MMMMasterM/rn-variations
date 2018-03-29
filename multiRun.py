import os, sys, ast, subprocess, datetime
from threading import Thread, Lock

logDir = 'log'
try:
    os.stat(logDir)
except:
    os.mkdir(logDir)

gpuList = [0]

if len(sys.argv) >= 2:
    try:
        gpuList = [int(sys.argv[1])]
    except:
        gpuList = ast.literal_eval(sys.argv[1])

print("gpuList: " + str(gpuList))

#array of parameter lists to hand to train.py
runList = [
    [8, '--layers', 1, '--learningRate', 0.00005, '--g_layers', 4, '--f_layers', 2, '--appendPosVec', '--batch_size', 8],
    [8, '--layers', 1, '--learningRate', 0.00005, '--g_layers', 4, '--f_layers', 2, '--appendPosVec', '--batch_size', 16],
    [8, '--layers', 1, '--learningRate', 0.00005, '--g_layers', 4, '--f_layers', 2, '--appendPosVec', '--batch_size', 32],
    [8, '--layers', 1, '--learningRate', 0.00005, '--g_layers', 4, '--f_layers', 2, '--appendPosVec', '--batch_size', 64],
    #[8, '--layers', 1, '--learningRate', 0.000075, '--g_layers', 4, '--f_layers', 2, '--appendPosVec'],
    #[8, '--layers', 1, '--learningRate', 0.00005, '--g_layers', 4, '--f_layers', 2, '--appendPosVec', '--obj_dim', 512],
    #[8, '--layers', 1, '--learningRate', 0.00005, '--g_layers', 4, '--f_layers', 2, '--appendPosVec', '--obj_dim', 64, '--question_dim', 32],

    # [8, '--layers', 1, '--learningRate', 0.00005, '--g_layers', 4, '--f_layers', 2],
    # [8, '--layers', 1, '--learningRate', 0.00005, '--g_layers', 4, '--f_layers', 2, '--clr'],
    # [8, '--layers', 0, '--learningRate', 0.00005, '--g_layers', 4, '--f_layers', 2],
    # [8, '--layers', 0, '--learningRate', 0.00005, '--g_layers', 4, '--f_layers', 2, '--clr'],
    # [8, '--layers', 1, '--learningRate', 0.00005, '--g_layers', 4, '--f_layers', 2, '--appendPosVec'],
    # [8, '--layers', 1, '--learningRate', 0.00005, '--g_layers', 4, '--f_layers', 2, '--obj_dim', 128],

    # [8, '--layers', 0, '--learningRate', 0.00005, '--questionAwareContext'],
    # [8, '--layers', 0, '--learningRate', 0.00005],
    # [8, '--layers', 1, '--learningRate', 0.00005],
    # [8, '--layers', 2, '--learningRate', 0.00005],
    # [8, '--layers', 3, '--learningRate', 0.00005],

    # [8, '--layers', 1, '--learningRate', 0.00005, '--f_layers', 2],
    # [8, '--layers', 1, '--learningRate', 0.00005, '--f_layers', 4],
    

    #[1],#run1
    #[8, 3],#run2
    #[5],#run3
    #[2]#run4
]

nextWorkItem = 0
workItemLock = Lock()

threadList = []

parentEnv = dict(os.environ)

def worker(gpuNumber):
    global nextWorkItem
    workerEnv = parentEnv.copy()
    workerEnv["CUDA_VISIBLE_DEVICES"] = str(gpuNumber)
    while True:
        workItemLock.acquire()
        workItem = nextWorkItem
        nextWorkItem += 1
        workItemLock.release()

        if workItem >= len(runList):
            break

        logFileName = datetime.datetime.now().strftime("%F_%H_%M_%S") + "__" + str(gpuNumber) + ".txt"
        print("Running run parameters " + str(runList[workItem]) + " on " + str(gpuNumber) + " logFile " + logFileName)

        with open(os.path.join(logDir, logFileName), 'w') as logFile:
            proc = subprocess.Popen(['python', '-u', 'train.py'] + [str(arg) for arg in runList[workItem]], env=workerEnv, stdout=logFile, stderr=logFile)
            proc.wait()

        print("Run " + str(runList[workItem]) + " on " + str(gpuNumber) + " finished")

for i in gpuList:
    t = Thread(target=worker, args=(i,))
    t.start()
    threadList = threadList + [t]

for t in threadList:
    t.join()