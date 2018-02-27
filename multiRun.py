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
    [1],#run1
    [5],#run2
    [8, 3]#run3
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

        with open(os.path.join(logDir, logFileName), 'wb') as logFile:
            proc = subprocess.Popen(['python', 'train.py'] + [str(arg) for arg in runList[workItem]], env=workerEnv, stdout=logFile, stderr=logFile)
            proc.wait()

for i in gpuList:
    t = Thread(target=worker, args=(i,))
    t.start()
    threadList = threadList + [t]

for t in threadList:
    t.join()