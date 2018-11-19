import numpy as np
import sys
import os
import pickle
import argparse


def saveLog(message, filePath, fileName):
    if not os.path.isdir(filePath):
        os.makedirs(filePath)

    with open(filePath + fileName, 'a+') as f:
        f.write(message + '\n')


def loadPickle(pklFilePath, pklFileName):
    with open(pklFilePath + pklFileName, 'rb') as f:
        message = pickle.load(f)

    return message


def numImagesPerFile(image_array, label_array, logFilePath, logFileName, fileName, global_label_num):
    label_name = {0: 'Mul_CON',
                  1: 'Mul_GGO',
                  2: 'HCM',
                  3: 'EMP',
                  4: 'DIF_NOD',
                  5: 'NOR'}
    label_num = np.zeros([6])

    num = image_array.shape[0]
    for i in range(num):
        label_num[label_array[i]] += 1
        global_label_num[label_array[i]] += 1

    log = fileName + ' '
    for j in range(6):
        if label_num[j] == 0:
            continue
        else:
            log += label_name[j] + '[%d]' % label_num[j] + ' '

    saveLog(log, logFilePath, logFileName)


def totalNumImagesPerFile(label_num, logFilePath, logFileName):
    label_name = {0: 'Mul_CON',
                  1: 'Mul_GGO',
                  2: 'HCM',
                  3: 'EMP',
                  4: 'DIF_NOD',
                  5: 'NOR'}
    log = ''
    for j in range(6):
        log += label_name[j] + '[%d]\t' % label_num[j]

    print(log)
    saveLog(log, logFilePath, logFileName)


def getFileNameList(filePath):
    l = os.listdir(filePath)
    l = sorted(l, key=lambda x: x[:x.find('.')])

    return l


def summarizeNumImagesPerFile(fileList, global_label_num):
    for f in fileList:
        _img_array, _lab_array, _ = loadPickle(pklImagePath, f)
        numImagesPerFile(_img_array, _lab_array, logFilePath, logFileName, f, global_label_num)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', required=True)
    args = parser.parse_args()

    domain = args.d

    global_label_num = np.zeros(6)

    logFilePath = '../PKL_LogFile/'
    logFileName = 'Log_' + domain + '.txt'

    pklImagePath = '../Data_' + domain + 'Domain_PKL/'

    fileList = getFileNameList(pklImagePath)

    summarizeNumImagesPerFile(fileList, global_label_num)

    totalNumImagesPerFile(global_label_num, logFilePath, logFileName)
