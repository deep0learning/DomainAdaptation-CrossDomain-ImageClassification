import os
import sys
import numpy as np
import SimpleITK as sitk
import pickle
import shutil
from skimage import io


def CTWindow(image):
    CT_LEVEL = -650
    CT_WINDOW = 1500

    pixelValue_Max = CT_LEVEL + CT_WINDOW // 2
    pixelValue_Min = CT_LEVEL - CT_WINDOW // 2

    image[image > pixelValue_Max] = pixelValue_Max
    image[image < pixelValue_Min] = pixelValue_Min

    image = (image - pixelValue_Min) / (pixelValue_Max - pixelValue_Min) * 255
    image = image.astype(np.int32)

    return image


# def visualizeImage(imageSlice, savePath):
#     pixel_max = imageSlice.max()
#     pixel_min = imageSlice.min()
#
#     print(pixel_max)
#     print(pixel_min)
#
#     size_y, size_x = imageSlice.shape
#
#     for i in range(size_y):
#         for j in range(size_x):
#             imageSlice[i][j] = (imageSlice[i][j] - pixel_min) / (pixel_max - pixel_min) * 1.0
#
#     io.imsave(savePath, imageSlice)


def getFileNameList(filePath):
    l = os.listdir(filePath)
    l = sorted(l, key=lambda x: x[:x.find('.')])

    return l


def getSpecificTypeFileList(fileList, fileType):
    fileList_return = []
    for f in fileList:
        if f[f.rfind('.'):] == fileType:
            fileList_return.append(f)

    return fileList_return


def readCTVolume(CTVolumeFilePath, CTVolumeFileName):
    volume = sitk.ReadImage(CTVolumeFilePath + CTVolumeFileName)
    volume_arr = sitk.GetArrayFromImage(volume)

    return volume_arr


def savePickle(input_array, pickleFilePath, pickleFileName):
    if not os.path.isdir(pickleFilePath):
        os.makedirs(pickleFilePath)

    with open(pickleFilePath + pickleFileName, 'wb') as f:
        pickle.dump(input_array, f)

    print('File %s has been successfully saved' % pickleFileName)


def loadPickle(pklFilePath, pklFileName):
    with open(pklFilePath + pklFileName, 'rb') as f:
        message = pickle.load(f)

    return message


def getCentralPointClass(array, centralPoint):
    return array[centralPoint[0], centralPoint[1]]


# def removeEmptyPKL(pklFilePath, pklLabelPath):
#     flist = getFileNameList(pklFilePath)
#     for f in flist:
#         message = loadPickle(pklFilePath, f)
#         if len(message) == 0:
#             os.remove(pklFilePath + f)
#             os.remove(pklLabelPath + f)
#             print('File %s has been removed for empty' % f)


def calculateActiveAreaOfMask(mask_array, threshold):
    size_y, size_x = mask_array.shape
    activePoint = 0
    for i in range(size_y):
        for j in range(size_x):
            if mask_array[i][j] != 0:
                activePoint += 1

    activeAreaPercentage = activePoint / (size_x * size_y)

    if activeAreaPercentage >= threshold:
        return True
    else:
        return False


def sortVariousPairs(pairList):
    return sorted(pairList, key=lambda x: x[1])


def splitImage_Label(pairWiseArray):
    num = len(pairWiseArray)
    split_image = []
    split_label = []

    for i in range(num):
        split_image.append(pairWiseArray[i][0])
        split_label.append(pairWiseArray[i][1])

    split_image = np.array(split_image, dtype=np.float32)
    split_label = np.array(split_label, dtype=np.int32)

    return split_image, split_label


def getPatch_DiseaseMask(imageArray, diseaseRegionArray, ROI_X, ROI_Y, ROI_Z, threshold, fileName):
    STRIDE = {'Mul_CON': 2,
              'Mul_GGO': 5,
              'HCM': 6,
              'EMP': 7,
              'DIF_NOD': 4,
              'NOR': 9
              }

    classIndicator = {1: 0,
                      2: 1,
                      3: 2,
                      5: 3,
                      6: 4,
                      10: 5}

    point_x_begin = 0
    point_y_begin = 0
    point_z_begin = 0

    image_label_list = []
    index_z_list = []

    fileClass = fileName[:fileName.find('.')]
    fileClass = fileClass[:fileClass.rfind('_')]

    size_z, size_y, size_x = imageArray.shape
    for z in range(point_z_begin, size_z, ROI_Z):
        for y in range(point_y_begin, size_y - ROI_Y, STRIDE[fileClass]):
            for x in range(point_x_begin, size_x - ROI_X, STRIDE[fileClass]):
                centralPoint = [y + ROI_Y // 2, x + ROI_X // 2]
                centralPointClass = getCentralPointClass(diseaseRegionArray[z, ...], centralPoint)
                if centralPointClass != 0 and centralPointClass != 4:
                    _ext_img = imageArray[z, y:y + ROI_Y, x:x + ROI_X]
                    _ext_msk = diseaseRegionArray[z, y:y + ROI_Y, x:x + ROI_X]
                    if calculateActiveAreaOfMask(_ext_msk, threshold=threshold):
                        _ext_img = CTWindow(_ext_img)
                        image_label_list.append([_ext_img, classIndicator[centralPointClass]])
                        index_z_list.append(z)

    index_z_list = list(set(index_z_list))
    print('Totally get %d images [specific diseases region]' % len(image_label_list))
    return image_label_list, index_z_list


def getPatch_SegmentationMask(imageArray, lungSegArray, ROI_X, ROI_Y, index_z_lib, threshold):
    point_x_begin = 0
    point_y_begin = 0

    image_list = []

    size_z, size_y, size_x = imageArray.shape
    for z in index_z_lib:
        for y in range(point_y_begin, size_y - ROI_Y, ROI_Y):
            for x in range(point_x_begin, size_x - ROI_X, ROI_X):
                centralPoint = [y + ROI_Y // 2, x + ROI_X // 2]
                centralPointClass = getCentralPointClass(lungSegArray[z, ...], centralPoint)
                if centralPointClass != 0:
                    _ext_img = imageArray[z, y:y + ROI_Y, x:x + ROI_X]
                    _ext_msk = lungSegArray[z, y:y + ROI_Y, x:x + ROI_X]
                    if calculateActiveAreaOfMask(_ext_msk, threshold=threshold):
                        _ext_img = CTWindow(_ext_img)
                        image_list.append(_ext_img)

    image_array = np.array(image_list, dtype=np.float32)
    print('Totally get %d images [lung segmentation region]' % len(image_list))
    return image_array


def initialization(fileList, ctFilePath, diseaseFilePath, segmentationFilePath, picklePath):
    for i in range(len(fileList)):
        if 'Ret' in fileList[i]:
            continue

        print('Process file : [%s]' % (fileList[i]))
        _image_volume = readCTVolume(ctFilePath, fileList[i])
        _mask_volume = readCTVolume(diseaseFilePath, fileList[i])
        _seg_volume = readCTVolume(segmentationFilePath, fileList[i])

        _image_label_pair_list, _index_z_list = getPatch_DiseaseMask(_image_volume, _mask_volume, ROI_X=32, ROI_Y=32,
                                                                     ROI_Z=1, threshold=0.5, fileName=fileList[i])

        if len(_image_label_pair_list) == 0:
            print('!' * 20 + 'File is empty, skip to the next file' + '!' * 20)
            continue

        imageSet_Nooverlap = getPatch_SegmentationMask(_image_volume, _seg_volume, ROI_X=32, ROI_Y=32,
                                                       index_z_lib=_index_z_list, threshold=0.5)

        image_label_pair_list = sortVariousPairs(_image_label_pair_list)
        imageSet, labelSet = splitImage_Label(image_label_pair_list)

        gathered_Set = [imageSet, labelSet, imageSet_Nooverlap]

        savePickle(gathered_Set, picklePath, fileList[i][:fileList[i].find('.')] + '.pkl')


if __name__ == '__main__':
    ctFilePath = '../Domain_Adaptation_Data/targetDomain/isoCT/'
    diseaseMaskFilePath = '../Domain_Adaptation_Data/targetDomain/isoMask/'
    lungSegMaskFilePath = '../Domain_Adaptation_Data/targetDomain/isoSegMask/'

    picklePath = '../Data_TargetDomain_PKL/'

    ctFileList = getFileNameList(ctFilePath)

    fileList = getSpecificTypeFileList(ctFileList, fileType='.gz')
    initialization(fileList, ctFilePath, diseaseMaskFilePath, lungSegMaskFilePath, picklePath)

    # 测试pickle文件
    # image = loadPickle('D:/Workspace/Domain_Adaptation/', 'DIF_NOD_009.pkl')
    #
    # img = image[0]
    # lab = image[1]
    # pics = image[2]
    # img = np.array(img)
    # lab = np.array(lab)
    # pics = np.array(pics)
    # print(img.shape)
    # print(lab.shape)
    # print(pics.shape)