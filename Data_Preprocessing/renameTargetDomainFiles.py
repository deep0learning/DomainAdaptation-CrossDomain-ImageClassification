# 目标域的数据集文件名混乱，通过label图像中的标记类别进行重命名，
# 将无类别标记的数据以及多类别标记的数据移动到其他的对应文件夹分开存放，只使用单类别标记数据

import os
import shutil
import SimpleITK as sitk


def renameFile(srcPath, srcName, tarPath, tarName):
    os.rename(srcPath + srcName, tarPath + tarName)


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


def getClassOfVolume(imageFilePath, diseaseFilePath, segmentationFilePath, fileName):
    img_file_name = fileName
    hdr_file_name = fileName[:fileName.find('.')] + '.hdr'

    labelArray = readCTVolume(diseaseFilePath, img_file_name)
    labelLib = labelArray[labelArray > 0]
    labelLib = list(set(labelLib))

    if len(labelLib) == 0:
        shutil.move(imageFilePath + fileName, emptyClassImageFilePath + img_file_name)
        shutil.move(diseaseFilePath + fileName, emptyClassDiseaseFilePath + img_file_name)
        shutil.move(segmentationFilePath + fileName, emptyClassSegmentationFilePath + img_file_name)

        shutil.move(imageFilePath + hdr_file_name, emptyClassImageFilePath + hdr_file_name)
        shutil.move(diseaseFilePath + hdr_file_name, emptyClassDiseaseFilePath + hdr_file_name)
        shutil.move(segmentationFilePath + hdr_file_name, emptyClassSegmentationFilePath + hdr_file_name)
        print('%s file has been moved to empty directory' % fileName)

        return 'EMPTY'

    elif len(labelLib) > 1:
        shutil.move(imageFilePath + fileName, multiClassImageFilePath + img_file_name)
        shutil.move(diseaseFilePath + fileName, multiClassDiseaseFilePath + img_file_name)
        shutil.move(segmentationFilePath + fileName, multiClassSegmentationFilePath + img_file_name)

        shutil.move(imageFilePath + hdr_file_name, multiClassImageFilePath + hdr_file_name)
        shutil.move(diseaseFilePath + hdr_file_name, multiClassDiseaseFilePath + hdr_file_name)
        shutil.move(segmentationFilePath + hdr_file_name, multiClassSegmentationFilePath + hdr_file_name)
        print('%s file has been moved to multi-class directory' % fileName)

        return 'MULTI-CLASS'

    else:
        return labelLib[0]


def moveIndependentFiles(imageFilePath, diseaseFilePath, segmentationFilePath, independentFilePath):
    independent_img_path = independentFilePath + 'image/'
    independent_seg_path = independentFilePath + 'segmentation/'

    if not os.path.isdir(independent_img_path):
        os.makedirs(independent_img_path)
    if not os.path.isdir(independent_seg_path):
        os.makedirs(independent_seg_path)

    imageFileList = getFileNameList(imageFilePath)
    diseaseileList = getFileNameList(diseaseFilePath)

    for img_file in imageFileList:
        if img_file not in diseaseileList:
            shutil.move(imageFilePath + img_file, independent_img_path + img_file)
            shutil.move(segmentationFilePath + img_file, independent_seg_path + img_file)
            print('Independent File %s has been moved to independent directory' % img_file)


def renameTargetDomainFiles(ImageFilePath, DiseaseFilePath, SegmentationFilePath):
    fileClasses = {1: 'Mul_CON',
                   2: 'Mul_GGO',
                   3: 'HCM',
                   4: 'Ret_GGO',
                   5: 'EMP',
                   6: 'DIF_NOD',
                   10: 'NOR'}

    classCounter = {'Mul_CON': 1,
                    'Mul_GGO': 1,
                    'HCM': 1,
                    'Ret_GGO': 1,
                    'EMP': 1,
                    'DIF_NOD': 1,
                    'NOR': 1}

    fileList = getFileNameList(ImageFilePath)
    fileList = getSpecificTypeFileList(fileList, fileType='.gz')

    for f in fileList:
        _label = getClassOfVolume(ImageFilePath, DiseaseFilePath, SegmentationFilePath, f)
        if _label == 'MULTI-CLASS' or _label == 'EMPTY':
            continue
        else:
            fileNum = classCounter[fileClasses[_label]]
            if fileNum > 0 and fileNum <= 9:
                img_f_name = '00' + str(fileNum) + '.img.gz'
                hdr_f_name = '00' + str(fileNum) + '.hdr'

            elif fileNum >= 10 and fileNum <= 99:
                img_f_name = '0' + str(fileNum) + '.img.gz'
                hdr_f_name = '0' + str(fileNum) + '.hdr'

            else:
                img_f_name = str(fileNum) + '.img.gz'
                hdr_f_name = str(fileNum) + '.hdr'

            renamed_img_name = fileClasses[_label] + '_' + img_f_name
            renamed_hdr_name = fileClasses[_label] + '_' + hdr_f_name

            renameFile(ImageFilePath, f, ImageFilePath, renamed_img_name)
            renameFile(DiseaseFilePath, f, DiseaseFilePath, renamed_img_name)
            renameFile(SegmentationFilePath, f, SegmentationFilePath, renamed_img_name)

            renameFile(ImageFilePath, f.strip('.img.gz') + '.hdr', ImageFilePath, renamed_hdr_name)
            renameFile(DiseaseFilePath, f.strip('.img.gz') + '.hdr', DiseaseFilePath, renamed_hdr_name)
            renameFile(SegmentationFilePath, f.strip('.img.gz') + '.hdr', SegmentationFilePath, renamed_hdr_name)
            classCounter[fileClasses[_label]] += 1
            print(f + ' has been renamed to ' + renamed_img_name)

    print('File Renaming Finish')


if __name__ == '__main__':
    srcImageFilePath = '../Domain_Adaptation_Data/targetDomain/isoCT/'
    srcDiseaseFilePath = '../Domain_Adaptation_Data/targetDomain/isoMask/'
    srcSegmentationFilePath = '../Domain_Adaptation_Data/targetDomain/isoSegMask/'

    multiClassImageFilePath = '../Domain_Adaptation_Data/targetDomain/multiClass/Image/'
    multiClassDiseaseFilePath = '../Domain_Adaptation_Data/targetDomain/multiClass/Disease_Mask/'
    multiClassSegmentationFilePath = '../Domain_Adaptation_Data/targetDomain/multiClass/Segmentation_Mask/'

    emptyClassImageFilePath = '../Domain_Adaptation_Data/targetDomain/emptyClass/Image/'
    emptyClassDiseaseFilePath = '../Domain_Adaptation_Data/targetDomain/emptyClass/Disease_Mask/'
    emptyClassSegmentationFilePath = '../Domain_Adaptation_Data/targetDomain/emptyClass/Segmentation_Mask/'

    if not os.path.isdir(multiClassImageFilePath):
        os.makedirs(multiClassImageFilePath)
    if not os.path.isdir(multiClassDiseaseFilePath):
        os.makedirs(multiClassDiseaseFilePath)
    if not os.path.isdir(multiClassSegmentationFilePath):
        os.makedirs(multiClassSegmentationFilePath)

    if not os.path.isdir(emptyClassImageFilePath):
        os.makedirs(emptyClassImageFilePath)
    if not os.path.isdir(emptyClassDiseaseFilePath):
        os.makedirs(emptyClassDiseaseFilePath)
    if not os.path.isdir(emptyClassSegmentationFilePath):
        os.makedirs(emptyClassSegmentationFilePath)

    independentFilePath = '../Domain_Adaptation_Data/targetDomain/independentFiles/'

    moveIndependentFiles(srcImageFilePath, srcDiseaseFilePath, srcSegmentationFilePath, independentFilePath)

    renameTargetDomainFiles(srcImageFilePath, srcDiseaseFilePath, srcSegmentationFilePath)
