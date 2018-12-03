import DomainAdaptation_Initialization as DA_init
import numpy as np

sourceDomainPath = '../SourceDomain/'
targetDomainPath = '../TargetDomain/'

experimentalPath = '../experiment_data/'


def gatherImages_Labels_NooverlapImages(rootPath, mode):
    pklFilePath = rootPath + mode + '/'

    image_label_pairs = []
    Nooverlap_images = []

    fileNameList = DA_init.getFileNameList(pklFilePath)

    for f in fileNameList:
        _img, _lab, _nooverlap_img = DA_init.loadPickle(pklFilePath, f)

        for _i, _l in zip(_img, _lab):
            image_label_pairs.append([_i, _l])
        Nooverlap_images.append(_nooverlap_img)

    sorted_pairs = DA_init.sortVariousPairs(image_label_pairs)

    img_lib, lab_lib = [], []

    for i in range(len(sorted_pairs)):
        img_lib.append(sorted_pairs[i][0])
        lab_lib.append(sorted_pairs[i][1])

    img_lib = np.array(img_lib, dtype=np.float32)
    lab_lib = np.array(lab_lib, dtype=np.int32)

    img_lib = np.expand_dims(img_lib, axis=3)
    lab_lib = DA_init.onehotEncoder(lab_lib, num_class=6)

    Nooverlap_images = np.concatenate(Nooverlap_images, axis=0)
    Nooverlap_images_lib = np.expand_dims(Nooverlap_images, axis=3)

    print('-' * 20 + mode + ' dataset process finish' + '-' * 20)
    print('Mode %s image lib shape: %s label lib shape: %s nooverlap images shape: %s' % (
        mode, str(img_lib.shape), str(lab_lib.shape), str(Nooverlap_images_lib.shape)))

    return img_lib, lab_lib, Nooverlap_images_lib


# 开始处理两个域的数据
print('start processing source domain data')

training_img, training_lab, training_nooverlap = gatherImages_Labels_NooverlapImages(sourceDomainPath,
                                                                                     mode='training')
validation_img, validation_lab, _ = gatherImages_Labels_NooverlapImages(sourceDomainPath, mode='validation')
test_img, test_lab, _ = gatherImages_Labels_NooverlapImages(sourceDomainPath, mode='test')

DA_init.savePickle([training_img, training_lab], experimentalPath, 'source_training.pkl')
DA_init.savePickle([validation_img, validation_lab], experimentalPath, 'source_validation.pkl')
DA_init.savePickle([test_img, test_lab], experimentalPath, 'source_test.pkl')
DA_init.savePickle(training_nooverlap, experimentalPath, 'source_target.pkl')

print('training image shape', str(training_img.shape))
print('training label shape', str(training_lab.shape))
print('training image mean/std', str(training_img.mean()), str(training_img.std()))

print('validation image shape', str(validation_img.shape))
print('validation label shape', str(validation_lab.shape))
print('validation image mean/std', str(validation_img.mean()), str(validation_img.std()))

print('test image shape', str(test_img.shape))
print('test label shape', str(test_lab.shape))
print('test label mean/std', str(test_img.mean()), str(test_img.std()))

print('training nooverlap image shape', str(training_nooverlap.shape))
print('training nooverlap image mean/std', str(training_nooverlap.mean()), str(training_nooverlap.std()))

print('start processing target domain data')

training_img, training_lab, training_nooverlap = gatherImages_Labels_NooverlapImages(targetDomainPath,
                                                                                     mode='training')
validation_img, validation_lab, _ = gatherImages_Labels_NooverlapImages(targetDomainPath, mode='validation')
test_img, test_lab, _ = gatherImages_Labels_NooverlapImages(targetDomainPath, mode='test')

DA_init.savePickle([training_img, training_lab], experimentalPath, 'target_training.pkl')
DA_init.savePickle([validation_img, validation_lab], experimentalPath, 'target_validation.pkl')
DA_init.savePickle([test_img, test_lab], experimentalPath, 'target_test.pkl')
DA_init.savePickle(training_nooverlap, experimentalPath, 'target_source.pkl')

print('training image shape', str(training_img.shape))
print('training label shape', str(training_lab.shape))
print('training image mean/std', str(training_img.mean()), str(training_img.std()))

print('validation image shape', str(validation_img.shape))
print('validation label shape', str(validation_lab.shape))
print('validation image mean/std', str(validation_img.mean()), str(validation_img.std()))

print('test image shape', str(test_img.shape))
print('test label shape', str(test_lab.shape))
print('test label mean/std', str(test_img.mean()), str(test_img.std()))

print('training nooverlap image shape', str(training_nooverlap.shape))
print('training nooverlap image mean/std', str(training_nooverlap.mean()), str(training_nooverlap.std()))
