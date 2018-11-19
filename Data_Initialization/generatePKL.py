import DomainAdaptation_Initialization as DA_init

sourceDomainPath = '../SourceDomain/'
targetDomainPath = '../TargetDomain/'

experimentalPath = '../experiment_data/'

print('start processing source domain data')

training_img, training_lab, training_nooverlap = DA_init.gatherImages_Labels_NooverlapImages(sourceDomainPath,
                                                                                             mode='training')
validation_img, validation_lab, _ = DA_init.gatherImages_Labels_NooverlapImages(sourceDomainPath, mode='validation')
test_img, test_lab, _ = DA_init.gatherImages_Labels_NooverlapImages(sourceDomainPath, mode='test')

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

DA_init.savePickle([training_img, training_lab], experimentalPath, 'source_training.pkl')
DA_init.savePickle([validation_img, validation_lab], experimentalPath, 'source_validation.pkl')
DA_init.savePickle([test_img, test_lab], experimentalPath, 'source_test.pkl')
DA_init.savePickle(training_nooverlap, experimentalPath, 'source_target.pkl')

print('start processing target domain data')

training_img, training_lab, training_nooverlap = DA_init.gatherImages_Labels_NooverlapImages(targetDomainPath,
                                                                                             mode='training')
validation_img, validation_lab, _ = DA_init.gatherImages_Labels_NooverlapImages(targetDomainPath, mode='validation')
test_img, test_lab, _ = DA_init.gatherImages_Labels_NooverlapImages(targetDomainPath, mode='test')

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

DA_init.savePickle([training_img, training_lab], experimentalPath, 'target_training.pkl')
DA_init.savePickle([validation_img, validation_lab], experimentalPath, 'target_validation.pkl')
DA_init.savePickle([test_img, test_lab], experimentalPath, 'target_test.pkl')
DA_init.savePickle(training_nooverlap, experimentalPath, 'target_source.pkl')