import sys

sys.path.append('../Data_Initialization/')
import os
from network_model1 import Image_Generator_Model
import DomainAdaptation_Initialization as DA_init
import argparse
import tensorflow as tf
import utils_model1

parser = argparse.ArgumentParser()
parser.add_argument('-model_name', required=True, help='[the name of the model]')
parser.add_argument('-train_phase', required=True, help='[whether to train or test the model]')
parser.add_argument('-gpu', required=True, help='[set particular gpu for calculation]')
parser.add_argument('-data_domain', required=True, help='[choose the data domain between source and target]')

parser.add_argument('-epoch', default=100, type=int)
parser.add_argument('-restore_epoch', default=0, type=int)
parser.add_argument('-learning_rate', default=2e-4, type=float)
parser.add_argument('-batch_size', default=128, type=int)
parser.add_argument('-img_height', default=32, type=int)
parser.add_argument('-img_width', default=32, type=int)
args = parser.parse_args()

os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

if args.data_domain == 'Source':
    src_name = 'source'
    tar_name = 'target'
elif args.data_domain == 'Target':
    src_name = 'target'
    tar_name = 'source'
else:
    src_name = ''
    tar_name = ''

src_training = DA_init.loadPickle(utils_model1.experimentalPath, src_name + '_training.pkl')
src_validation = DA_init.loadPickle(utils_model1.experimentalPath, src_name + '_validation.pkl')
src_test = DA_init.loadPickle(utils_model1.experimentalPath, src_name + '_test.pkl')

tar_training = DA_init.loadPickle(utils_model1.experimentalPath, tar_name + '_' + src_name + '.pkl')
tar_test = DA_init.loadPickle(utils_model1.experimentalPath, tar_name + '_test.pkl')

print('source training image shape', str(src_training[0].shape))
print('source training label shape', src_training[1].shape)
print('source training image mean/std', str(src_training[0].mean()), str(src_training[0].std()))

print('source validation image shape', str(src_validation[0].shape))
print('source validation label shape', src_validation[1].shape)
print('source validation image mean/std', str(src_validation[0].mean()), str(src_validation[0].std()))

print('source test image shape', src_test[0].shape)
print('source test label shape', src_test[1].shape)
print('source test image mean/std', str(src_test[0].mean()), str(src_test[0].std()))

print('target training image shape', str(tar_training.shape))
print('target training image mean/std', str(tar_training.mean()), str(tar_training.std()))

print('target test image shape', tar_test[0].shape)
print('target test label shape', tar_test[1].shape)
print('target test image mean/std', str(tar_test[0].mean()), str(tar_test[0].std()))

config = tf.ConfigProto()
config.gpu_options.allow_growth = True

with tf.Session(config=config) as sess:
    res_model = Image_Generator_Model(model_name=args.model_name,
                                      sess=sess,
                                      train_data=[src_training, tar_training],
                                      val_data=[src_validation],
                                      tst_data=[src_test, tar_test],
                                      epoch=args.epoch,
                                      restore_epoch=args.restore_epoch,
                                      learning_rate=args.learning_rate,
                                      batch_size=args.batch_size,
                                      img_height=args.img_height,
                                      img_width=args.img_width,
                                      train_phase=args.train_phase)

    if args.train_phase == 'Train':
        res_model.train()

    if args.train_phase == 'Test':
        pass
