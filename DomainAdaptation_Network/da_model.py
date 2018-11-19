import sys

sys.path.append('../Data_Initialization/')
import numpy as np
import tensorflow as tf
import tensorflow.contrib.layers as layers
import DomainAdaptation_Initialization as DA_init
import da_utils
import time


class DA_Model(object):
    def __init__(self, model_name, sess, train_data, val_data, tst_data, epoch, restore_epoch, num_class, ksize,
                 out_channel1, out_channel2, out_channel3, learning_rate, weight_decay, batch_size, img_height,
                 img_width, train_phase):

        self.sess = sess
        self.source_training_data = train_data[0]
        self.source_validation_data = val_data[0]
        self.source_test_data = tst_data[0]
        self.target_training_data = train_data[1]
        self.target_test_data = tst_data[1]
        self.eps = epoch
        self.res_eps = restore_epoch
        self.model = model_name
        self.ckptDir = '../checkpoint/' + self.model + '/'
        self.k = ksize
        self.oc1 = out_channel1
        self.oc2 = out_channel2
        self.oc3 = out_channel3
        self.lr = learning_rate
        self.wd = weight_decay
        self.bs = batch_size
        self.img_h = img_height
        self.img_w = img_width
        self.num_class = num_class
        self.train_phase = train_phase

        self.build_model()
        if self.train_phase == 'Train':
            self.saveConfiguration()

    def saveConfiguration(self):
        da_utils.save2file('epoch : %d' % self.eps, self.ckptDir, self.model)
        da_utils.save2file('model : %s' % self.model, self.ckptDir, self.model)
        da_utils.save2file('ksize : %d' % self.k, self.ckptDir, self.model)
        da_utils.save2file('out channel 1 : %d' % self.oc1, self.ckptDir, self.model)
        da_utils.save2file('out channel 2 : %d' % self.oc2, self.ckptDir, self.model)
        da_utils.save2file('out channel 3 : %d' % self.oc3, self.ckptDir, self.model)
        da_utils.save2file('learning rate : %g' % self.lr, self.ckptDir, self.model)
        da_utils.save2file('weight decay : %g' % self.wd, self.ckptDir, self.model)
        da_utils.save2file('batch size : %d' % self.bs, self.ckptDir, self.model)
        da_utils.save2file('image height : %d' % self.img_h, self.ckptDir, self.model)
        da_utils.save2file('image width : %d' % self.img_w, self.ckptDir, self.model)
        da_utils.save2file('num class : %d' % self.num_class, self.ckptDir, self.model)
        da_utils.save2file('train phase : %s' % self.train_phase, self.ckptDir, self.model)

    def convLayer(self, inputMap, out_channel, ksize, stride, scope_name, padding='SAME'):
        with tf.variable_scope(scope_name):
            conv_weight = tf.get_variable('conv_weight', [ksize, ksize, inputMap.get_shape()[-1], out_channel],
                                          initializer=layers.variance_scaling_initializer(),
                                          regularizer=layers.l2_regularizer(self.wd))

            conv_result = tf.nn.conv2d(inputMap, conv_weight, strides=[1, stride, stride, 1], padding=padding)

            tf.summary.histogram('conv_weight', conv_weight)
            tf.summary.histogram('conv_result', conv_result)

            return conv_result

    def bnLayer(self, inputMap, scope_name, is_training):
        with tf.variable_scope(scope_name):
            return tf.layers.batch_normalization(inputMap, training=is_training, epsilon=1e-5, momentum=0.9)

    def reluLayer(self, inputMap, scope_name):
        with tf.variable_scope(scope_name):
            return tf.nn.relu(inputMap)

    def leakyReluLayer(self, inputMap, scope_name):
        with tf.variable_scope(scope_name):
            return tf.nn.leaky_relu(inputMap)

    def bnLReluConvLayer(self, inputMap, out_channel, ksize, stride, scope_name, is_training):
        with tf.variable_scope(scope_name):
            _bn = self.bnLayer(inputMap, scope_name='_bn', is_training=is_training)
            _relu = self.leakyReluLayer(_bn, scope_name='_leakyReLU')
            _conv = self.convLayer(_relu, out_channel=out_channel, ksize=ksize, stride=stride, scope_name='_conv')

        return _conv

    def avgPoolLayer(self, inputMap, ksize, stride, scope_name, padding='SAME'):
        with tf.variable_scope(scope_name):
            return tf.nn.avg_pool(inputMap, ksize=[1, ksize, ksize, 1], strides=[1, stride, stride, 1], padding=padding)

    def globalPoolLayer(self, inputMap, scope_name):
        with tf.variable_scope(scope_name):
            size = inputMap.get_shape()[1]
            return self.avgPoolLayer(inputMap, size, size, padding='VALID', scope_name=scope_name)

    def fcLayer(self, inputMap, out_channel, scope_name):
        with tf.variable_scope(scope_name):
            in_channel = inputMap.get_shape()[-1]
            fc_weight = tf.get_variable('fc_weight', [in_channel, out_channel],
                                        initializer=layers.variance_scaling_initializer(),
                                        regularizer=layers.l2_regularizer(self.wd))
            fc_bias = tf.get_variable('fc_bias', [out_channel], initializer=tf.zeros_initializer())

            fc_result = tf.matmul(inputMap, fc_weight) + fc_bias

            tf.summary.histogram('fc_weight', fc_weight)
            tf.summary.histogram('fc_bias', fc_bias)
            tf.summary.histogram('fc_result', fc_result)

            return fc_result

    def flattenLayer(self, inputMap, scope_name):
        with tf.variable_scope(scope_name):
            return tf.layers.flatten(inputMap)

    def residualUnitLayer(self, inputMap, out_channel, ksize, unit_name, down_sampling, is_training, first_conv=False):
        with tf.variable_scope(unit_name):
            in_channel = inputMap.get_shape().as_list()[-1]
            if down_sampling:
                stride = 2
                increase_dim = True
            else:
                stride = 1
                increase_dim = False

            if first_conv:
                conv_layer1 = self.convLayer(inputMap, out_channel, ksize, stride, scope_name='conv_layer1')
            else:
                bn_layer1 = self.bnLayer(inputMap, scope_name='bn_layer1', is_training=is_training)
                relu_layer1 = self.reluLayer(bn_layer1, scope_name='relu_layer1')
                conv_layer1 = self.convLayer(relu_layer1, out_channel, ksize, stride, scope_name='conv_layer1')

            bn_layer2 = self.bnLayer(conv_layer1, scope_name='bn_layer2', is_training=is_training)
            relu_layer2 = self.reluLayer(bn_layer2, scope_name='relu_layer2')
            conv_layer2 = self.convLayer(relu_layer2, out_channel, ksize, stride=1, scope_name='conv_layer2')

            if increase_dim:
                identical_mapping = self.avgPoolLayer(inputMap, ksize=2, stride=2, scope_name='identical_pool')
                identical_mapping = tf.pad(identical_mapping, [[0, 0], [0, 0], [0, 0],
                                                               [(out_channel - in_channel) // 2,
                                                                (out_channel - in_channel) // 2]])
            else:
                identical_mapping = inputMap

            return tf.add(conv_layer2, identical_mapping)

    def residualSectionLayer(self, inputMap, ksize, out_channel, unit_num, section_name, down_sampling, first_conv,
                             is_training):
        with tf.variable_scope(section_name):
            _out = inputMap
            _out = self.residualUnitLayer(_out, out_channel, ksize, unit_name='unit_1', down_sampling=down_sampling,
                                          first_conv=first_conv, is_training=is_training)
            for n in range(2, unit_num + 1):
                _out = self.residualUnitLayer(_out, out_channel, ksize, unit_name='unit_' + str(n),
                                              down_sampling=False, first_conv=False, is_training=is_training)

            return _out

    def resnet_model(self, input_x, model_name, unit_num1, unit_num2, unit_num3):
        with tf.variable_scope(model_name, reuse=tf.AUTO_REUSE):
            _conv = self.convLayer(input_x, self.oc1, self.k, stride=1, scope_name='unit1_conv')
            _bn = self.bnLayer(_conv, scope_name='unit1_bn', is_training=self.is_training)
            _relu = self.reluLayer(_bn, scope_name='unit1_relu')

            sec1_out = self.residualSectionLayer(inputMap=_relu,
                                                 ksize=self.k,
                                                 out_channel=self.oc1,
                                                 unit_num=unit_num1,
                                                 section_name='section1',
                                                 down_sampling=False,
                                                 first_conv=True,
                                                 is_training=self.is_training)

            sec2_out = self.residualSectionLayer(inputMap=sec1_out,
                                                 ksize=self.k,
                                                 out_channel=self.oc2,
                                                 unit_num=unit_num2,
                                                 section_name='section2',
                                                 down_sampling=True,
                                                 first_conv=False,
                                                 is_training=self.is_training)

            sec3_out = self.residualSectionLayer(inputMap=sec2_out,
                                                 ksize=self.k,
                                                 out_channel=self.oc3,
                                                 unit_num=unit_num3,
                                                 section_name='section3',
                                                 down_sampling=True,
                                                 first_conv=False,
                                                 is_training=self.is_training)

            _fm_bn = self.bnLayer(sec3_out, scope_name='_fm_bn', is_training=self.is_training)
            _fm_relu = self.reluLayer(_fm_bn, scope_name='_fm_relu')
            _fm_pool = self.globalPoolLayer(_fm_relu, scope_name='_fm_gap')
            _fm_flatten = self.flattenLayer(_fm_pool, scope_name='_fm_flatten')

            y_pred = self.fcLayer(_fm_flatten, self.num_class, scope_name='fc_pred')
            y_pred_softmax = tf.nn.softmax(y_pred)

            return y_pred, y_pred_softmax, [sec1_out, sec2_out, sec3_out]

    def featureDiscriminator(self, inputMap, ksize, scope_name, is_training):
        with tf.variable_scope(scope_name + '_feature_discriminator', reuse=tf.AUTO_REUSE):
            in_channel = inputMap.get_shape().as_list()[-1]
            _layer1 = self.bnLReluConvLayer(inputMap, out_channel=in_channel, ksize=ksize, stride=1,
                                            scope_name='_layer1', is_training=is_training)
            _layer2 = self.bnLReluConvLayer(_layer1, out_channel=in_channel, ksize=ksize, stride=1,
                                            scope_name='_layer2', is_training=is_training)
            _layer3 = self.bnLReluConvLayer(_layer2, out_channel=in_channel * 2, ksize=ksize, stride=1,
                                            scope_name='_layer3', is_training=is_training)
            _layer4 = self.bnLReluConvLayer(_layer3, out_channel=in_channel * 2, ksize=ksize, stride=1,
                                            scope_name='_layer4', is_training=is_training)

        return _layer4

    def predictionDiscriminator(self, inputMap, scope_name):
        with tf.variable_scope(scope_name + '_prediction_discriminator', reuse=tf.AUTO_REUSE):
            _relu0 = self.reluLayer(inputMap, scope_name='_relu0')
            _fc1 = self.fcLayer(_relu0, out_channel=128, scope_name='_fc1')
            _relu1 = self.reluLayer(_fc1, scope_name='_relu1')
            _fc2 = self.fcLayer(_relu1, out_channel=256, scope_name='_fc2')
            _relu2 = self.reluLayer(_fc2, scope_name='_relu2')
            _fc3 = self.fcLayer(_relu2, out_channel=512, scope_name='_fc3')
            _relu3 = self.reluLayer(_fc3, scope_name='_relu3')
            _fc4 = self.fcLayer(_relu3, out_channel=1, scope_name='_fc4')

        return _fc4

    def build_model(self):
        self.x_source = tf.placeholder(tf.float32, shape=[None, self.img_h, self.img_w, 1], name='x_source')
        self.x_target = tf.placeholder(tf.float32, shape=[None, self.img_h, self.img_w, 1], name='x_target')
        self.y_source = tf.placeholder(tf.int32, shape=[None, self.num_class], name='y_source')
        self.y_target = tf.placeholder(tf.int32, shape=[None, self.num_class], name='y_target')

        self.is_training = tf.placeholder(tf.bool, name='is_training')

        tf.summary.image('source_input', self.x_source)
        tf.summary.image('target_input', self.x_target)

        self.y_pred_source, self.y_pred_softmax_source, self.featureMapZoo_Source = self.resnet_model(
            input_x=self.x_source,
            model_name='generator',
            unit_num1=3,
            unit_num2=3,
            unit_num3=3)

        self.y_pred_target, self.y_pred_softmax_target, self.featureMapZoo_Target = self.resnet_model(
            input_x=self.x_target,
            model_name='generator',
            unit_num1=3,
            unit_num2=3,
            unit_num3=3)

        self.sec1_fm_source = self.featureDiscriminator(self.featureMapZoo_Source[0], ksize=3,
                                                        scope_name='sec1_featureMap', is_training=self.is_training)
        self.sec1_fm_target = self.featureDiscriminator(self.featureMapZoo_Target[0], ksize=3,
                                                        scope_name='sec1_featureMap', is_training=self.is_training)

        self.sec2_fm_source = self.featureDiscriminator(self.featureMapZoo_Source[1], ksize=3,
                                                        scope_name='sec2_featureMap', is_training=self.is_training)
        self.sec2_fm_target = self.featureDiscriminator(self.featureMapZoo_Target[1], ksize=3,
                                                        scope_name='sec2_featureMap', is_training=self.is_training)

        self.sec3_fm_source = self.featureDiscriminator(self.featureMapZoo_Source[2], ksize=3,
                                                        scope_name='sec3_featureMap', is_training=self.is_training)
        self.sec3_fm_target = self.featureDiscriminator(self.featureMapZoo_Target[2], ksize=3,
                                                        scope_name='sec3_featureMap', is_training=self.is_training)

        self.pred_source = self.predictionDiscriminator(self.y_pred_source, scope_name='prediction')
        self.pred_target = self.predictionDiscriminator(self.y_pred_target, scope_name='prediction')

        with tf.variable_scope('loss'):
            # source domain supervised loss
            self.cost = tf.reduce_mean(
                tf.nn.softmax_cross_entropy_with_logits(logits=self.y_pred_source, labels=self.y_source))
            self.l2_cost = tf.add_n(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))
            self.loss_source = self.cost + self.l2_cost

            # generator losses
            self.g_sec1_target_loss = tf.reduce_mean(tf.losses.mean_squared_error(
                predictions=self.sec1_fm_target,
                labels=tf.ones_like(self.sec1_fm_target)))
            self.g_sec2_target_loss = tf.reduce_mean(tf.losses.mean_squared_error(
                predictions=self.sec2_fm_target,
                labels=tf.ones_like(self.sec2_fm_target)))
            self.g_sec3_target_loss = tf.reduce_mean(tf.losses.mean_squared_error(
                predictions=self.sec3_fm_target,
                labels=tf.ones_like(self.sec3_fm_target)))
            self.g_pred_target_loss = tf.reduce_mean(tf.losses.mean_squared_error(
                predictions=self.pred_target,
                labels=tf.ones_like(self.pred_target)))

            self.g_loss = self.g_sec1_target_loss * 0.01 + self.g_sec2_target_loss * 0.01 + \
                          self.g_sec3_target_loss * 0.01 + self.g_pred_target_loss * 0.01 + \
                          self.loss_source

            # discriminator losses
            self.featureMap_d_sec1_source_loss = tf.reduce_mean(tf.losses.mean_squared_error(
                predictions=self.sec1_fm_source,
                labels=tf.ones_like(self.sec1_fm_source)))
            self.featureMap_d_sec1_target_loss = tf.reduce_mean(tf.losses.mean_squared_error(
                predictions=self.sec1_fm_target,
                labels=tf.zeros_like(self.sec1_fm_target)))

            self.featureMap_d_sec2_source_loss = tf.reduce_mean(tf.losses.mean_squared_error(
                predictions=self.sec2_fm_source,
                labels=tf.ones_like(self.sec2_fm_source)))
            self.featureMap_d_sec2_target_loss = tf.reduce_mean(tf.losses.mean_squared_error(
                predictions=self.sec2_fm_target,
                labels=tf.zeros_like(self.sec2_fm_target)))

            self.featureMap_d_sec3_source_loss = tf.reduce_mean(tf.losses.mean_squared_error(
                predictions=self.sec3_fm_source,
                labels=tf.ones_like(self.sec3_fm_source)))
            self.featureMap_d_sec3_target_loss = tf.reduce_mean(tf.losses.mean_squared_error(
                predictions=self.sec3_fm_target,
                labels=tf.zeros_like(self.sec3_fm_target)))

            self.prediction_d_source_loss = tf.reduce_mean(tf.losses.mean_squared_error(
                predictions=self.pred_source,
                labels=tf.ones_like(self.pred_source)))
            self.prediction_d_target_loss = tf.reduce_mean(tf.losses.mean_squared_error(
                predictions=self.pred_target,
                labels=tf.zeros_like(self.pred_target)))

            self.feature_d_loss = self.featureMap_d_sec1_source_loss + self.featureMap_d_sec1_target_loss + \
                                  self.featureMap_d_sec2_source_loss + self.featureMap_d_sec2_target_loss + \
                                  self.featureMap_d_sec3_source_loss + self.featureMap_d_sec3_target_loss

            self.prediction_d_loss = self.prediction_d_source_loss + self.prediction_d_target_loss
            self.d_loss = self.feature_d_loss + self.prediction_d_loss

            tf.summary.scalar('source loss', self.loss_source)
            tf.summary.scalar('generatort_sec1_loss', self.g_sec1_target_loss)
            tf.summary.scalar('generator_sec2_loss', self.g_sec2_target_loss)
            tf.summary.scalar('generator_sec3_loss', self.g_sec3_target_loss)
            tf.summary.scalar('g_loss', self.g_loss)
            tf.summary.scalar('feature_discriminator_loss', self.feature_d_loss)
            tf.summary.scalar('prediction_discriminator_loss', self.prediction_d_loss)
            tf.summary.scalar('d_loss', self.d_loss)

        with tf.variable_scope('optimization_variables'):
            self.t_var = tf.trainable_variables()

            self.g_var = [var for var in self.t_var if 'generator' in var.name]

            self.d_feature_var = [var for var in self.t_var if '_feature_discriminator' in var.name]
            self.d_prediction_var = [var for var in self.t_var if '_prediction_discriminator' in var.name]
            self.d_var = [var for var in self.t_var if '_discriminator' in var.name]

        with tf.variable_scope('optimize'):
            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            with tf.control_dependencies(update_ops):
                self.supervised_train_op = tf.train.AdamOptimizer(self.lr, beta1=0.5).minimize(self.loss_source,
                                                                                               var_list=self.g_var)
                self.g_train_op = tf.train.AdamOptimizer(self.lr, beta1=0.5).minimize(self.g_loss, var_list=self.g_var)

                self.d_feature_train_op = tf.train.AdamOptimizer(self.lr, beta1=0.5).minimize(
                    self.feature_d_loss,
                    var_list=self.d_feature_var)
                self.d_prediction_train_op = tf.train.AdamOptimizer(self.lr, beta1=0.5).minimize(
                    self.prediction_d_loss,
                    var_list=self.d_prediction_var)
                self.d_train_op = tf.train.AdamOptimizer(self.lr, beta1=0.5).minimize(self.d_loss, var_list=self.d_var)

        with tf.variable_scope('tfSummary'):
            self.merged = tf.summary.merge_all()
            if self.train_phase == 'Train':
                self.writer = tf.summary.FileWriter(self.ckptDir, self.sess.graph)

        with tf.variable_scope('saver'):
            var_list = tf.trainable_variables()
            g_list = tf.global_variables()
            bn_moving_vars = [g for g in g_list if 'moving_mean' in g.name]
            bn_moving_vars += [g for g in g_list if 'moving_variance' in g.name]
            var_list += bn_moving_vars
            self.saver = tf.train.Saver(var_list=var_list, max_to_keep=self.eps)

        with tf.variable_scope('accuracy'):
            self.distribution_source = [tf.argmax(self.y_source, 1), tf.argmax(self.y_pred_softmax_source, 1)]
            self.distribution_target = [tf.argmax(self.y_target, 1), tf.argmax(self.y_pred_softmax_target, 1)]

            self.correct_prediction_source = tf.equal(self.distribution_source[0], self.distribution_source[1])
            self.correct_prediction_target = tf.equal(self.distribution_target[0], self.distribution_target[1])

            self.accuracy_source = tf.reduce_mean(tf.cast(self.correct_prediction_source, 'float'))
            self.accuracy_target = tf.reduce_mean(tf.cast(self.correct_prediction_target, 'float'))

    def f_value(self, matrix):
        f = 0.0
        length = len(matrix[0])
        for i in range(length):
            recall = matrix[i][i] / np.sum([matrix[i][m] for m in range(self.num_class)])
            precision = matrix[i][i] / np.sum([matrix[n][i] for n in range(self.num_class)])
            result = (recall * precision) / (recall + precision)
            f += result
        f *= (2 / self.num_class)

        return f

    def validation_procedure(self, validation_data, distribution_op, loss_op, inputX, inputY):
        confusion_matrics = np.zeros([self.num_class, self.num_class], dtype="int")
        val_loss = 0.0

        val_batch_num = int(np.ceil(validation_data[0].shape[0] / self.bs))
        for step in range(val_batch_num):
            _validationImg = validation_data[0][step * self.bs:step * self.bs + self.bs]
            _validationLab = validation_data[1][step * self.bs:step * self.bs + self.bs]

            [matrix_row, matrix_col], tmp_loss = self.sess.run([distribution_op, loss_op],
                                                               feed_dict={inputX: _validationImg,
                                                                          inputY: _validationLab,
                                                                          self.is_training: False})
            for m, n in zip(matrix_row, matrix_col):
                confusion_matrics[m][n] += 1

            val_loss += tmp_loss

        validation_accuracy = float(np.sum([confusion_matrics[q][q] for q in range(self.num_class)])) / float(
            np.sum(confusion_matrics))
        validation_loss = val_loss / val_batch_num

        return validation_accuracy, validation_loss

    def test_procedure(self, test_data, distribution_op, inputX, inputY, mode):
        confusion_matrics = np.zeros([self.num_class, self.num_class], dtype="int")

        tst_batch_num = int(np.ceil(test_data[0].shape[0] / self.bs))
        for step in range(tst_batch_num):
            _testImg = test_data[0][step * self.bs:step * self.bs + self.bs]
            _testLab = test_data[1][step * self.bs:step * self.bs + self.bs]

            matrix_row, matrix_col = self.sess.run(distribution_op, feed_dict={inputX: _testImg,
                                                                               inputY: _testLab,
                                                                               self.is_training: False})
            for m, n in zip(matrix_row, matrix_col):
                confusion_matrics[m][n] += 1

        test_accuracy = float(np.sum([confusion_matrics[q][q] for q in range(self.num_class)])) / float(
            np.sum(confusion_matrics))
        detail_test_accuracy = [confusion_matrics[i][i] / np.sum(confusion_matrics[i]) for i in range(self.num_class)]
        log0 = "Mode: " + mode
        log1 = "Test Accuracy : %g" % test_accuracy
        log2 = np.array(confusion_matrics.tolist())
        log3 = ''
        for j in range(self.num_class):
            log3 += 'category %s test accuracy : %g\n' % (da_utils.pulmonary_category[j], detail_test_accuracy[j])
        log3 = log3[:-1]
        log4 = 'F_Value : %g\n' % self.f_value(confusion_matrics)

        da_utils.save2file(log0, self.ckptDir, self.model)
        da_utils.save2file(log1, self.ckptDir, self.model)
        da_utils.save2file(log2, self.ckptDir, self.model)
        da_utils.save2file(log3, self.ckptDir, self.model)
        da_utils.save2file(log4, self.ckptDir, self.model)

    def train(self):
        print('Start to run in mode [Domain Adaptation Across Source and Target Domain]')
        self.sess.run(tf.global_variables_initializer())
        self.train_itr = min(len(self.source_training_data[0]), len(self.target_training_data)) // self.bs

        for e in range(1, self.eps + 1):
            _src_tr_img, _src_tr_lab = DA_init.shuffle_data(self.source_training_data[0], self.source_training_data[1])
            _tar_tr_img = DA_init.shuffle_data_nolabel(self.target_training_data)

            source_training_acc = 0.0
            source_training_loss = 0.0
            g_loss = 0.0
            d_loss = 0.0

            for itr in range(self.train_itr):
                _src_tr_img_batch, _src_tr_lab_batch = DA_init.next_batch(_src_tr_img, _src_tr_lab, self.bs, itr)
                _tar_tr_img_batch = DA_init.next_batch_nolabel(_tar_tr_img, self.bs, itr)

                if e < 50:
                    _ = self.sess.run(self.supervised_train_op, feed_dict={self.x_source: _src_tr_img_batch,
                                                                           self.y_source: _src_tr_lab_batch,
                                                                           self.x_target: _tar_tr_img_batch,
                                                                           self.is_training: True})
                else:
                    _, _ = self.sess.run([self.d_train_op, self.g_train_op],
                                         feed_dict={self.x_source: _src_tr_img_batch,
                                                    self.y_source: _src_tr_lab_batch,
                                                    self.x_target: _tar_tr_img_batch,
                                                    self.is_training: True})

                _training_accuracy, _training_loss, _g_loss, _d_loss = self.sess.run(
                    [self.accuracy_source, self.loss_source, self.g_loss, self.d_loss],
                    feed_dict={
                        self.x_source: _src_tr_img_batch,
                        self.y_source: _src_tr_lab_batch,
                        self.x_target: _tar_tr_img_batch,
                        self.is_training: False})

                source_training_acc += _training_accuracy
                source_training_loss += _training_loss
                g_loss += _g_loss
                d_loss += _d_loss

            source_training_acc = float(source_training_acc / self.train_itr)
            source_training_loss = float(source_training_loss / self.train_itr)
            g_loss = float(g_loss / self.train_itr)
            d_loss = float(d_loss / self.train_itr)

            source_validation_acc, source_validation_loss = self.validation_procedure(
                validation_data=self.source_validation_data, distribution_op=self.distribution_source,
                loss_op=self.loss_source, inputX=self.x_source, inputY=self.y_source)

            tf.summary.scalar('source_training_accuracy', source_training_acc)
            tf.summary.scalar('source_training_loss', source_training_loss)
            tf.summary.scalar('g_loss', g_loss)
            tf.summary.scalar('d_loss', d_loss)
            tf.summary.scalar('source_validation_accuracy', source_validation_acc)
            tf.summary.scalar('source_validation_loss', source_validation_loss)

            log1 = "Epoch: [%d], Domain: Source, Training Accuracy: [%g], Validation Accuracy: [%g], " \
                   "Training Loss: [%g], Validation Loss: [%g], generator Loss: [%g], Discriminator Loss: [%g], " \
                   "Time: [%s]" % (
                   e, source_training_acc, source_validation_acc, source_training_loss, source_validation_loss,
                   g_loss, d_loss, time.ctime(time.time()))

            da_utils.save2file(log1, self.ckptDir, self.model)

            summary = self.sess.run(self.merged, feed_dict={self.x_source: _src_tr_img_batch,
                                                            self.y_source: _src_tr_lab_batch,
                                                            self.x_target: _tar_tr_img_batch,
                                                            self.is_training: False})
            self.writer.add_summary(summary, e)

            self.saver.save(self.sess, self.ckptDir + self.model + '-' + str(e) + '.ckpt')

            self.test_procedure(self.source_test_data, distribution_op=self.distribution_source, inputX=self.x_source,
                                inputY=self.y_source, mode='source')
            self.test_procedure(self.target_test_data, distribution_op=self.distribution_target, inputX=self.x_target,
                                inputY=self.y_target, mode='target')

    def test(self):
        print('Start to run in mode [Test in Target Domain]')
        self.saver.restore(self.sess, self.ckptDir + self.model + '-' + str(self.res_eps) + '.ckpt')
        self.test_procedure(self.target_test_data, distribution_op=self.distribution_target, inputX=self.x_target,
                            inputY=self.y_target, mode='target')
