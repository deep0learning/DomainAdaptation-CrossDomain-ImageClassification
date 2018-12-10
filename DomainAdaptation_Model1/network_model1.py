import sys

sys.path.append('../Data_Initialization/')
import tensorflow as tf
import DomainAdaptation_Initialization as DA_init
import tensorflow.contrib.layers as layers
import utils_model1
import time


class Image_Generator_Model(object):
    def __init__(self, model_name, sess, train_data, val_data, tst_data, epoch, restore_epoch, learning_rate,
                 batch_size, img_height, img_width, train_phase):

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
        self.lr = learning_rate
        self.bs = batch_size
        self.img_h = img_height
        self.img_w = img_width
        self.train_phase = train_phase
        self.plt_epoch = []
        self.plt_g_loss = []
        self.plt_d_loss = []

        self.build_model()
        if self.train_phase == 'Train':
            self.saveConfiguration()

    def saveConfiguration(self):
        utils_model1.save2file('epoch : %d' % self.eps, self.ckptDir, self.model)
        utils_model1.save2file('restore epoch : %d' % self.res_eps, self.ckptDir, self.model)
        utils_model1.save2file('model : %s' % self.model, self.ckptDir, self.model)
        utils_model1.save2file('learning rate : %g' % self.lr, self.ckptDir, self.model)
        utils_model1.save2file('batch size : %d' % self.bs, self.ckptDir, self.model)
        utils_model1.save2file('image height : %d' % self.img_h, self.ckptDir, self.model)
        utils_model1.save2file('image width : %d' % self.img_w, self.ckptDir, self.model)
        utils_model1.save2file('train phase : %s' % self.train_phase, self.ckptDir, self.model)

    def convLayer(self, inputMap, out_channel, ksize, stride, scope_name, padding='SAME'):
        with tf.variable_scope(scope_name):
            conv_weight = tf.get_variable('conv_weight',
                                          [ksize, ksize, inputMap.get_shape().as_list()[-1], out_channel],
                                          initializer=layers.xavier_initializer())

            conv_result = tf.nn.conv2d(inputMap, conv_weight, strides=[1, stride, stride, 1], padding=padding)

            tf.summary.histogram('conv_weight', conv_weight)
            tf.summary.histogram('conv_result', conv_result)

            return conv_result

    def deconvLayer(self, inputMap, out_channel, out_shape, ksize, stride, scope_name, padding='SAME'):
        with tf.variable_scope(scope_name):
            conv_weight = tf.get_variable('deconv_weight',
                                          [ksize, ksize, out_channel, inputMap.get_shape().as_list()[-1]],
                                          initializer=layers.xavier_initializer())

            conv_result = tf.nn.conv2d_transpose(value=inputMap,
                                                 filter=conv_weight,
                                                 output_shape=out_shape,
                                                 strides=[1, stride, stride, 1],
                                                 padding=padding)

            tf.summary.histogram('deconv_weight', conv_weight)
            tf.summary.histogram('deconv_result', conv_result)

            return conv_result

    def bnLayer(self, inputMap, scope_name, is_training):
        with tf.variable_scope(scope_name):
            return tf.layers.batch_normalization(inputMap, training=is_training, epsilon=1e-5, momentum=0.9)

    def reluLayer(self, inputMap, scope_name):
        with tf.variable_scope(scope_name):
            return tf.nn.relu(inputMap)

    def leakyReluLayer(self, inputMap, scope_name, rate=0.2):
        with tf.variable_scope(scope_name):
            return tf.nn.leaky_relu(inputMap, alpha=rate)

    def avgPoolLayer(self, inputMap, ksize, stride, scope_name, padding='SAME'):
        with tf.variable_scope(scope_name):
            return tf.nn.avg_pool(inputMap, ksize=[1, ksize, ksize, 1], strides=[1, stride, stride, 1], padding=padding)

    def convBnLReluLayer(self, inputMap, ksize, stride, out_channel, scope_name, is_training):
        with tf.variable_scope(scope_name):
            _conv = self.convLayer(inputMap, out_channel=out_channel, ksize=ksize, stride=stride, scope_name='_conv')
            _bn = self.bnLayer(_conv, scope_name='_bn', is_training=is_training)
            _relu = self.leakyReluLayer(_bn, scope_name='_lrelu')

        return _relu

    def bnReluConvLayer(self, inputMap, ksize, stride, out_channel, scope_name, is_training):
        with tf.variable_scope(scope_name):
            _bn = self.bnLayer(inputMap, scope_name='_bn', is_training=is_training)
            _relu = self.reluLayer(_bn, scope_name='_relu')
            _conv = self.convLayer(_relu, out_channel=out_channel, ksize=ksize, stride=stride, scope_name='_conv')

        return _conv

    def bnReluDeconvLayer(self, inputMap, ksize, stride, out_channel, out_shape, scope_name, is_training):
        with tf.variable_scope(scope_name):
            _bn = self.bnLayer(inputMap, scope_name='_bn', is_training=is_training)
            _relu = self.reluLayer(_bn, scope_name='_relu')
            _deconv = self.deconvLayer(_relu, out_channel=out_channel, out_shape=out_shape, ksize=ksize, stride=stride,
                                       scope_name='_deconv')

        return _deconv

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

    def unet_model(self, input_x, ksize, out_channel1, out_channel2, out_channel3, model_name, is_training):
        with tf.variable_scope(model_name, reuse=tf.AUTO_REUSE):
            _conv = self.convLayer(input_x, out_channel1, ksize=ksize, stride=1, scope_name='unit1_conv')
            _bn = self.bnLayer(_conv, scope_name='unit1_bn', is_training=self.is_training)
            _relu = self.reluLayer(_bn, scope_name='unit1_relu')

            sec1_out = self.residualSectionLayer(inputMap=_relu,
                                                 ksize=ksize,
                                                 out_channel=out_channel1,
                                                 unit_num=2,
                                                 section_name='section1',
                                                 down_sampling=False,
                                                 first_conv=True,
                                                 is_training=self.is_training)

            sec2_out = self.residualSectionLayer(inputMap=sec1_out,
                                                 ksize=ksize,
                                                 out_channel=out_channel2,
                                                 unit_num=2,
                                                 section_name='section2',
                                                 down_sampling=True,
                                                 first_conv=False,
                                                 is_training=self.is_training)

            sec3_out = self.residualSectionLayer(inputMap=sec2_out,
                                                 ksize=ksize,
                                                 out_channel=out_channel3,
                                                 unit_num=2,
                                                 section_name='section3',
                                                 down_sampling=True,
                                                 first_conv=False,
                                                 is_training=self.is_training)

            deconv1_out = self.bnReluDeconvLayer(sec3_out,
                                                 ksize=ksize,
                                                 stride=2,
                                                 out_channel=out_channel2,
                                                 out_shape=tf.shape(sec2_out),
                                                 scope_name='deconv_out1',
                                                 is_training=is_training)

            concat_1 = tf.concat([sec2_out, deconv1_out], axis=-1, name='concat_1')

            deconv2_out = self.bnReluDeconvLayer(concat_1,
                                                 ksize=ksize,
                                                 stride=2,
                                                 out_channel=out_channel1,
                                                 out_shape=tf.shape(sec1_out),
                                                 scope_name='deconv_out2',
                                                 is_training=is_training)

            concat_2 = tf.concat([sec1_out, deconv2_out], axis=-1, name='concat_2')

            conv3_out = self.bnReluConvLayer(concat_2,
                                             ksize=3,
                                             stride=1,
                                             out_channel=64,
                                             scope_name='conv3_out',
                                             is_training=is_training)

            conv_out = self.convLayer(inputMap=conv3_out,
                                      ksize=1,
                                      stride=1,
                                      out_channel=1,
                                      scope_name='conv_out')

        return tf.nn.relu(conv_out)

    def Discriminator(self, inputMap, ksize, scope_name, is_training):
        with tf.variable_scope(scope_name + '_discriminator', reuse=tf.AUTO_REUSE):
            _layer1 = self.convLayer(inputMap=inputMap,
                                     out_channel=64,
                                     ksize=ksize,
                                     stride=1,
                                     scope_name='_layer1_conv')
            _layer1 = self.leakyReluLayer(_layer1, scope_name='_layer1_lrelu')
            _layer2 = self.convBnLReluLayer(_layer1,
                                            out_channel=128,
                                            ksize=ksize,
                                            stride=2,
                                            scope_name='_layer3',
                                            is_training=is_training)
            _layer3 = self.convBnLReluLayer(_layer2,
                                            out_channel=256,
                                            ksize=ksize,
                                            stride=2,
                                            scope_name='_layer5',
                                            is_training=is_training)
            _layer4 = self.convLayer(inputMap=_layer3,
                                     out_channel=1,
                                     ksize=1,
                                     stride=1,
                                     scope_name='_layer4')

        return _layer4

    def build_model(self):
        self.x_source = tf.placeholder(tf.float32, shape=[None, self.img_h, self.img_w, 1], name='x_source')
        self.x_target = tf.placeholder(tf.float32, shape=[None, self.img_h, self.img_w, 1], name='x_target')

        self.is_training = tf.placeholder(tf.bool, name='is_training')

        tf.summary.image('source_input', self.x_source)
        tf.summary.image('target_input', self.x_target)

        self.g1_source_fake = self.unet_model(self.x_source,
                                              ksize=3,
                                              out_channel1=64,
                                              out_channel2=128,
                                              out_channel3=256,
                                              model_name='G1',
                                              is_training=self.is_training)
        self.g1_target_fake = self.unet_model(self.x_target,
                                              ksize=3,
                                              out_channel1=64,
                                              out_channel2=128,
                                              out_channel3=256,
                                              model_name='G1',
                                              is_training=self.is_training)

        tf.summary.image('source fake', self.g1_source_fake)
        tf.summary.image('target fake', self.g1_target_fake)

        self.g1_source_fake_dis = self.Discriminator(self.g1_source_fake,
                                                     ksize=3,
                                                     scope_name='D1',
                                                     is_training=self.is_training)
        self.target_dis = self.Discriminator(self.x_target,
                                             ksize=3,
                                             scope_name='D1',
                                             is_training=self.is_training)

        with tf.variable_scope('loss'):
            # reconstruction loss
            self.rec_loss = tf.reduce_mean(tf.losses.absolute_difference(self.x_target, self.g1_target_fake))

            # generator losses
            self.g1_source_fake_gloss = tf.reduce_mean(tf.losses.mean_squared_error(
                predictions=self.g1_source_fake_dis,
                labels=tf.ones_like(self.g1_source_fake_dis)))

            self.g_loss = self.g1_source_fake_gloss + 10 * self.rec_loss

            # discriminator losses
            self.g1_source_fake_dloss = tf.reduce_mean(tf.losses.mean_squared_error(
                predictions=self.g1_source_fake_dis,
                labels=tf.zeros_like(self.g1_source_fake_dis)))
            self.g1_target_dloss = tf.reduce_mean(tf.losses.mean_squared_error(
                predictions=self.target_dis,
                labels=tf.ones_like(self.target_dis)))

            self.d_loss = self.g1_source_fake_dloss + self.g1_target_dloss

            tf.summary.scalar('g loss', self.g_loss)
            tf.summary.scalar('d loss', self.d_loss)
            tf.summary.scalar('g gan loss', self.g1_source_fake_gloss)
            tf.summary.scalar('reconstruction loss', self.rec_loss)
            tf.summary.scalar('d gan source loss', self.g1_source_fake_dloss)
            tf.summary.scalar('d gan target loss', self.g1_target_dloss)

        with tf.variable_scope('optimization_variables'):
            self.t_var = tf.trainable_variables()
            self.g1_var = [var for var in self.t_var if 'G1' in var.name]
            self.d1_var = [var for var in self.t_var if 'D1' in var.name]

        with tf.variable_scope('optimize'):
            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            g_bn_ops = [var for var in update_ops if 'G1' in var.name]
            d_bn_ops = [var for var in update_ops if 'D1' in var.name]
            with tf.control_dependencies(g_bn_ops):
                self.g_train_op = tf.train.AdamOptimizer(self.lr, beta1=0.5).minimize(self.g_loss,
                                                                                      var_list=self.g1_var)
            with tf.control_dependencies(d_bn_ops):
                self.d_train_op = tf.train.AdamOptimizer(self.lr, beta1=0.5).minimize(self.d_loss,
                                                                                      var_list=self.d1_var)
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

    def train(self):
        print('Start to run in mode [Domain Adaptation Across Source and Target Domain]')
        self.sess.run(tf.global_variables_initializer())

        self.train_itr = len(self.source_training_data[0]) // self.bs

        for e in range(1, self.eps + 1):
            _src_tr_img, _src_tr_lab = DA_init.shuffle_data(self.source_training_data[0],
                                                            self.source_training_data[1])
            _tar_tr_img = DA_init.shuffle_data_nolabel(self.target_training_data)

            g_loss = 0.0
            d_loss = 0.0

            for itr in range(self.train_itr):
                _src_tr_img_batch, _src_tr_lab_batch = DA_init.next_batch(_src_tr_img, _src_tr_lab, self.bs, itr)
                _tar_tr_img_batch = DA_init.next_batch_nolabel(_tar_tr_img, self.bs)

                feed_dict = {self.x_source: _src_tr_img_batch,
                             self.x_target: _tar_tr_img_batch,
                             self.is_training: True}
                feed_dict_eval = {self.x_source: _src_tr_img_batch,
                                  self.x_target: _tar_tr_img_batch,
                                  self.is_training: False}

                _ = self.sess.run(self.d_train_op, feed_dict=feed_dict)
                _ = self.sess.run(self.g_train_op, feed_dict=feed_dict)

                _g_loss, _d_loss = self.sess.run([self.g_loss, self.d_loss], feed_dict=feed_dict_eval)

                g_loss += _g_loss
                d_loss += _d_loss

            summary = self.sess.run(self.merged, feed_dict=feed_dict_eval)

            g_loss = float(g_loss / self.train_itr)
            d_loss = float(d_loss / self.train_itr)

            log1 = "Epoch: [%d], G Loss: [%g], D Loss: [%g], Time: [%s]" % (e, g_loss, d_loss, time.ctime(time.time()))

            self.plt_epoch.append(e)
            self.plt_d_loss.append(d_loss)
            self.plt_g_loss.append(g_loss)

            utils_model1.plotLoss(x=self.plt_epoch,
                                  y1=self.plt_d_loss,
                                  y2=self.plt_g_loss,
                                  figName=self.model + '_GD_Loss',
                                  line1Name='D_Loss',
                                  line2Name='G_Loss',
                                  savePath=self.ckptDir)

            utils_model1.save2file(log1, self.ckptDir, self.model)

            self.writer.add_summary(summary, e)

            self.saver.save(self.sess, self.ckptDir + self.model + '-' + str(e))
