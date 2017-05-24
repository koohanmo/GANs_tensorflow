import tensorflow as tf
from modelHelper import layers

class VGG19:
    def __init__(self, input, t):
        if input is None: return
        self.out, self.phi = self.build_model(input)
        self.loss = self.inference_loss(self.out, t)

    def build_model(self, t_input, reuse=False):

        with tf.variable_scope('vgg19', reuse=reuse):
            phi = []
            with tf.variable_scope('conv1a'):
                conv1a = layers.conv2d(
                                    layerName='vgg19_conv1a',
                                    t_input=t_input,
                                    ksize= [3, 3, 1, 64], #<--[3,3,3,64]
                                    strides=[1,1,1,1],
                                    padding='SAME')
                BN = layers.batch_norm()
                conv1a = BN(conv1a)
                conv1a = layers.lrelu(conv1a)


            with tf.variable_scope('conv1b'):
                conv1b = layers.conv2d(layerName='vgg19_conv1b',
                                  t_input=conv1a,
                                  ksize=[3,3,64,64],
                                  strides=[1,1,1,1],
                                  padding='SAME')
                BN = layers.batch_norm()
                conv1b = BN(conv1b)
                conv1b = layers.lrelu(conv1b)
            phi.append(conv1b)

            pool_first = layers.maxPool(
                                        layerName='pool_first',
                                        t_input=conv1b,
                                        ksize=[1,2,2,1],
                                        strides=[1,2,2,1],
                                        padding='SAME'
                                        )
            # pool 1

            with tf.variable_scope('conv2a'):
                conv2a = layers.conv2d(layerName='vgg19_conv2a',
                                       t_input=pool_first,
                                       ksize=[3, 3, 64, 128],
                                       strides=[1,1,1,1],
                                       padding='SAME')

                BN = layers.batch_norm()
                conv2a = BN(conv2a)
                conv2a = layers.lrelu(conv2a)


            with tf.variable_scope('conv2b'):
                conv2b = layers.conv2d(layerName='vgg19_conv2b',
                                       t_input=conv2a,
                                       ksize=[3, 3, 128, 128],
                                       strides=[1,1,1,1],
                                       padding='SAME')
                BN = layers.batch_norm()
                conv2b = BN(conv2b)
                conv2b = layers.lrelu(conv2b)
            phi.append(conv2b)

            pool_second= layers.maxPool(
                                        layerName='pool_first',
                                        t_input=conv2b,
                                        ksize=[1, 2, 2, 1],
                                        strides=[1, 2, 2, 1],
                                        padding='SAME'
                                        )
            # pool 2

            with tf.variable_scope('conv3a'):
                conv3a = layers.conv2d(layerName='vgg19_conv3a',
                                       t_input=pool_second,
                                       ksize=[3, 3, 128, 256],
                                       strides=[1,1,1,1],
                                       padding='SAME')

                BN = layers.batch_norm()
                conv3a = BN(conv3a)
                conv3a = layers.lrelu(conv3a)


            with tf.variable_scope('conv3b'):
                conv3b = layers.conv2d(layerName='vgg19_conv3b',
                                       t_input=conv3a,
                                       ksize=[3, 3, 256, 256],
                                       strides=[1,1,1,1],
                                       padding='SAME')

                BN = layers.batch_norm()
                conv3b = BN(conv3b)
                conv3b = layers.lrelu(conv3b)

            with tf.variable_scope('conv3c'):
                conv3c = layers.conv2d(layerName='vgg19_conv3c',
                                       t_input=conv3b,
                                       ksize=[3, 3, 256, 256],
                                       strides=[1,1,1,1],
                                       padding='SAME')
                BN = layers.batch_norm()
                conv3c = BN(conv3c)
                conv3c = layers.lrelu(conv3c)

            with tf.variable_scope('conv3d'):
                conv3d = layers.conv2d(layerName='vgg19_conv3d',
                                       t_input=conv3c,
                                       ksize=[3, 3, 256, 256],
                                       strides=[1,1,1,1],
                                       padding='SAME')
                BN = layers.batch_norm()
                conv3d = BN(conv3d)
                conv3d = layers.lrelu(conv3d)
            phi.append(conv3d)

            pool_third =layers.maxPool(
                                        layerName='pool_first',
                                        t_input=conv3d,
                                        ksize=[1, 2, 2, 1],
                                        strides=[1, 2, 2, 1],
                                        padding='SAME'
                                        )

            with tf.variable_scope('conv4a'):
                conv4a = layers.conv2d(layerName='vgg19_conv4a',
                                       t_input=pool_third,
                                       ksize=[3, 3, 256, 512],
                                       strides=[1,1,1,1],
                                       padding='SAME')
                BN = layers.batch_norm()
                conv4a = BN(conv4a)
                conv4a = layers.lrelu(conv4a)

            with tf.variable_scope('conv4b'):
                conv4b = layers.conv2d(layerName='vgg19_conv4b',
                                       t_input=conv4a,
                                       ksize=[3, 3, 512, 512],
                                       strides=[1, 1, 1, 1],
                                       padding='SAME')
                BN = layers.batch_norm()
                conv4b = BN(conv4b)
                conv4b = layers.lrelu(conv4b)


            with tf.variable_scope('conv4c'):
                conv4c = layers.conv2d(layerName='vgg19_conv4c',
                                       t_input=conv4b,
                                       ksize=[3, 3, 512, 512],
                                       strides=[1, 1, 1, 1],
                                       padding='SAME')
                BN = layers.batch_norm()
                conv4c = BN(conv4c)
                conv4c = layers.lrelu(conv4c)


            with tf.variable_scope('conv4d'):
                conv4d = layers.conv2d(layerName='vgg19_conv4d',
                                       t_input=conv4c,
                                       ksize=[3, 3, 512, 512],
                                       strides=[1, 1, 1, 1],
                                       padding='SAME')
                BN = layers.batch_norm()
                conv4d = BN(conv4d)
                conv4d = layers.lrelu(conv4d)

            phi.append(conv4d)

            pool_fourth = layers.maxPool(
                                        layerName='pool_first',
                                        t_input=conv4d,
                                        ksize=[1, 2, 2, 1],
                                        strides=[1, 2, 2, 1],
                                        padding='SAME'
                                        )



            with tf.variable_scope('conv5a'):
                conv5a = layers.conv2d(layerName='vgg19_conv5a',
                                       t_input=pool_fourth,
                                       ksize=[3, 3, 512, 512],
                                       strides=[1, 1, 1, 1],
                                       padding='SAME')
                BN = layers.batch_norm()
                conv5a = BN(conv5a)
                conv5a = layers.lrelu(conv5a)

            with tf.variable_scope('conv5b'):
                conv5b = layers.conv2d(layerName='vgg19_conv5b',
                                       t_input=conv5a,
                                       ksize=[3, 3, 512, 512],
                                       strides=[1, 1, 1, 1],
                                       padding='SAME')
                BN = layers.batch_norm()
                conv5b = BN(conv5b)
                conv5b = layers.lrelu(conv5b)

            with tf.variable_scope('conv5c'):
                conv5c = layers.conv2d(layerName='vgg19_conv5c',
                                       t_input=conv5b,
                                       ksize=[3, 3, 512, 512],
                                       strides=[1, 1, 1, 1],
                                       padding='SAME')
                BN = layers.batch_norm()
                conv5c = BN(conv5c)
                conv5c = layers.lrelu(conv5c)

            with tf.variable_scope('conv5d'):
                conv5d = layers.conv2d(layerName='vgg19_conv5d',
                                       t_input=conv5c,
                                       ksize=[3, 3, 512, 512],
                                       strides=[1, 1, 1, 1],
                                       padding='SAME')
                BN = layers.batch_norm()
                conv5d = BN(conv5d)
                conv5d = layers.lrelu(conv5d)

            phi.append(conv5d)

            pool_fifth = layers.maxPool(
                                        layerName='pool_first',
                                        t_input=conv5d,
                                        ksize=[1, 2, 2, 1],
                                        strides=[1, 2, 2, 1],
                                        padding='SAME'
                                        )

            flatten = layers.flatten(t_input=pool_fifth, flatDim=512)

            with tf.variable_scope('fc1'):
                fc1 = layers.fullyConnected(flatten, [512, 4096], layerName='fc1')
                fc1 = layers.lrelu(fc1)
            with tf.variable_scope('fc2'):
                fc2 = layers.fullyConnected(fc1, [4096, 4096], layerName='fc2')
                fc2 = layers.lrelu(fc2)
            with tf.variable_scope('softmax'):
                fc3 = layers.fullyConnected(fc2, [4096, 10], layerName='fc3')
            return fc3, phi


    def inference_loss(self, out, t):
        cross_entropy = tf.nn.softmax_cross_entropy_with_logits(
            labels=t,
            #labels=tf.one_hot(t, 10), <-원래이거씀
            logits=out)
        return tf.reduce_mean(cross_entropy)


if __name__=="__main__":
       from tensorflow.examples.tutorials.mnist import input_data
       mnist = input_data.read_data_sets('MNIST_data', one_hot = True)

       X_data = tf.placeholder(tf.float32, shape = [None, 28*28])
       X_image = tf.reshape(X_data, [-1, 28, 28, 1])
       Y_label = tf.placeholder(tf.float32, shape = [None, 10])
       tf.summary.image('input', X_image, 12)
       TRIAN_LOG_DIR = './logs/train'
       TEST_LOG_DIR = './logs/test'
       CHECK_POINT_DIR = './checkpoint'

       vgg= VGG19(None,None)
       logits, _=vgg.build_model(X_image, False)

       cost = vgg.inference_loss(out=logits, t=Y_label)
       train =layers.trainOptimizer(cost = cost)

       with tf.name_scope('accuracy'):
           prediction = tf.equal(tf.argmax(logits, 1) , tf.argmax(Y_label, 1))
           accuracy = tf.reduce_mean(tf.cast(prediction, tf.float32))
           tf.summary.scalar(name = 'accuracy', tensor = accuracy)

       merged = tf.summary.merge_all()

       with tf.Session() as sess:
           # last_epoch
           # 복귀시킬 epoch 를 선언.
           # tf.전역변수초기화하기 전 선언시켜둬야함.
           last_epoch = tf.Variable(0, name='last_epoch')

           sess.run(tf.global_variables_initializer())
           writer_test = tf.summary.FileWriter(logdir = TEST_LOG_DIR, graph = sess.graph)
           writer_train = tf.summary.FileWriter(logdir = TRIAN_LOG_DIR, graph = sess.graph)
           # tf.summary.FileWriter(logdir = DIRECTORY , graph = sess.graph)
           # 또는
           # tf.summary.FileWriter(logdir = DIRECTORY)
           # tf.summary.add_graph(sess.graph)

           saver = layers.restoreSaver(session = sess,
                               check_dir = CHECK_POINT_DIR)
           start_from = sess.run(last_epoch)
           epoch_run = 1000
           batch_size = 100

           total_batch = int(mnist.train.num_examples / batch_size)
           for epoch in range(start_from, epoch_run):
               for step in range(total_batch):
                   print("epoch:",epoch," step:",step)
                   batch = mnist.train.next_batch(batch_size=batch_size)
                   sess.run(train, feed_dict={X_data:batch[0],
                                              Y_label:batch[1]})

               # Exhaused Memory Error
               # batch size modified
               # test data(10000) -> 분할해야함.
               # train data - 분할

               import random
               random_batch = random.randint(0, len(mnist.test.images)-1000)

               summary, acc = sess.run([merged, accuracy], feed_dict={X_data: mnist.test.images[random_batch:random_batch+1000],
                                                           Y_label: mnist.test.labels[random_batch:random_batch+1000]})
               print('===EPOCH %4s===' % (epoch))
               writer_test.add_summary(summary, global_step=epoch)
               print("[Test]Accuracy at epoch %s: %s" % (epoch, acc))

               summary, acc = sess.run([merged, accuracy],
                                       feed_dict={X_data: batch[0],
                                                  Y_label: batch[1]})
               writer_train.add_summary(summary, global_step=epoch)
               print("[Training]Accuracy at epoch %s: %s" % (epoch, acc))

               saver = layers.saveSaver(session=sess,
                                 last_epoch=last_epoch,
                                 global_step=epoch,
                                 saver=saver,
                                 check_dir=CHECK_POINT_DIR,
                                 model_name='test_model'
                                 )