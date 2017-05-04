import tensorflow as tf
import variables

class batch_norm(object):
    """
    Batch normalization Layer 
    origin source : https://github.com/carpedm20/DCGAN-tensorflow
    참고 : https://www.tensorflow.org/api_docs/python/tf/contrib/layers/batch_norm
    Ex) bn = batch_norm('sample layer')
        afterBN = bn(preBN_variable)
    """
    def __init__(self, epsilon=1e-5, momentum=0.9, name="batch_norm"):
        """
        :param epsilon: 
        Small float added to variance to avoid dividing by zero.
        
        :param momentum:
        Decay for the moving average. 
        Reasonable values for decay are close to 1.0, typically in the multiple-nines range: 0.999, 0.99, 0.9, etc. 
        Lower decay value (recommend trying decay=0.9) if model experiences reasonably good training performance but poor validation and/or test performance.
        Try zero_debias_moving_mean=True for improved stability.
        
        :param name: 
        Variable name
        """
        with tf.variable_scope(name):
            self.epsilon = epsilon
            self.momentum = momentum
            self.name = name

    def __call__(self, x, train=True):
        """
        call tf.contrib.layers.batch_norm
        :param x: 
        :param train:
        Whether or not the layer is in training mode. 
        In training mode it would accumulate the statistics of the moments into moving_mean and moving_variance using an exponential moving average with the given decay. 
        When it is not in training mode then it would use the values of the moving_mean and the moving_variance.
        
        :return:  
        """
        return tf.contrib.layers.batch_norm(x,
                                                decay=self.momentum,
                                                updates_collections=None,
                                                epsilon=self.epsilon,
                                                scale=True,
                                                is_training=train,
                                                scope=self.name)


def conv2d(input,ksize,strides,padding,layerName,is_batch_norm=True,initializer=variables.variable_random,act=tf.nn.relu):
    """
    Convolution Layer
    :param input:
     Input tensor([batch_size, height, width, channel])
    :param ksize:
     Kernel(filter) size
     Ex) [1,2,2,1]
    :param strides: 
     Stride dim
     Ex) [1,2,2,1]
    :param padding: 
    Padding Type
     Ex) 'SAME' or 'VALID'
    :param layerName:
     Tensorboard name
    :param is_batch_norm
     True or False
     Apply batch_normalization?
     default = True
    :param initializer:
     Default : xavier
    :return: 
     Output tensor
    """
    with tf.name_scope(layerName):
        with tf.name_scope('weight'):
            W = initializer(name=layerName, shape=ksize)
            #variables.variable_summaries(name=W) # initialize 에서 board에 체크
        with tf.name_scope('activation'):
            output = tf.nn.conv2d(input = input, filter = W, strides = strides, padding = padding)
            if(is_batch_norm):
                batch_normalization = batch_norm(name = layerName+'_Batch_norm')
                batch_normalization(output)
            output = act(output)
        print(output)
        return output


def maxPool(input,ksize,strides,padding,layerName):
    """
    MaxPool Layer
    :param input: 
     Input tensor
    :param ksize:
     kernel(filter) size
    :param strides: 
     Stride dim
    :param padding:
     Padding Type
     Ex) 'SAME' or 'VALID'
    :param layerName:
     Tensorboard name
    :return: 
     Output tensor
    """
    with tf.name_scope(layerName):
        output = tf.nn.max_pool(input, ksize = ksize, strides=strides, padding = padding)
        variables.variable_summaries(name=layerName, var=output)
    print(output)
    return output


def avgPool(input,ksize,strides,padding,layerName):
    """
    AvgPool Layer
    :param input: 
     Input tensor
    :param ksize:
     kernel(filter) size
    :param strides: 
     Stride dim
    :param padding:
     Padding Type
     Ex) 'SAME' or 'VALID'
    :param layerName:
     Tensorboard name
    :return: 
     Output tensor
    """
    with tf.name_scope(layerName):
        output = tf.nn.avg_pool(input,
                                ksize = ksize,
                                strides= strides,
                                padding = padding)
    print(output)
    return output

def flatten(input, flatDim, layerName='flattenLayer'):
    """
    Flatten Layer
    :param input:
     Input tensor
    :param flatDim:
     Output tensor dim
    :param layerName:
     Tensorboard name
    :return:
     Output tensor
    """
    with tf.name_scope(layerName):
        flatten = tf.reshape(input, [-1, flatDim])
        variables.variable_summaries(name=layerName, var=flatten)
    print(flatten)
    return flatten

def nnLayer(input,outputSize,layerName,initializer=variables.variable_xavier):
    """
    이름이 안떠올라요 도와주세요....
    WX+B
    :param input:
     Input tensor
    :param outputSize:
      Output tensor dim
    :param layerName: 
     Tensorboard name
    :param initializer:
     Default : xavier
    :return:
     Output tensor
    """
    pass

def fullyConnected(input, shape, layerName, is_batch_norm = True, initializer = variables.variable_xavier,act=tf.nn.relu):
    """

    :param input:
        input tensor
    :param shape:
        dimension
    :param layerName:
        layer Name
    :param initializer:
        init
        default = variables.variable_xavier
    :param batch_norm:
        True or False
        Apply batch_normalization?
        default = True
    :param act:
        activation function
        default = tf.nn.relu
    :return:
        output act tensor
    """
    with tf.name_scope(layerName):
        with tf.name_scope('weight'):
            W = initializer(name=layerName+'/weight',shape = shape)
            #variables.variable_summaries(W)
        with tf.name_scope('bias'):
            B = initializer(name=layerName+'/bias',shape = shape[-1])
            #variables.variable_summaries(B)
        with tf.name_scope('preActivate'):
            preActivate = tf.matmul(input, W) + B
            tf.summary.histogram(name=layerName+'/preActivate', values=preActivate)
        with tf.name_scope('activation'):
            if (is_batch_norm):
                batch_normalization = batch_norm(name = layerName + '_Batch_norm')
                batch_normalization(preActivate)
            activations = act(preActivate)
    print(activations)
    return activations

def crossEntropy(labels, logits, name = 'lossFunction'):
    '''
    Cross Entropy
    :param labels:
     labels = training or test data
    :param logits:
     logits = outputLayer
    :param name:
     name = crossEntropy Name
     default = 'crossEntropy'
    :return:
     Output cost function
    '''
    with tf.name_scope(name):
        cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels = labels, logits = logits))
        tf.summary.scalar(name=name,tensor=cost)
    return cost

def trainOptimizer(cost, learning_rate = 1e-3, optimizer = tf.train.AdamOptimizer, name = 'train'):
    '''

    :param cost:
        cost function
    :param learning_rate:
        learning_rate
    :param optimizer:
        default = AdamOptimizer
    :param name:
        trainName
    :return:
        optimize
    '''
    with tf.name_scope(name):
        optimize = optimizer(learning_rate = learning_rate).minimize(cost)
    return optimize

def restoreSaver(session,check_dir):
    """
    (*) 실행하기 전 last_epoch = tf.Variable(0, name='last_epoch') 선언
    (*) session.run(global_variables_initializer()) 하기 전에 선언시켜줘야함.
    (*) 복귀된 epoch는 session.run(last_epoch)의 리턴값으로 얻어짐.
        Restore Saver
    :param session:
        현재 실행시킨 session
    :param check_dir:
        저장된 디렉토리 경로
    :return:
        세이버 리턴
    """
    saver = tf.train.Saver()
    check_point = tf.train.get_checkpoint_state(check_dir)

    if check_point and check_point.model_checkpoint_path:
        try:
            saver.restore(session, check_point.model_checkpoint_path)
            print('successfully loaded : ', check_point.model_checkpoint_path)
        except:
            print('fail to load')
    else:
        print('could not find load data')

    return saver

def saveSaver(session, last_epoch ,global_step ,saver, check_dir, model_name = 'model'):
    """
        체크포인트 세이브
    :param session:
        현재 세션
    :param last_epoch:
        마지막 epoch
        last_epoch - tf.Variable
    :param global_step:
        현재 global_step
    :param saver:
        저장시킬 saver
    :param check_dir:
        저장시킬 directory - restoreSaver 의 check_dir 과 같음.
    :param model_name:
        저장시킬 모델이름.
        default = 'model'
    :return:
        세이버 리턴
    """
    import os
    if not os.path.exists(check_dir):
        os.makedirs(check_dir)
    try:
        session.run(last_epoch.assign(global_step+1))
        saver.save(sess=session,save_path = check_dir+'/'+model_name, global_step = global_step)
        print('step',global_step,'successfully save dir :', check_dir+'/'+model_name+'-'+str(global_step))
    except:
        print('fail to save')

    return saver

if __name__=="__main__":
    """
    Test code...
    MNIST CNN source code 에 layers 적용.
    GPU 메모리 부족현상이 발생해서 test 배치사이즈를 순차적인 1000개를 임의적으로 선택.

    세이버 저장/불러오기추가.
    """
    from tensorflow.examples.tutorials.mnist import input_data
    mnist = input_data.read_data_sets('MNIST_data', one_hot = True)

    X_data = tf.placeholder(tf.float32, shape = [None, 28*28])
    X_image = tf.reshape(X_data, [-1, 28, 28, 1])
    Y_label = tf.placeholder(tf.float32, shape = [None, 10])
    tf.summary.image('input', X_image, 12)
    TRIAN_LOG_DIR = './logs/train'
    TEST_LOG_DIR = './logs/test'
    # path == userpath
    # tensorboard --logdir = path/GANs_tensorflow/modelHelper/logs/test
    # tensorboard --logdir = path/GANs_tensorflow/modelHelper/logs/train
    # cd GANs_tensorflow\modelHelper\logs
    CHECK_POINT_DIR = './checkpoint'
    # userpath/checkpoint/*model-*

    convLayer1 = conv2d(X_image,
                        ksize = [3,3,1,32],
                        strides = [1,1,1,1],
                        padding='SAME',
                        layerName='convLayer1',
                        is_batch_norm=True
                        )


    maxpoolLayer1 = maxPool(convLayer1,
                            ksize = [1,2,2,1],
                            strides=[1,2,2,1],
                            padding='SAME',
                            layerName='maxpoolLayer1',
                            )

    convLayer2 = conv2d(maxpoolLayer1,
                        ksize=[3, 3, 32, 64],
                        strides=[1, 1, 1, 1],
                        padding='SAME',
                        layerName='convLayer2',
                        is_batch_norm=True)

    maxpoolLayer2 = maxPool(convLayer2,
                            ksize=[1, 2, 2, 1],
                            strides=[1, 2, 2, 1],
                            layerName='maxpoolLayer2',
                            padding='SAME')

    flattenLayer = flatten(maxpoolLayer2, 7*7*64)

    fcLayer1 = fullyConnected(input=flattenLayer,
                              shape=[7*7*64, 100],
                              layerName='fullyConnected1',
                              is_batch_norm=True
                              )

    fcLayer2 = fullyConnected(input=fcLayer1,
                              shape=[100, 50],
                              layerName='fullyConnected2',
                              is_batch_norm=True
                              )

    fcLayer3 = fullyConnected(input=fcLayer2,
                              shape=[50, 25],
                              layerName='fullyConnected3',
                              is_batch_norm=True
                              )

    with tf.name_scope('logits'):
        W = variables.variable_xavier('outW', [25,10])
        B = variables.variable_random('outB', [10])
        logits = tf.matmul(fcLayer3, W) + B
        cost = crossEntropy(labels = Y_label, logits = logits)

    train = trainOptimizer(cost = cost)

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

        saver = restoreSaver(session = sess,
                            check_dir = CHECK_POINT_DIR)
        start_from = sess.run(last_epoch)
        epoch_run = 1000
        batch_size = 1000

        total_batch = int(mnist.train.num_examples / batch_size)
        for epoch in range(start_from, epoch_run):
            for step in range(total_batch):
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

            saver = saveSaver(session=sess,
                              last_epoch=last_epoch,
                              global_step=epoch,
                              saver=saver,
                              check_dir=CHECK_POINT_DIR,
                              model_name='test_model'
                              )

