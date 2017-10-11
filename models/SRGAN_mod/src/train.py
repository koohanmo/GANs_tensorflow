import os
import numpy as np
import tensorflow as tf
import cv2
import matplotlib.pyplot as plt
from tqdm import tqdm
from srgan import SRGAN
import load
from PIL import Image

learning_rate = 1e-3
batch_size = 32
vgg_model = '../vgg19/backup/latest'

def train():
    x = tf.placeholder(tf.float32, [None, 96, 96, 3])
    #x_= tf.placeholder(tf.float32, [None, 24, 24, 3])
    is_training = tf.placeholder(tf.bool, [])

    model = SRGAN(x, is_training, batch_size)
    sess = tf.Session()
    with tf.variable_scope('srgan'):
        global_step = tf.Variable(0, name='global_step', trainable=False)
    opt = tf.train.AdamOptimizer(learning_rate=learning_rate)
    g_train_op = opt.minimize(
        model.g_loss, global_step=global_step, var_list=model.g_variables)
    d_train_op = opt.minimize(
        model.d_loss, global_step=global_step, var_list=model.d_variables)
    init = tf.global_variables_initializer()
    sess.run(init)

    # Restore the VGG-19 network
    var = tf.global_variables()
    vgg_var = [var_ for var_ in var if "vgg19" in var_.name]
    saver = tf.train.Saver(vgg_var)
    saver.restore(sess, vgg_model)
    print('restore vgg19....')

    # Restore the SRGAN network
    if tf.train.get_checkpoint_state('backup/'):
        saver = tf.train.Saver()
        saver.restore(sess, 'backup/latest')
        print('restore srgan.....')
    else:
        print('srgan-save file is empty... ')

    # Load the data
    x_train, x_test = load.load()

    # Train the SRGAN model
    n_iter = int(len(x_train) / batch_size)
    while True:
        epoch = int(sess.run(global_step) / n_iter / 2) + 1
        print('epoch:', epoch)
        np.random.shuffle(x_train)
        for i in tqdm(range(n_iter)):
            x_batch = normalize(x_train[i*batch_size:(i+1)*batch_size])
            sess.run(
                [g_train_op, d_train_op],
                feed_dict={x: x_batch, is_training: True})

        # Validate
        raw = normalize(x_test[:batch_size])
        mos, fake = sess.run(
            [model.downscaled, model.imitation],
            feed_dict={x: raw, is_training: False})
        save_img([mos, fake, raw], ['Input', 'Output', 'Ground Truth'], epoch)

        # Save the model
        saver = tf.train.Saver()
        saver.save(sess, 'backup/latest', write_meta_graph=False)

def train_conan():
    x = tf.placeholder(tf.float32, [None, 96, 96, 3])

    is_training = tf.placeholder(tf.bool, [])

    model = SRGAN(x, is_training, batch_size)
    sess = tf.Session()
    with tf.variable_scope('srgan'):
        global_step = tf.Variable(0, name='global_step', trainable=False)
    opt = tf.train.AdamOptimizer(learning_rate=learning_rate)
    g_train_op = opt.minimize(
        model.g_loss, global_step=global_step, var_list=model.g_variables)
    d_train_op = opt.minimize(
        model.d_loss, global_step=global_step, var_list=model.d_variables)
    init = tf.global_variables_initializer()
    sess.run(init)

    # Restore the VGG-19 network
    var = tf.global_variables()
    vgg_var = [var_ for var_ in var if "vgg19" in var_.name]
    saver = tf.train.Saver(vgg_var)
    saver.restore(sess, vgg_model)
    print('restore vgg19....')

    # Restore the SRGAN network
    if tf.train.get_checkpoint_state('backup/'):
        saver = tf.train.Saver()
        saver.restore(sess, 'backup/latest')
        print('restore srgan.....')
    else:
        print('srgan-save file is empty... ')

    # Load the conan data
    x_train, x_test = load.load_conan()

    # Train the SRGAN model
    n_iter = int(len(x_train) / batch_size)
    while True:
        epoch = int(sess.run(global_step) / n_iter / 2) + 1
        print('epoch:', epoch)
        np.random.shuffle(x_train)
        for i in tqdm(range(n_iter)):
            x_batch = normalize(x_train[i*batch_size:(i+1)*batch_size])
            sess.run(
                [g_train_op, d_train_op],
                feed_dict={x: x_batch, is_training: True})

        # Validate
        raw = normalize(x_test[:batch_size])
        mos, fake = sess.run(
            [model.downscaled, model.imitation],
            feed_dict={x: raw, is_training: False})
        save_img([mos, fake, raw], ['Input', 'Output', 'Ground Truth'], epoch-138+435)

        # Save the model
        saver = tf.train.Saver()
        saver.save(sess, 'backup/latest', write_meta_graph=False)

def save_img(imgs, label, epoch):
    for i in range(batch_size):
        fig = plt.figure()
        for j, img in enumerate(imgs):
            im = np.uint8((img[i]+1)*127.5)
            im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
            fig.add_subplot(1, len(imgs), j+1)
            plt.imshow(im)
            plt.tick_params(labelbottom='off')
            plt.tick_params(labelleft='off')
            plt.gca().get_xaxis().set_ticks_position('none')
            plt.gca().get_yaxis().set_ticks_position('none')
            plt.xlabel(label[j])
        seq_ = "{0:09d}".format(i+1)
        epoch_ = "{0:09d}".format(epoch)
        path = os.path.join('result', seq_, '{}.jpg'.format(epoch_))
        if os.path.exists(os.path.join('result', seq_)) == False:
            os.mkdir(os.path.join('result', seq_))
        plt.savefig(path)
        plt.close()

def normalize(images):
    return np.array([image/127.5-1 for image in images])

def video_info(infilename):
    """

    :param infilename:
        videofile
    :return:
        h, w, l, fps
    """
    cap = cv2.VideoCapture(infilename)

    if not cap.isOpened():
        print("could not open", infilename)
        exit(0)

    length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    return height, width, length, fps

def video2frame(invideofilename, save_path):
    """

    :param invideofilename:
        input video
    :param save_path:
        save_path is HR_path
    :return:
    """
    vidcap = cv2.VideoCapture(invideofilename)
    count = 0
    while True:
        success, image = vidcap.read()
        if not success:
            break
        print('Read a new frame: ', success)
        fname = "{}.png".format("{0:08d}".format(count))
        cv2.imwrite(save_path + '/'+ fname, image)  # save frame as JPEG file
        count += 1
    print("{} images are extracted in {}.".format(count, save_path))

def divide2merge(crop_path, save_path, save_name, img_size, grid):
    """

    :param size:
    :param number:
    :param grid:
    grid[0] is height
    grid[1] is width
    :return:
    """
    # LR
    merge_img = Image.new('RGB', img_size)
    merge_img_list = []
    for i in range(0, int(int(img_size[1]/grid) * int(img_size[0]/grid))):
        name = "{}.png".format("{0:08d}".format(i))
        merge_img_list.append(crop_path + '/' + name)

    for w in range(int(img_size[1]/grid)):
        for h in range(int(img_size[0]/grid)):
            bbox = (h * grid, w * grid, (h + 1) * (grid), (w + 1) * (grid))
            # bbox = (w * grid_w, h * grid_h, (w + 1) * (grid_w), (h + 1) * (grid_h))
            # 가로 세로 시작 가로 세로 끝
            file_cnt = (int(img_size[0]/grid)) * w + h
            #print('image number is ' + str(file_cnt))
            temp_merge_img = Image.open(merge_img_list[file_cnt])
            merge_img.paste(temp_merge_img, bbox)

    #merge_img.show()
    merge_img.save(save_path + '/' + save_name)

def lr_frame_crop(inimgfile, save_file_name, hr_crop_save_path,lr_save_path, lr_crop_save_path):

    """
    hr_frame to lr_frame, crop-frame(hr,lr)
    :param inimgfile:
    :param hr_save_path:
    :param lr_save_path:
    :return:
    """
    from resizeimage import resizeimage
    img = Image.open(inimgfile + '/' + save_file_name)
    (img_h, img_w) = img.size
    grid_w = 96
    grid_h = 96

    range_w = (int)(img_w / 96)
    range_h = (int)(img_h / 96)

    #print(range_w, range_h)

    i = 0
    lr_img = resizeimage.resize_thumbnail(img, (img_h/4, img_w/4))
    lr_img.save(lr_save_path + '/' + save_file_name)

    from resizeimage import resizeimage
    for w in range(range_w):
        for h in range(range_h):
            bbox = (h * grid_h, w * grid_w, (h + 1) * (grid_h), (w + 1) * (grid_w))
            #print(h * grid_h, w * grid_w, (h + 1) * (grid_h), (w + 1) * (grid_w))
            # 가로 세로 시작, 가로 세로 끝
            crop_img = img.crop(bbox)
            fname = "{}.png".format("{0:08d}".format(i))
            savename = hr_crop_save_path + '/' + fname
            crop_img.save(savename)

            #print('save file ' + savename + '....')
            hr_img = Image.open(savename)
            lr_img = resizeimage.resize_thumbnail(hr_img, (24, 24))
            lr_img.save(lr_crop_save_path + '/' + fname)
            i += 1

def frame2video(start_f, end_f, height, width, fps, dir_path, save_path):
    cnt = start_f

    fourcc = cv2.VideoWriter_fourcc(*'MP4V')  # codec 이상시 output 안나옴 : 일치하는 codec 설정해야함.
    out = cv2.VideoWriter(save_path, fourcc, fps, (width, height))  # 뽑아낼 video 설정

    while (True):
        fname = "{}.png".format("{0:08d}".format(cnt))
        cap = cv2.VideoCapture(dir_path + '/' + fname)
        if (not cap.isOpened): break
        ret, frame = cap.read()
        if (ret == True):
            out.write(frame)
            cv2.imshow('frame', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            break
        cnt += 1
        if(cnt == end_f): break

    cap.release()
    out.release()
    cv2.destroyAllWindows()


def video2sr(invideo):
    """
    png file {%08d}.png
    o 1. video_info
    o 2. extract video to HR-frame
    o 3. HR-frame to LR-frame
        --unit frame--
    o 4. crop LR-frame (size)
    o 5. super-resolution crop-LR-frame to crop-SR-frame

    o 6. ith sr-frame merge from crop-SR-frames to SR-frame

    video merge from LR-frames to LR-video
    video merge from SR-frames to SR-video
    """
    height, width, length, fps = video_info(invideo)
    print(height,width,length,fps)
    length = 2800
    #video2frame(invideo, 'H:/SRGAN_DC/CONAN/HR')

    import numpy as np
    import scipy.misc
    import cv2
    import os
    import tensorflow as tf

    x = tf.placeholder(tf.float32, [None, 24, 24, 3])
    is_training = tf.placeholder(tf.bool, [])

    model = SRGAN(x, is_training, batch_size)
    sess = tf.Session()

    init = tf.global_variables_initializer()
    sess.run(init)

    print('restore srgan')
    # Restore the SRGAN network
    if tf.train.get_checkpoint_state('backup/'):
        saver = tf.train.Saver()
        saver.restore(sess, 'backup/latest')


    C_SRpath = 'H:/SRGAN_DC/CONAN/CROP/SR'
    M_SRpath = 'H:/SRGAN_DC/CONAN/SR'

    C_LRpath = 'H:/SRGAN_DC/CONAN/CROP/LR'
    M_LRpath = 'H:/SRGAN_DC/CONAN/LR'

    C_HRpath = 'H:/SRGAN_DC/CONAN/CROP/HR'


    print('starting to convert LR to SR')
    import time

    for frame_idx in range(842, length):
        start = time.time()
        frame_name = "{}.png".format("{0:08d}".format(frame_idx))
        lr_frame_crop('H:/SRGAN_DC/CONAN/HR', frame_name, C_HRpath, M_LRpath ,C_LRpath)
        for i in range(int(int(height/96) * int(width/96))):

            name = "{}.png".format("{0:08d}".format(i))
            imgpath = os.path.join(C_LRpath, name)
            img = cv2.imread(imgpath) # load img
            #print(imgpath)
            face = img[:, :]
            face = (face / (255. / 2.)) - 1
            input_ = np.zeros((batch_size, 24, 24, 3)) # batchsize , height, width, rgb
            input_[0] = face

            #####################

            mos, fake = sess.run(
                [model.downscaled, model.imitation],
                feed_dict={x: input_, is_training: False})

            saveSR = os.path.join(C_SRpath, name)
            SRimage = np.uint8((fake[0]+1)*(255. / 2.))
            SRimage = cv2.cvtColor(SRimage, cv2.COLOR_BGR2RGB)
            scipy.misc.imsave(saveSR, SRimage)
        divide2merge(C_SRpath, M_SRpath, frame_name, (width, height), 96)
        process_time = (time.time() - start)
        print(str(process_time),'sec')
        print('save merge sr-frame', frame_name)

def LRSRHR(s, e):
    LRpath = 'H:/SRGAN_DC/CONAN/LR'
    SRpath = 'H:/SRGAN_DC/CONAN/SR'
    HRpath = 'H:/SRGAN_DC/CONAN/HR'
    LRSRHRpath = 'H:/SRGAN_DC/CONAN/LRSRHR'

    from resizeimage import resizeimage
    for i in range(s,e):
        merge_img = Image.new('RGB', (1920 * 3, 1080))
        frame_name = "{}.png".format("{0:08d}".format(i))
        jpg_name = "{}.jpg".format("{0:08d}".format(i))
        merge_list = []
        merge_list.append(LRpath + '/' + frame_name)
        merge_list.append(SRpath + '/' + frame_name)
        merge_list.append(HRpath + '/' + frame_name)

        for f in range(3):

            bbox = tuple((f * 1920, 0, (f+1)*1920, 1080))
            print(bbox)
            #bbox = (0, f * 1920, 1080, (f + 1) * 1920)
            img = Image.open(merge_list[f])
            img.save(LRSRHRpath + '/TEMP' + jpg_name)
            img = Image.open(LRSRHRpath + '/TEMP' + jpg_name)
            cimg = img.resize((1920, 1080))
            print(cimg.size)
            merge_img.paste(cimg, bbox)
        print(i,"frame")
        merge_img.save(LRSRHRpath + '/' + frame_name)


if __name__ == '__main__':
    #train()
    #image_crop('H:/mod_t_conan.png')
    #test_image_crop('H:/mod_t_conan.jpg')
    #divide2merge()
    #full2divide('H:/mod_t_conan_2.jpg')
    #full240divide('H:/t_960.jpg')
    #test_image_crop()
    #test_hr()
    #test_lr()

    #train_conan()
    #sr96()
    #divide2merge(96, 24)

    #sr24()
    #divide2merge(24, 24)

    #video2sr('H:/SRGAN_DC/3.mkv')
    #LRSRHR(514, 2799)
    frame2video(514, 2798, 1080, 1920*3, 23.9, 'H:/SRGAN_DC/CONAN/LRSRHR', 'H:/SRGAN_DC/CONAN/output_lrsrhr.mp4')
    #frame2video(514, 2799,1080,1920,23.9,'H:/SRGAN_DC/CONAN/SR','H:/SRGAN_DC/CONAN/output_sr.mp4')
    #frame2video(514, 2799, 270, 480, 23.9, 'H:/SRGAN_DC/CONAN/LR', 'H:/SRGAN_DC/CONAN/output_lr.mp4')
    #frame2video(514,2799,1080,1920,23.9,'H:/SRGAN_DC/CONAN/HR','H:/SRGAN_DC/CONAN/output_hr.mp4')