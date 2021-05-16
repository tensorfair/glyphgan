import glob
import os
import numpy as np
import tensorflow as tf
os.environ['CUDA_VISIBLE_DEVICES'] = '0'


# ----- To repress Tensorflow deprecation warnings ----- #
from warnings import simplefilter
simplefilter(action='ignore', category=FutureWarning)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
old_v = tf.compat.v1.logging.get_verbosity()
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)


def process_image(feed, width, height):
    """
    :param feed: image/s location.
    :param width: width of image.
    :param height: height of image.
    :return: This function returns the processed and resized image data.
    """
    image_string = tf.read_file(feed)
    image_decoded = tf.io.decode_image(image_string, 3, expand_animations=False)
    image_decoded = tf.image.resize(image_decoded, [height, width])
    image_decoded.set_shape((height, width, 3))
    return image_decoded


def build_inputs(specs, input_type, shuffle=True, num_threads=4):
    """
    :param specs: dictionary of info on input and output data.
    :param input_type: the folder location of ground truth images.
    :param shuffle: randomize batches.
    :param num_threads: tf.train.batch threads to use.
    :return: This function returns a tensor batch queue from files provided.
    """
    batch_dir = dict()
    for character in range(26):
        feed = np.array(glob.glob(os.path.join(os.path.join(input_type, '%d' % character), '**', '*.*'), recursive=True))
        train_list = tf.constant(feed)
        filename_queue = tf.train.string_input_producer(train_list, shuffle=shuffle)
        image = process_image(filename_queue.dequeue(), specs['x_size'], specs['y_size'])
        image_batch = tf.train.batch([image], batch_size=specs['batch_size'], num_threads=num_threads, capacity=specs['batch_size']*2)
        batch_dir['%d' % character] = image_batch
    return batch_dir


def prepare_input(batch_size, character):
    """
    :param batch_size: number of images to feed to graph at a time.
    :param character: value between 0 and 25 to indicate character.
    :return: This function returns a one-hot encoding of the character label concatenated with latent space.
    """
    z_style = np.random.uniform(-1, 1, size=(batch_size, 100)).astype(np.float32)
    one_hot = []
    for i in range(26):
        one_hot.append(0 if i != character else 1)
    z_character = np.array([one_hot, ]*batch_size).astype(np.float32)
    z_character = z_character.reshape([batch_size, 26])
    prepared_input = np.concatenate([z_style, z_character], 1)
    prepared_input = prepared_input.reshape([batch_size, 1, 1, 126])
    return prepared_input


def build_log_dir(main_name, path_name=None, second_path_name=None):
    """
    :param main_name: top folder name.
    :param path_name: child folder name.
    :param second_path_name: child within a child folder name.
    :return: Returns path to main folder and creates the folder.
    """
    log_path = main_name
    if second_path_name is not None:
        log_path = os.path.join(log_path, path_name)
        log_path = os.path.join(log_path, second_path_name)
    elif path_name is not None:
        log_path = os.path.join(log_path, path_name)
    if not os.path.exists(log_path):
        os.makedirs(log_path)
    return log_path


def tf_config_setup():
    """
    :return: Returns tensorflow configuration.
    """
    tf_config = tf.ConfigProto(allow_soft_placement=False)
    tf_config.gpu_options.allow_growth = True
    tf_config.gpu_options.polling_inactive_delay_msecs = 50
    return tf_config


def initialization(sess):
    """
    :param sess: tensorflow active session.
    :return: Returns the initialized tensorflow session.
    """
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)
    sess.run(tf.local_variables_initializer())
    sess.run(tf.global_variables_initializer())
    return coord, threads
