import tensorflow as tf
import time

###################################################################################################
#                                                                                                 #
#                              Functions to train the network                                     #
#                                                                                                 #
###################################################################################################


def ae_loss(logits, images):
    """
    This function computes the autoencoder quadratique loss
    :param logits: output of the autoencoder
    :param images: input
    :return: loss
    """
    return tf.reduce_mean(tf.pow(logits - images, 2))


def ae_train(model_path, inference, filename, batch_size, num_epochs, learning_rate, device_name, sum_path=None,
          finetune=None):
    """
    This function trains the autoencoder
    :param model_path: path where the learned filters will be saved
    :param inference: the autoencoder architecture
    :param filename: input database is a tfrecord
    :param batch_size: batch_size
    :param num_epochs: num_epochs
    :param learning_rate: learning rate
    :param device_name: gpu or cpu
    :param sum_path: path for the summary to display tensorboard
    :param finetune: default None, if True the the weights are initialized from the given path
    :return: Trained autoencoder
    """
    with tf.Graph().as_default():
        images, labels = input_data(filename, batch_size, num_epochs)
        global_step = tf.Variable(0, name='global_step', trainable=False)
        with tf.device(device_name):
            logits = inference(images)
            cost = ae_loss(logits, images)
            optimizer = tf.train.GradientDescentOptimizer(learning_rate)
            train_op = optimizer.minimize(cost, global_step=global_step)
            init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
        tf.summary.scalar('loss', cost)
        saver = tf.train.Saver()
        sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
        if sum_path is not None:
            merged = tf.summary.merge_all()
            train_writer = tf.summary.FileWriter(sum_path + '/train', sess.graph)
        if finetune is None:
            sess.run(init_op)
        else:
            saver.restore(sess, finetune)
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)
        try:
            step = 0
            start_time = time.time()
            while not coord.should_stop():
                if sum_path is not None:
                    summary, _, loss_value = sess.run([merged, train_op, cost])
                    if step % 1000 == 0:
                        train_writer.add_summary(summary, step)
                else:
                    _, loss_value = sess.run([train_op, cost])
                duration = time.time() - start_time
                if step % 5000 == 0:
                    save_path = saver.save(sess, model_path + "/model", global_step=step)
                    print("Model saved in file: %s" % save_path)
                if step % 100 == 0:
                    print('Step %d: loss = %.2f (%.3f sec)' % (step, loss_value, duration))
                step += 1
        except tf.errors.OutOfRangeError:
            print('Done training for %d epochs, %d steps.' % (num_epochs, step))
        finally:
            coord.request_stop()
        if sum_path is not None:
            train_writer.close()
        coord.join(threads)
        sess.close()


def ae_feed(data_in, model_path, inference, nbr_input):
    with tf.Graph().as_default():
        X = tf.placeholder(tf.float64, [None, nbr_input])
        hidden = inference(X)
        saver = tf.train.Saver()
        with tf.Session() as sess:
            saver.restore(sess, model_path)
            h = sess.run([hidden], feed_dict={X: data_in})
    return h

###################################################################################################
#                                                                                                 #
#                              Functions to create architecture                                   #
#                                                                                                 #
###################################################################################################


def encoder(x, nbr_input, nbr_hidden1, nbr_hidden2, nbr_hidden3, nbr_hidden4):
    weights = {
        'encoder_h1': tf.Variable(tf.random_normal([nbr_input, nbr_hidden1], dtype=tf.float64)),
        'encoder_h2': tf.Variable(tf.random_normal([nbr_hidden1, nbr_hidden2], dtype=tf.float64)),
        'encoder_h3': tf.Variable(tf.random_normal([nbr_hidden2, nbr_hidden3], dtype=tf.float64)),
        'encoder_h4': tf.Variable(tf.random_normal([nbr_hidden3, nbr_hidden4], dtype=tf.float64)),
    }
    biases = {
        'encoder_b1': tf.Variable(tf.random_normal([nbr_hidden1], dtype=tf.float64)),
        'encoder_b2': tf.Variable(tf.random_normal([nbr_hidden2], dtype=tf.float64)),
        'encoder_b3': tf.Variable(tf.random_normal([nbr_hidden3], dtype=tf.float64)),
        'encoder_b4': tf.Variable(tf.random_normal([nbr_hidden4], dtype=tf.float64)),
    }
    encoder_1 = tf.nn.tanh(tf.add(tf.matmul(x, weights['encoder_h1']), biases['encoder_b1']))
    encoder_2 = tf.nn.tanh(tf.add(tf.matmul(encoder_1, weights['encoder_h2']), biases['encoder_b2']))
    encoder_3 = tf.nn.tanh(tf.add(tf.matmul(encoder_2, weights['encoder_h3']), biases['encoder_b3']))
    encoder_4 = tf.nn.tanh(tf.add(tf.matmul(encoder_3, weights['encoder_h4']), biases['encoder_b4']))
    return encoder_4


def decoder(x, nbr_input, nbr_hidden1, nbr_hidden2, nbr_hidden3, nbr_hidden4):
    weights = {
        'decoder_h1': tf.Variable(tf.random_normal([nbr_hidden4, nbr_hidden3], dtype=tf.float64)),
        'decoder_h2': tf.Variable(tf.random_normal([nbr_hidden3, nbr_hidden2], dtype=tf.float64)),
        'decoder_h3': tf.Variable(tf.random_normal([nbr_hidden2, nbr_hidden1], dtype=tf.float64)),
        'decoder_h4': tf.Variable(tf.random_normal([nbr_hidden1, nbr_input], dtype=tf.float64)),
    }
    biases = {
        'decoder_b1': tf.Variable(tf.random_normal([nbr_hidden3], dtype=tf.float64)),
        'decoder_b2': tf.Variable(tf.random_normal([nbr_hidden2], dtype=tf.float64)),
        'decoder_b3': tf.Variable(tf.random_normal([nbr_hidden1], dtype=tf.float64)),
        'decoder_b4': tf.Variable(tf.random_normal([nbr_input], dtype=tf.float64)),
    }
    decoder_1 = tf.nn.tanh(tf.add(tf.matmul(x, weights['decoder_h1']), biases['decoder_b1']))
    decoder_2 = tf.nn.tanh(tf.add(tf.matmul(decoder_1, weights['decoder_h2']), biases['decoder_b2']))
    decoder_3 = tf.nn.tanh(tf.add(tf.matmul(decoder_2, weights['decoder_h3']), biases['decoder_b3']))
    decoder_4 = tf.nn.tanh(tf.add(tf.matmul(decoder_3, weights['decoder_h4']), biases['decoder_b4']))
    return decoder_4

###################################################################################################
#                                                                                                 #
#                              Functions to get data and plots                                    #
#                                                                                                 #
###################################################################################################


def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def read_and_decode(filename_queue):
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)
    features = tf.parse_single_example(
      serialized_example,
      features={
          'width': tf.FixedLenFeature([], tf.int64),
          'height': tf.FixedLenFeature([], tf.int64),
          'image_raw': tf.FixedLenFeature([], tf.string),
          'label': tf.FixedLenFeature([], tf.int64),
      })
    image = tf.decode_raw(features['image_raw'], tf.float64)
    image.set_shape([28 * 28])
    label = tf.cast(features['label'], tf.int32)
    return image, label


def input_data(filename, batch_size, num_epochs=1):
    with tf.name_scope('input'):
        filename_queue = tf.train.string_input_producer([filename], num_epochs=num_epochs)
        image, label = read_and_decode(filename_queue)
        images, sparse_labels = tf.train.shuffle_batch([image, label], batch_size=batch_size, num_threads=2,
                                                       capacity=1000 + 3 * batch_size, min_after_dequeue=1000)
    return images, sparse_labels
