# -*- coding: utf-8 -*-

# Sample code to use string producer.
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plot


def one_hot(x, n):
    """
    :param x: label (int)
    :param n: number of bits
    :return: one hot code
    """
    o_h = np.zeros(n)
    o_h[x] = 1.
    return o_h


def chart(a):
    plot.title('Convolutive practice - Validation error')
    plot.ylabel('Errors')
    plot.xlabel('Epoch')
    valid_handle, = plot.plot(a)
    plot.legend(handles=[valid_handle],
                labels=['Validation error'])
    plot.savefig('./charts/Grafica.png')
    plot.show()


num_classes = 3
batch_size = 10


# --------------------------------------------------
#
#       DATA SOURCE
#
# --------------------------------------------------

def dataSource(paths, batch_size):
    min_after_dequeue = 10
    capacity = min_after_dequeue + 3 * batch_size

    example_batch_list = []
    label_batch_list = []

    for i, p in enumerate(paths):
        filename = tf.train.match_filenames_once(p)
        filename_queue = tf.train.string_input_producer(filename, shuffle=False)
        reader = tf.WholeFileReader()
        _, file_image = reader.read(filename_queue)
        image, label = tf.image.decode_jpeg(file_image), one_hot(i, num_classes)  # [float(i)]
        image = tf.image.resize_image_with_crop_or_pad(image, 80, 140)
        image = tf.image.rgb_to_grayscale(image, name=None)
        image = tf.reshape(image, [80, 140, 1])
        image = tf.to_float(image) / 255. - 0.5
        example_batch, label_batch = tf.train.shuffle_batch([image, label], batch_size=batch_size, capacity=capacity,
                                                            min_after_dequeue=min_after_dequeue)
        example_batch_list.append(example_batch)
        label_batch_list.append(label_batch)

    example_batch = tf.concat(values=example_batch_list, axis=0)
    label_batch = tf.concat(values=label_batch_list, axis=0)

    return example_batch, label_batch


# --------------------------------------------------
#
#       MODEL
#
# --------------------------------------------------

def myModel(X, reuse=False):
    with tf.variable_scope('ConvNet', reuse=reuse):
        o1 = tf.layers.conv2d(inputs=X, filters=32, kernel_size=3, activation=tf.nn.relu)
        o2 = tf.layers.max_pooling2d(inputs=o1, pool_size=2, strides=2)
        o3 = tf.layers.conv2d(inputs=o2, filters=64, kernel_size=3, activation=tf.nn.relu)
        o4 = tf.layers.max_pooling2d(inputs=o3, pool_size=2, strides=2)

        h = tf.layers.dense(inputs=tf.reshape(o4, [batch_size * 3, 18 * 33 * 64]), units=5, activation=tf.nn.relu)
        y = tf.layers.dense(inputs=h, units=num_classes, activation=tf.nn.softmax)
    return y


example_batch_train, label_batch_train = dataSource(
    ["data3/train/0/*.jpg", "data3/train/1/*.jpg", "data3/train/2/*.jpg"],
    batch_size=batch_size)
example_batch_valid, label_batch_valid = dataSource(
    ["data3/valid/0/*.jpg", "data3/valid/1/*.jpg", "data3/valid/2/*.jpg"],
    batch_size=batch_size)
example_batch_test, label_batch_test = dataSource(["data3/test/0/*.jpg", "data3/test/1/*.jpg", "data3/test/2/*.jpg"],
                                                  batch_size=batch_size)

example_batch_train_predicted = myModel(example_batch_train, reuse=False)
example_batch_valid_predicted = myModel(example_batch_valid, reuse=True)
example_batch_test_predicted = myModel(example_batch_test, reuse=True)

cost = tf.reduce_sum(tf.square(example_batch_train_predicted - tf.cast(label_batch_train, tf.float32)))
cost_valid = tf.reduce_sum(tf.square(example_batch_valid_predicted - tf.cast(label_batch_valid, tf.float32)))
# cost = tf.reduce_mean(-tf.reduce_sum(label_batch * tf.log(y), reduction_indices=[1]))
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001).minimize(cost)

# --------------------------------------------------
#
#       TRAINING
#
# --------------------------------------------------

# Add ops to save and restore all the variables.

saver = tf.train.Saver()

with tf.Session() as sess:
    file_writer = tf.summary.FileWriter('./logs', sess.graph)

    sess.run(tf.local_variables_initializer())
    sess.run(tf.global_variables_initializer())

    # Start populating the filename queue.
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord, sess=sess)

    currentValidError = 0.0
    previousValidError = 10000.0
    listvalidErrors = []
    diffError = 1
    epoch = 0

    # for epoch in range(200):
    while diffError > 0.001:
        sess.run(optimizer)
        previousValidError = currentValidError
        currentValidError = sess.run(cost_valid)
        listvalidErrors.append(currentValidError)
        diffError = abs(previousValidError - currentValidError)
        epoch += 1
        if epoch % 20 == 0:
            print("Iter:" + str(epoch) + "---------------------------------------------")
            print(sess.run(label_batch_valid))
            print(sess.run(example_batch_valid_predicted))
            print("Error: " + str(sess.run(cost)))
            # print("Error:", sess.run(cost_valid))

    result2 = []
    label = []
    total = 0.0
    error = 0.0
    epoch =0

    for epoch in range(10):
        result = sess.run(example_batch_test_predicted)
        lab = sess.run(label_batch_test)
        result2.extend(result)
        label.extend(lab)
    for b, r in zip(label, result2):
        if np.argmax(b) != np.argmax(r):
            print("Next has a error")
            error += 1
        print(b, "-->", r)
        total += 1

    print("Error del " + str(round((error / total) * 100, 2)) + " %")
    print ("Total: " + str(total))
    print ("Errors: " + str(error))
    save_path = saver.save(sess, "./tmp/model.ckpt")
    print("Model saved in file: %s" % save_path)

    chart(listvalidErrors)

    coord.request_stop()
    coord.join(threads)
