import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("/tmp/data/", one_hot = True)
import os
import cv2
import numpy as np

#n_classes = 2
#n_classes = 8
n_classes = 10

def list_images(directory):
    """
    Get all the images and labels in directory/label/*.jpg
    """
    labels = os.listdir(directory)
    # Sort the labels so that training and validation get them in the same order
    labels.sort()

    files_and_labels = []
    for label in labels:
        for f in os.listdir(os.path.join(directory, label)):
            files_and_labels.append((os.path.join(directory, label, f), label))

    filenames, labels = zip(*files_and_labels)
    filenames = list(filenames)
    labels = list(labels)
    unique_labels = list(set(labels))

    label_to_int = {}
    for i, label in enumerate(unique_labels):
        label_to_int[label] = i#tf.one_hot(i, n_classes)

    labels = [label_to_int[l] for l in labels]

    return filenames, labels

#batch_size = 16
batch_size = 128

def _input_parser(img_path, label):
    # convert the label to one-hot encoding
    one_hot = tf.one_hot(label, n_classes)

    # read the img from file
    img_file = tf.read_file(img_path)
    img_decoded = tf.image.decode_jpeg(img_file, channels=3)
    img_decoded = tf.image.resize_images(img_decoded, [300,300])
    #img_decoded = tf.reshape(img_decoded, shape=[300 * 300 * 3])

    return img_decoded, one_hot

#train_filenames, train_labels = list_images("weija/train")
#val_filenames, val_labels = list_images("weija/val")

train_filenames, train_labels = list_images("coco-animals/train")
val_filenames, val_labels = list_images("coco-animals/val")




train_dataset = tf.data.Dataset.from_tensor_slices((tf.constant(train_filenames), tf.constant(train_labels)))
train_dataset = train_dataset.map(_input_parser)
batched_train_dataset = train_dataset.batch(batch_size)

val_dataset = tf.data.Dataset.from_tensor_slices((tf.constant(val_filenames), tf.constant(val_labels)))
val_dataset = val_dataset.map(_input_parser)
batched_val_dataset = val_dataset.batch(batch_size)

iterator = tf.data.Iterator.from_structure(batched_train_dataset.output_types, batched_train_dataset.output_shapes)
train_init_op = iterator.make_initializer(batched_train_dataset)
val_init_op = iterator.make_initializer(batched_val_dataset)


# iterator = tf.contrib.data.Iterator.from_structure(train_dataset.output_types, train_dataset.output_shapes)
# train_init_op = iterator.make_initializer(train_dataset)
# val_init_op = iterator.make_initializer(val_dataset)

# Indicates whether we are in training or in test mode
is_training = tf.placeholder(tf.bool)

next_element = iterator.get_next()

# x = tf.placeholder('float', [None, 270000])
x = tf.placeholder('float', [None, 784])
# x = tf.placeholder('float', [None, 300, 300, 3])
y = tf.placeholder('float')

gpu_options = tf.GPUOptions(allow_growth=True)
sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
#sess.run(tf.global_variables_initializer())

# sess.run(train_init_op)
# while True:
#     try:
#         elem = sess.run(next_element)
#         print(elem[0].shape)
#     except tf.errors.OutOfRangeError:
#         print("End of training dataset.")
#         break


keep_rate = 0.8
keep_prob = tf.placeholder(tf.float32)


def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1,1,1,1], padding='SAME')

def maxpool2d(x):
    return tf.nn.max_pool(x, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')

def convolutional_neural_network(x):
    #weights = {'W_conv1':tf.Variable(tf.random_normal([5,5,3,32])),
    weights = {'W_conv1':tf.Variable(tf.random_normal([5,5,1,32])),
               'W_conv2':tf.Variable(tf.random_normal([5,5,32,64])),
               #'W_fc':tf.Variable(tf.random_normal([75*75*64,1024])),
               'W_fc':tf.Variable(tf.random_normal([7*7*64,1024])),
               'out':tf.Variable(tf.random_normal([1024, n_classes]))}

    biases = {'b_conv1':tf.Variable(tf.random_normal([32])),
               'b_conv2':tf.Variable(tf.random_normal([64])),
               'b_fc':tf.Variable(tf.random_normal([1024])),
               'out':tf.Variable(tf.random_normal([n_classes]))}

    x = tf.reshape(x, shape=[-1, 28, 28, 1])
    #x = tf.reshape(x, shape=[-1, 300, 300, 3])

    conv1 = tf.nn.relu(conv2d(x, weights['W_conv1']) + biases['b_conv1'])
    conv1 = maxpool2d(conv1)

    conv2 = tf.nn.relu(conv2d(conv1, weights['W_conv2']) + biases['b_conv2'])
    conv2 = maxpool2d(conv2)

    fc = tf.reshape(conv2,[-1, 7*7*64])
    #fc = tf.reshape(conv2,[-1, 75*75*64]) #for some reason we divide initial width by 4?
    fc = tf.nn.relu(tf.matmul(fc, weights['W_fc'])+biases['b_fc'])
    fc = tf.nn.dropout(fc, keep_rate)
    #dim of fc is 1 x 1024?

    #so dim of output is 1xn_classes?
    #bias is also 1xn_classes
    #we want output to be 1x2 right
    output = tf.matmul(fc, weights['out'])+biases['out']

    return output

def train_neural_network(x):
    prediction = convolutional_neural_network(x)
    cost = tf.reduce_mean( tf.nn.softmax_cross_entropy_with_logits(logits=prediction,labels=y) )
    optimizer = tf.train.AdamOptimizer().minimize(cost)


    sess.run(tf.global_variables_initializer())

    hm_epochs = 10

    for epoch in range(hm_epochs):
        epoch_loss = 0

        """

        sess.run(train_init_op)
        while True:
            try:
                epoch_x, epoch_y = sess.run(next_element)
                print(type(epoch_x), type(epoch_y), epoch_x.shape, epoch_y.shape)

                _, c = sess.run([optimizer, cost], feed_dict={x: epoch_x, y: epoch_y})
                epoch_loss += c
            except tf.errors.OutOfRangeError:
                break

	"""

        for _ in range(int(mnist.train.num_examples/batch_size)):
            epoch_x, epoch_y = mnist.train.next_batch(batch_size)
            _, c = sess.run([optimizer, cost], feed_dict={x: epoch_x, y: epoch_y})
            epoch_loss += c

        print('Epoch', epoch + 1, 'completed out of',hm_epochs,'loss:',epoch_loss)

    correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
    a = sess.run([accuracy], {x:mnist.test.images, y:mnist.test.labels})
    print('Accuracy:', a[0])


    """
    sess.run(val_init_op)
    total = 0
    score = 0
    while True:
	try:
	    epoch_x, epoch_y = sess.run(next_element)
	    (n, _) = epoch_y.shape
       	    a = sess.run([accuracy], feed_dict={x: epoch_x, y: epoch_y})
	    total += n
	    score += a[0] * n
	except tf.errors.OutOfRangeError:
	    break

    print score / total
    """

train_neural_network(x)
