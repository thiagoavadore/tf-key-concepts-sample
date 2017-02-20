import tensorflow as tf
import helperVisualisation
from tensorflow.contrib.learn.python.learn.datasets.mnist import read_data_sets
tf.set_random_seed(916)

# Download images and labels into mnist.test and mnist.train
mnist = read_data_sets("data", one_hot=True, reshape=False, validation_size=0)

# You can index images on the batches by putting the first dimension NONE
X = tf.placeholder(tf.float32, [None, 28, 28, 1])

Y_ = tf.placeholder(tf.float32, [None, 10])

W = tf.Variable(tf.zeros([784, 10]))

b = tf.Variable(tf.zeros([10]))

# Our data is not flattened yet... so we flat and call it XX
# Reshape to -1 means to select the dimension that will preserve the number of elements.
XX = tf.reshape(X, [-1, 784])

# We just saw the model!
Y = tf.nn.softmax(tf.matmul(XX, W) + b)

# Define the cross entropy
cross_entropy = -tf.reduce_mean(Y_ * tf.log(Y)) * 1000.0
# log of each tensor element
# * multiplies the tensors element by element
# The tf.reduce_mean wii all all this parts in the tensor.
# Thus, the end result is the cross entropy for all images in the batch (100 images)
# But doing the reduce_mean, you divided by 10 as well... that is why we multiply by 1000 (100 images plus the extra 10)

# accuracy of the trained model
correct_prediction = tf.equal(tf.argmax(Y, 1), tf.argmax(Y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# training, learning rate = 0.005
train_step = tf.train.GradientDescentOptimizer(0.005).minimize(cross_entropy)

# visulatisation
allweights = tf.reshape(W, [-1])
allbiases = tf.reshape(b, [-1])
I = helperVisualisation.tf_format_mnist_images(X, Y, Y_)  # assembles 10x10 images by default
It = helperVisualisation.tf_format_mnist_images(X, Y, Y_, 1000, lines=25)  # 1000 images on 25 lines
datavis = helperVisualisation.MnistDataVis()

# init
init = tf.initialize_all_variables()
sess = tf.Session()
sess.run(init)


# You can call this function in a loop to train the model, 100 images at a time
def training_step(i, update_test_data, update_train_data):

    # training on batches of 100 images with 100 labels
    batch_X, batch_Y = mnist.train.next_batch(100)

    # compute training values for visualisation
    if update_train_data:
        a, c, im, w, b = sess.run([accuracy, cross_entropy, I, allweights, allbiases], feed_dict={X: batch_X, Y_: batch_Y})
        datavis.append_training_curves_data(i, a, c)
        datavis.append_data_histograms(i, w, b)
        datavis.update_image1(im)
        print(str(i) + ": accuracy:" + str(a) + " loss: " + str(c))

    # compute test values for visualisation
    if update_test_data:
        a, c, im = sess.run([accuracy, cross_entropy, It], feed_dict={X: mnist.test.images, Y_: mnist.test.labels})
        datavis.append_test_curves_data(i, a, c)
        datavis.update_image2(im)
        print(str(i) + ": ********* epoch " + str(i*100//mnist.train.images.shape[0]+1) + " ********* test accuracy:" + str(a) + " test loss: " + str(c))

    # the backpropagation training step
    sess.run(train_step, feed_dict={X: batch_X, Y_: batch_Y})


datavis.animate(training_step, iterations=2000+1, train_data_update_freq=10, test_data_update_freq=50, more_tests_at_start=True)

# to save the animation as a movie, add save_movie=True as an argument to datavis.animate
# to disable the visualisation use the following line instead of the datavis.animate line
# for i in range(2000+1): training_step(i, i % 50 == 0, i % 10 == 0)

print("max test accuracy: " + str(datavis.get_max_test_accuracy()))

# final max test accuracy = 0.9268 (10K iterations). Accuracy should peak above 0.92 in the first 2000 iterations.
