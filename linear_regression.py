# -*- coding: utf-8 -*-

import tensorflow as tf
import numpy  # ベクトル演算を出来るようにするための科学計算パッケージ

# def lin_model(X, w, b):
#     return X * w + b

learning_rate = 0.04  # cost = 384699424.000000000
training_epochs = 1000  # 最急降下法のイテレーション回数
display_step = 50

# 物件の専有面積(m2)
train_X = numpy.asarray([10, 11.55, 20.46, 23.89, 23.89, 44.36, 43.79, 40.8, 26.35, 22.4, 26.6, 37.6, 39.09, 54.51, 56.45, 40.23, 40.23, 42.71, 42.71, 42.71, 37.6, 36.79, 15.5, 25.11, 22.43, 21.42, 22.04, 28, 27.64, 57.78])
# 物件の月額家賃(円)
train_Y = numpy.asarray([30000, 31000, 58000, 72000, 74000, 136000, 137000, 164000, 109000, 112000, 133000, 143000, 155000, 205000, 206000, 130000, 131000, 133000, 134000, 134000, 143000, 143000, 58000, 95000, 107000, 80000, 89000, 105000, 110000, 150000])
n_samples = train_X.shape[0]

X = tf.placeholder("float")
Y = tf.placeholder("float")

# Set model weights
W = tf.Variable(numpy.random.randn(), name="weight")
b = tf.Variable(numpy.random.randn(), name="bias")

# Construct a linear model
activation = tf.add(tf.mul(X, W), b)

cost = tf.reduce_sum(tf.pow(activation-Y, 2))/(2*n_samples)  # 二乗誤差
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)  # 最急降下法

# Initializing the variables
init = tf.initialize_all_variables()

with tf.Session() as sess:
    sess.run(init)

    for epoch in range(training_epochs):
        for (x, y) in zip(train_X, train_Y):
            sess.run(optimizer, feed_dict={X: x, Y: y})

        if epoch % display_step == 0:
            print "Epoch:", '%04d' % (epoch+1), "cost=", "{:.9f}".format(sess.run(cost, feed_dict={X: train_X, Y:train_Y})), \
                "W=", sess.run(W), "b=", sess.run(b)

    print "Optimization Finished!"
    training_cost = sess.run(cost, feed_dict={X: train_X, Y: train_Y})
    print "Training cost=", training_cost, "W=", sess.run(W), "b=", sess.run(b), '\n'
