import tensorflow as tf
import numpy as np

x_data = np.array([
    [0,0],
    [0,1],
    [1,0],
    [1,1]])
y_data = np.array([
    [0],[1],[1],[0]
])

X = tf.placeholder(tf.float32, [None, 2])
Y= tf.placeholder(tf.float32, [None, 1])

W1 = tf.Variable(tf.random_uniform([2,4], -1., 1.))
W2 = tf.Variable(tf.random_uniform([4,1], -1., 1.))
b1 = tf.Variable(tf.zeros([4]))
b2 = tf.Variable(tf.zeros([1]))

layer1 = tf.matmul(X,W1) + b1
layer1 = tf.sigmoid(layer1)

layer2 = tf.matmul(layer1,W2) + b2
model = tf.sigmoid(layer2)

'''
cost = tf.reduce_mean(
    tf.nn.softmax_cross_entropy_with_logits(labels = Y, logits = model))
'''
cost = tf.reduce_mean(tf.reduce_sum(tf.square(model-Y), 1))


optimizer = tf.train.AdamOptimizer(0.01)
train_op = optimizer.minimize(cost)


#텐서플로우 세션 초기화
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

#학습을 100번 진행
for step in range(1000):
    sess.run(train_op, feed_dict={ X: x_data, Y: y_data})

    if step % 100 == 0:
        #학습 도중 100번에 한번씩 손실값 출력
        print(step+1, sess.run(cost, feed_dict={X: x_data, Y:y_data}))

#학습된 결과 확인
prediction = tf.floor(model + 0.5)
correct = tf.equal(prediction, Y)

print('예측값: ', sess.run(prediction, feed_dict={X:x_data,Y:y_data}))
print('실제값: ', sess.run(Y, feed_dict={X:x_data,Y:y_data}))


acc = tf.reduce_mean(tf.cast(correct, tf.float32))
print('정확도: %.2f' %sess.run(acc*100, feed_dict={X:x_data, Y:y_data}))
