import tensorflow as tf
import numpy as np

#기본 데이터 준비
x_data = np.array([
    [0,0],
    [1,0],
    [1,1],
    [0,0],
    [0,0],
    [0,1]])
y_data = np.array([
    [1,0,0],
    [0,1,0],
    [0,0,1],
    [1,0,0],
    [1,0,0],
    [0,0,1]
])

#신경망 만들기
X = tf.placeholder(tf.float32)
Y = tf.placeholder(tf.float32)

W = tf.Variable(tf.random_uniform([2,3], -1., 1.))
b = tf.Variable(tf.zeros([3]))

L = tf.add(tf.matmul(X,W), b)
L = tf.nn.relu(L)

model = tf.nn.softmax(L)
#axis = 1의 의미,  R은 apply에서 행열 순서였다면, python은 0:열 1:행으로 이해
#실제값 Y와 softmax값 model을 곱해준 결과를 행별로 합해준다. --> reduce_sum
cost = tf.reduce_mean(-tf.reduce_sum(Y*tf.log(model), axis = 1))

optimizer = tf.train.GradientDescentOptimizer(learning_rate= 0.1)
train_op = optimizer.minimize(cost)

#텐서플로우 세션 초기화
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

#학습을 100번 진행
for step in range(100):
    sess.run(train_op, feed_dict={ X: x_data, Y: y_data})

    #학습 도중 10번에 한번씩 손실값 출력
    if (step+1) % 10 == 0 :
        print(step+1, sess.run(cost, feed_dict={X: x_data, Y:y_data}))

#학습된 결과 확인
prediction = tf.argmax(model, axis = 1)
target = tf.argmax(Y, axis = 1)
print('예측값: ', sess.run(prediction, feed_dict={X:x_data}))
print('실제값: ', sess.run(target, feed_dict={Y:y_data}))

is_correct = tf.equal(prediction, target)
acc = tf.reduce_mean(tf.cast(is_correct, tf.float32))
print('정확도: %.2f' %sess.run(acc*100, feed_dict={X:x_data, Y:y_data}))