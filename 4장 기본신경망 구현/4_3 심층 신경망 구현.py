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

W1 = tf.Variable(tf.random_uniform([2,10], -1., 1.))
W2 = tf.Variable(tf.random_uniform([10,3], -1., 1.))


b1 = tf.Variable(tf.zeros([10]))
b2 = tf.Variable(tf.zeros([3]))

L1 = tf.add(tf.matmul(X,W1), b1)
L1 = tf.nn.relu(L1)


model = tf.add(tf.matmul(L1,W2), b2)

#텐서에서 제공하는 크로스 엔트로피 함수 이용
cost = tf.reduce_mean(
    tf.nn.softmax_cross_entropy_with_logits(labels = Y, logits = model))


optimizer = tf.train.AdamOptimizer(learning_rate= 0.01)
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