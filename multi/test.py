import tensorflow as tf

n = 10
#images = [[[[x * n + y + 1] for y in range(n)] for x in range(n)]]
#patch = tf.extract_image_patches(images=images, ksizes=[1, 3, 3, 1], strides=[1, 1, 1, 1], rates=[1, 1, 1, 1], padding='SAME')
#patch = tf.reshape(patch, [-1, 9])
#print(patch)
#w = tf.constant(1, shape = [1,9])
#print(w)
#result = tf.multiply(patch, w)
b = tf.constant(1, shape=[28, 28])
b1 = tf.stack([b for i in range(9)], axis=-1)
print(b1.shape)
for i in range(9):
    b1 = tf.stack([b1, b], axis = 2)
    print(b1.shape)

with tf.Session() as sess:
    sess.run(b)
    print(b)
