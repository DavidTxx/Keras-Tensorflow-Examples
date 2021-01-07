import tensorflow as tf 

x = tf.constant([[1., 2., 3.],
				 [4., 5., 6.],
				 [7., 8., 9.]])

print(x)

x = tf.reshape(x, [1, 3, 3, 1])

print(x) 

max_pool_2d = tf.keras.layers.MaxPooling2D(pool_size=(2,2),
	strides = (1,1), padding='valid')

max_pool_2d(x)