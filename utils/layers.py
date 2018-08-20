import tensorflow as tf


# layer where the variables are created inside
#   |---------------------|
# conv -> relu -> conv ----> relu (size remains same)
def residual_layer_conv(input_x, filter_h_w=[3, 3]):
    # first convolution
    weight_conv1 = tf.get_variable('conv1', shape=[filter_h_w[0], filter_h_w[1], int(input_x.get_shape()[3]),
                                                   int(input_x.get_shape()[3])])
    conv1_out = tf.nn.conv2d(input_x, weight_conv1, strides=[1, 1, 1, 1], padding='SAME')
    bias_conv1 = tf.get_variable('bias1', shape=[int(input_x.get_shape()[3])], initializer=tf.zeros_initializer())
    layer1_out = tf.nn.relu(conv1_out + bias_conv1)
    # second convolution
    weight_conv2 = tf.get_variable('conv2', shape=[filter_h_w[0], filter_h_w[1], int(input_x.get_shape()[3]),
                                                   int(input_x.get_shape()[3])])
    conv2_out = tf.nn.conv2d(layer1_out, weight_conv2, strides=[1, 1, 1, 1], padding='SAME')
    bias_conv2 = tf.get_variable('bias2', shape=[int(input_x.get_shape()[3])], initializer=tf.zeros_initializer())
    layer2_out = conv2_out + bias_conv2
    # residual connection, followed by activation
    output_x = tf.nn.relu(input_x + layer2_out)
    return output_x

# layer where the variables are created inside
# residual layer along with projection to double the number of channels
def residual_layer_conv_projection(input_x, filter_h_w=[3, 3]):
    # first convolution
    weight_conv1 = tf.get_variable('conv1', shape=[filter_h_w[0], filter_h_w[1], int(input_x.get_shape()[3]),
                                                   int(input_x.get_shape()[3]) * 2])
    conv1_out = tf.nn.conv2d(input_x, weight_conv1, strides=[1, 2, 2, 1], padding='SAME')
    bias_conv1 = tf.get_variable('bias1', shape=[int(input_x.get_shape()[3]) * 2], initializer=tf.zeros_initializer())
    layer1_out = tf.nn.relu(conv1_out + bias_conv1)
    # second convolution
    weight_conv2 = tf.get_variable('conv2', shape=[filter_h_w[0], filter_h_w[1], int(input_x.get_shape()[3]) * 2,
                                                   int(input_x.get_shape()[3]) * 2])
    conv2_out = tf.nn.conv2d(layer1_out, weight_conv2, strides=[1, 1, 1, 1], padding='SAME')
    bias_conv2 = tf.get_variable('bias2', shape=[int(input_x.get_shape()[3]) * 2], initializer=tf.zeros_initializer())
    layer2_out = conv2_out + bias_conv2
    # residual connection, followed by activation
    input_projection_pool = tf.nn.max_pool(input_x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    weight_conv_projection = tf.get_variable('conv3',
                                             shape=[1, 1, int(input_x.get_shape()[3]), int(input_x.get_shape()[3]) * 2])
    conv2_projection = tf.nn.conv2d(input_projection_pool, weight_conv_projection, strides=[1, 1, 1, 1], padding='SAME')
    # output
    output_x = tf.nn.relu(conv2_projection + layer2_out)
    return output_x

def residual_layer_conv_projection_3x3(input_x, filter_h_w=[3, 3]):
    # first convolution
    weight_conv1 = tf.get_variable('conv1', shape=[filter_h_w[0], filter_h_w[1], int(input_x.get_shape()[3]),
                                                   int(input_x.get_shape()[3])])
    conv1_out = tf.nn.conv2d(input_x, weight_conv1, strides=[1, 3, 3, 1], padding='SAME')
    bias_conv1 = tf.get_variable('bias1', shape=[int(input_x.get_shape()[3])], initializer=tf.zeros_initializer())
    layer1_out = tf.nn.relu(conv1_out + bias_conv1)
    # second convolution
    weight_conv2 = tf.get_variable('conv2', shape=[filter_h_w[0], filter_h_w[1], int(input_x.get_shape()[3]),
                                                   int(input_x.get_shape()[3])])
    conv2_out = tf.nn.conv2d(layer1_out, weight_conv2, strides=[1, 1, 1, 1], padding='SAME')
    bias_conv2 = tf.get_variable('bias2', shape=[int(input_x.get_shape()[3])], initializer=tf.zeros_initializer())
    layer2_out = conv2_out + bias_conv2
    # residual connection, followed by activation
    input_projection_pool = tf.nn.max_pool(input_x, ksize=[1, 3, 3, 1], strides=[1, 3, 3, 1], padding='SAME')
    weight_conv_projection = tf.get_variable('conv3',
                                             shape=[1, 1, int(input_x.get_shape()[3]), int(input_x.get_shape()[3])])
    conv2_projection = tf.nn.conv2d(input_projection_pool, weight_conv_projection, strides=[1, 1, 1, 1], padding='SAME')
    # output
    output_x = tf.nn.relu(conv2_projection + layer2_out)
    return output_x


def residual_layer_conv_projection_const_channel(input_x, filter_h_w=[3, 3]):
    # first convolution
    weight_conv1 = tf.get_variable('conv1', shape=[filter_h_w[0], filter_h_w[1], int(input_x.get_shape()[3]),
                                                   int(input_x.get_shape()[3])])
    conv1_out = tf.nn.conv2d(input_x, weight_conv1, strides=[1, 2, 2, 1], padding='SAME')
    bias_conv1 = tf.get_variable('bias1', shape=[int(input_x.get_shape()[3])], initializer=tf.zeros_initializer())
    layer1_out = tf.nn.relu(conv1_out + bias_conv1)
    # second convolution
    weight_conv2 = tf.get_variable('conv2', shape=[filter_h_w[0], filter_h_w[1], int(input_x.get_shape()[3]),
                                                   int(input_x.get_shape()[3])])
    conv2_out = tf.nn.conv2d(layer1_out, weight_conv2, strides=[1, 1, 1, 1], padding='SAME')
    bias_conv2 = tf.get_variable('bias2', shape=[int(input_x.get_shape()[3])], initializer=tf.zeros_initializer())
    layer2_out = conv2_out + bias_conv2
    # residual connection, followed by activation
    input_projection_pool = tf.nn.max_pool(input_x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    weight_conv_projection = tf.get_variable('conv3',
                                             shape=[1, 1, int(input_x.get_shape()[3]), int(input_x.get_shape()[3])])
    conv2_projection = tf.nn.conv2d(input_projection_pool, weight_conv_projection, strides=[1, 1, 1, 1], padding='SAME')
    # output
    output_x = tf.nn.relu(conv2_projection + layer2_out)
    return output_x


# layer where the variables are created inside
# residual layer along with projection to half the number of channels
def residual_layer_conv_projection_half(input_x, filter_h_w=[3, 3]):
    # first convolution
    weight_conv1 = tf.get_variable('conv1', shape=[filter_h_w[0], filter_h_w[1], int(input_x.get_shape()[3]),
                                                   int(int(input_x.get_shape()[3]) / 2)])
    conv1_out = tf.nn.conv2d(input_x, weight_conv1, strides=[1, 2, 2, 1], padding='SAME')
    bias_conv1 = tf.get_variable('bias1', shape=[int(int(input_x.get_shape()[3]) / 2)],
                                 initializer=tf.zeros_initializer())
    layer1_out = tf.nn.relu(conv1_out + bias_conv1)
    # second convolution
    weight_conv2 = tf.get_variable('conv2', shape=[filter_h_w[0], filter_h_w[1], int(int(input_x.get_shape()[3]) / 2),
                                                   int(int(input_x.get_shape()[3]) / 2)])
    conv2_out = tf.nn.conv2d(layer1_out, weight_conv2, strides=[1, 1, 1, 1], padding='SAME')
    bias_conv2 = tf.get_variable('bias2', shape=[int(int(input_x.get_shape()[3]) / 2)],
                                 initializer=tf.zeros_initializer())
    layer2_out = conv2_out + bias_conv2
    # residual connection, followed by activation
    input_projection_pool = tf.nn.max_pool(input_x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    weight_conv_projection = tf.get_variable('conv3', shape=[1, 1, int(input_x.get_shape()[3]),
                                                             int(int(input_x.get_shape()[3]) / 2)])
    conv2_projection = tf.nn.conv2d(input_projection_pool, weight_conv_projection, strides=[1, 1, 1, 1], padding='SAME')
    # output
    output_x = tf.nn.relu(conv2_projection + layer2_out)
    return output_x


# layer where the variables are created inside
# deconv -> relu -> deconv ----> relu (size remains same)
def residual_layer_deconv(input_x, filter_h_w=[3, 3], relu_after=True):
    # first convolution
    weight_deconv1 = tf.get_variable('deconv1', shape=[filter_h_w[0], filter_h_w[1], int(input_x.get_shape()[3]),
                                                       int(input_x.get_shape()[3])])
    bias_deconv1 = tf.get_variable('bias1', shape=[int(input_x.get_shape()[3])], initializer=tf.zeros_initializer())
    deconv1_out = tf.nn.relu(bias_deconv1 + tf.nn.conv2d_transpose(input_x, weight_deconv1,
                                                                   [int(input_x.get_shape()[0]),
                                                                    int(input_x.get_shape()[1]),
                                                                    int(input_x.get_shape()[2]),
                                                                    int(input_x.get_shape()[3])], strides=[1, 1, 1, 1],
                                                                   padding='SAME'))
    # second convolution
    weight_deconv2 = tf.get_variable('deconv2', shape=[filter_h_w[0], filter_h_w[1], int(input_x.get_shape()[3]),
                                                       int(input_x.get_shape()[3])])
    bias_deconv2 = tf.get_variable('bias2', shape=[int(input_x.get_shape()[3])], initializer=tf.zeros_initializer())
    deconv2_out = bias_deconv2 + tf.nn.conv2d_transpose(deconv1_out, weight_deconv2,
                                                        [int(input_x.get_shape()[0]), int(input_x.get_shape()[1]),
                                                         int(input_x.get_shape()[2]), int(input_x.get_shape()[3])],
                                                        strides=[1, 1, 1, 1], padding='SAME')
    # residual connection, followed by activation
    if relu_after:
        output_x = tf.nn.relu(input_x + deconv2_out)
    else:
        output_x = input_x + deconv2_out
    return output_x

# layer where the variables are created inside
# residual layer to double the size of channel but half the number of channels 
def residual_layer_deconv_projection(input_x, filter_h_w=[3, 3], relu_after=True, output_shape=None):
    if output_shape==None:
        output_shape = [int(input_x.get_shape()[0]), 2*int(input_x.get_shape()[1]),
                        2*int(input_x.get_shape()[2]), int(int(input_x.get_shape()[3])/2)]
    # first convolution
    weight_deconv1 = tf.get_variable('deconv1',
                                     shape=[filter_h_w[0], filter_h_w[1], int(int(input_x.get_shape()[3]) / 2),
                                            int(input_x.get_shape()[3])])
    bias_deconv1 = tf.get_variable('bias1', shape=[int(int(input_x.get_shape()[3]) / 2)],
                                   initializer=tf.zeros_initializer())
    deconv1_out = tf.nn.relu(bias_deconv1 + tf.nn.conv2d_transpose(input_x, weight_deconv1, 
                                                                    output_shape,
                                                                    strides=[1, 2, 2, 1], 
                                                                    padding='SAME'))
    # second convolution
    weight_deconv2 = tf.get_variable('deconv2', shape=[filter_h_w[0], filter_h_w[1], int(input_x.get_shape()[3]) / 2,
                                                       int(input_x.get_shape()[3]) / 2])
    bias_deconv2 = tf.get_variable('bias2', shape=[int(input_x.get_shape()[3]) / 2], initializer=tf.zeros_initializer())
    deconv2_out = bias_deconv2 + tf.nn.conv2d_transpose(deconv1_out, weight_deconv2, 
                                                        output_shape, 
                                                        strides=[1, 1, 1, 1],
                                                        padding='SAME')
    # residual deconvolution
    weight_res1 = tf.get_variable('res1', shape=[filter_h_w[0], filter_h_w[1], int(int(input_x.get_shape()[3]) / 2),
                                                 int(input_x.get_shape()[3])])
    res1_out = tf.nn.conv2d_transpose(input_x, weight_res1,
                                        output_shape,
                                        strides=[1, 2, 2, 1], 
                                        padding='SAME')
    # residual connection, followed by activation
    if relu_after:
        output_x = tf.nn.relu(res1_out + deconv2_out)
    else:
        output_x = res1_out + deconv2_out
    return output_x

def residual_layer_deconv_projection3x3(input_x, filter_h_w=[3, 3], relu_after=True):
    # first convolution
    weight_deconv1 = tf.get_variable('deconv1',
                                     shape=[filter_h_w[0], filter_h_w[1], int(int(input_x.get_shape()[3])),
                                            int(input_x.get_shape()[3])])
    bias_deconv1 = tf.get_variable('bias1', shape=[int(int(input_x.get_shape()[3]))],
                                   initializer=tf.zeros_initializer())
    deconv1_out = tf.nn.relu(bias_deconv1 + tf.nn.conv2d_transpose(input_x, weight_deconv1,
                                                                   [int(input_x.get_shape()[0]),
                                                                    3 * int(input_x.get_shape()[1]),
                                                                    3 * int(input_x.get_shape()[2]),
                                                                    int(int(input_x.get_shape()[3]))],
                                                                   strides=[1, 3, 3, 1], padding='SAME'))
    # second convolution
    weight_deconv2 = tf.get_variable('deconv2', shape=[filter_h_w[0], filter_h_w[1], int(input_x.get_shape()[3]),
                                                       int(input_x.get_shape()[3])])
    bias_deconv2 = tf.get_variable('bias2', shape=[int(input_x.get_shape()[3])], initializer=tf.zeros_initializer())
    deconv2_out = bias_deconv2 + tf.nn.conv2d_transpose(deconv1_out, weight_deconv2,
                                                        [int(input_x.get_shape()[0]), 3 * int(input_x.get_shape()[1]),
                                                         3 * int(input_x.get_shape()[2]),
                                                         int(int(input_x.get_shape()[3]))], strides=[1, 1, 1, 1],
                                                        padding='SAME')
    # residual deconvolution
    weight_res1 = tf.get_variable('res1', shape=[filter_h_w[0], filter_h_w[1], int(int(input_x.get_shape()[3])),
                                                 int(input_x.get_shape()[3])])
    res1_out = tf.nn.conv2d_transpose(input_x, weight_res1,
                                      [int(input_x.get_shape()[0]), 3 * int(input_x.get_shape()[1]),
                                       3 * int(input_x.get_shape()[2]), int(int(input_x.get_shape()[3]))],
                                      strides=[1, 3, 3, 1], padding='SAME')
    # residual connection, followed by activation
    if relu_after:
        output_x = tf.nn.relu(res1_out + deconv2_out)
    else:
        output_x = res1_out + deconv2_out
    return output_x


def residual_layer_deconv_projection_const_channel(input_x, filter_h_w=[3, 3], relu_after=True):
    # first convolution
    weight_deconv1 = tf.get_variable('deconv1',
                                     shape=[filter_h_w[0], filter_h_w[1], int(int(input_x.get_shape()[3])),
                                            int(input_x.get_shape()[3])])
    bias_deconv1 = tf.get_variable('bias1', shape=[int(int(input_x.get_shape()[3]))],
                                   initializer=tf.zeros_initializer())
    deconv1_out = tf.nn.relu(bias_deconv1 + tf.nn.conv2d_transpose(input_x, weight_deconv1,
                                                                   [int(input_x.get_shape()[0]),
                                                                    2 * int(input_x.get_shape()[1]),
                                                                    2 * int(input_x.get_shape()[2]),
                                                                    int(int(input_x.get_shape()[3]))],
                                                                   strides=[1, 2, 2, 1], padding='SAME'))
    # second convolution
    weight_deconv2 = tf.get_variable('deconv2', shape=[filter_h_w[0], filter_h_w[1], int(input_x.get_shape()[3]),
                                                       int(input_x.get_shape()[3])])
    bias_deconv2 = tf.get_variable('bias2', shape=[int(input_x.get_shape()[3])], initializer=tf.zeros_initializer())
    deconv2_out = bias_deconv2 + tf.nn.conv2d_transpose(deconv1_out, weight_deconv2,
                                                        [int(input_x.get_shape()[0]), 2 * int(input_x.get_shape()[1]),
                                                         2 * int(input_x.get_shape()[2]),
                                                         int(int(input_x.get_shape()[3]))], strides=[1, 1, 1, 1],
                                                        padding='SAME')
    # residual deconvolution
    weight_res1 = tf.get_variable('res1', shape=[filter_h_w[0], filter_h_w[1], int(input_x.get_shape()[3]),
                                                 int(input_x.get_shape()[3])])
    res1_out = tf.nn.conv2d_transpose(input_x, weight_res1,
                                      [int(input_x.get_shape()[0]), 2 * int(input_x.get_shape()[1]),
                                       2 * int(input_x.get_shape()[2]), int(int(input_x.get_shape()[3]))],
                                      strides=[1, 2, 2, 1], padding='SAME')
    # residual connection, followed by activation
    if relu_after:
        output_x = tf.nn.relu(res1_out + deconv2_out)
    else:
        output_x = res1_out + deconv2_out
    return output_x


# layer where the variables are created inside
# residual layer to double the size of channel but double the number of channels 
def residual_layer_deconv_projection_double(input_x, filter_h_w=[3, 3], relu_after=True):
    # first convolution
    weight_deconv1 = tf.get_variable('deconv1',
                                     shape=[filter_h_w[0], filter_h_w[1], int(int(input_x.get_shape()[3]) * 2),
                                            int(input_x.get_shape()[3])])
    bias_deconv1 = tf.get_variable('bias1', shape=[int(int(input_x.get_shape()[3]) * 2)],
                                   initializer=tf.zeros_initializer())
    deconv1_out = tf.nn.relu(bias_deconv1 + tf.nn.conv2d_transpose(input_x, weight_deconv1,
                                                                   [int(input_x.get_shape()[0]),
                                                                    2 * int(input_x.get_shape()[1]),
                                                                    2 * int(input_x.get_shape()[2]),
                                                                    int(int(input_x.get_shape()[3]) * 2)],
                                                                   strides=[1, 2, 2, 1], padding='SAME'))
    # second convolution
    weight_deconv2 = tf.get_variable('deconv2', shape=[filter_h_w[0], filter_h_w[1], int(input_x.get_shape()[3]) * 2,
                                                       int(input_x.get_shape()[3]) * 2])
    bias_deconv2 = tf.get_variable('bias2', shape=[int(input_x.get_shape()[3]) * 2], initializer=tf.zeros_initializer())
    deconv2_out = bias_deconv2 + tf.nn.conv2d_transpose(deconv1_out, weight_deconv2,
                                                        [int(input_x.get_shape()[0]), 2 * int(input_x.get_shape()[1]),
                                                         2 * int(input_x.get_shape()[2]),
                                                         int(int(input_x.get_shape()[3]) * 2)], strides=[1, 1, 1, 1],
                                                        padding='SAME')
    # residual deconvolution
    weight_res1 = tf.get_variable('res1', shape=[filter_h_w[0], filter_h_w[1], int(int(input_x.get_shape()[3]) * 2),
                                                 int(input_x.get_shape()[3])])
    res1_out = tf.nn.conv2d_transpose(input_x, weight_res1,
                                      [int(input_x.get_shape()[0]), 2 * int(input_x.get_shape()[1]),
                                       2 * int(input_x.get_shape()[2]), int(int(input_x.get_shape()[3]) * 2)],
                                      strides=[1, 2, 2, 1], padding='SAME')
    # residual connection, followed by activation
    if relu_after:
        output_x = tf.nn.relu(res1_out + deconv2_out)
    else:
        output_x = res1_out + deconv2_out
    return output_x


def residual_layer_fully_connected(input_x, relu_after=True):
    # first fully connected layer
    fc1 = tf.get_variable('f1', shape=[int(input_x.get_shape()[1]), int(input_x.get_shape()[1])])
    bias1 = tf.get_variable('bias1', shape=[int(input_x.get_shape()[1])], initializer=tf.zeros_initializer())
    fc1_out = tf.nn.relu(tf.matmul(input_x, fc1) + bias1)
    # seccond fully connected layer
    fc2 = tf.get_variable('f2', shape=[int(input_x.get_shape()[1]), int(input_x.get_shape()[1])])
    bias2 = tf.get_variable('bias2', shape=[int(input_x.get_shape()[1])], initializer=tf.zeros_initializer())
    fc2_out = tf.matmul(fc1_out, fc2) + bias2
    # skip connection and relu
    layer_out = input_x + fc2_out
    if relu_after:
        layer_out = tf.nn.relu(layer_out)
    return layer_out


def deconvolution_layer(input_x, filter_shape, deconv_strides, padding, output_size, relu_after=True):
    weight_deconv1 = tf.get_variable('deconv1', shape=filter_shape)
    bias_deconv1 = tf.get_variable('bias1', shape=[filter_shape[2]], initializer=tf.zeros_initializer())
    if relu_after:
        deconv1_out = tf.nn.relu(
            bias_deconv1 + tf.nn.conv2d_transpose(input_x, weight_deconv1, output_size, strides=deconv_strides,
                                                  padding=padding))
    else:
        deconv1_out = bias_deconv1 + tf.nn.conv2d_transpose(input_x, weight_deconv1, output_size,
                                                            strides=deconv_strides, padding=padding)
    return deconv1_out


def convolution_layer(input_x, filter_shape, conv_strides, padding, relu_after=True):
    weight_conv1 = tf.get_variable('c1', shape=filter_shape)
    conv1_out = tf.nn.conv2d(input_x, weight_conv1, strides=conv_strides, padding=padding)
    bias_conv1 = tf.get_variable('bias1', shape=[filter_shape[3]], initializer=tf.zeros_initializer())
    if relu_after:
        layer1_out = tf.nn.relu(conv1_out + bias_conv1)
    else:
        layer1_out = conv1_out + bias_conv1
    return layer1_out


def fully_connected(input_x, ouput_dim, relu_after=False):
    fc1 = tf.get_variable('f1', shape=[int(input_x.get_shape()[-1]), int(ouput_dim)])
    bias1 = tf.get_variable('bias1', shape=[int(ouput_dim)], initializer=tf.zeros_initializer())
    fc1_out = tf.matmul(input_x, fc1) + bias1
    if relu_after:
        fc1_out = tf.nn.relu(fc1_out)
    return fc1_out


def batch_norm(inputs, is_training, decay=0.9, epsilon=1e-5):
    """
    Batch normalization on convolutional maps.
    Args:
        inputs:      2D Tensor, 
        decay:       1 - momentum
        scope:       string, variable scope
    Return:
        normed:      batch-normalized maps
    """
    # inputs = ops.convert_to_tensor(inputs)
    inputs_shape = inputs.get_shape()
    n_out = inputs_shape[-1]
    beta = tf.Variable(tf.constant(0.0, shape=[n_out]),
                       name='beta', trainable=True)
    gamma = tf.Variable(tf.constant(1.0, shape=[n_out]),
                        name='gamma', trainable=True)
    batch_mean, batch_var = tf.nn.moments(inputs, [0], name='moments')
    ema = tf.train.ExponentialMovingAverage(decay=decay)

    def mean_var_with_update():
        ema_apply_op = ema.apply([batch_mean, batch_var])
        with tf.control_dependencies([ema_apply_op]):
            return tf.identity(batch_mean), tf.identity(batch_var)

    mean, var = tf.cond(is_training,
                        mean_var_with_update,
                        lambda: (ema.average(batch_mean), ema.average(batch_var)))
    normed = tf.nn.batch_normalization(inputs, mean, var, beta, gamma, epsilon)
    return normed

def batch_norm_conv(inputs, is_training, decay=0.9, epsilon=1e-5):
    """
    Batch normalization on convolutional maps.
    Args:
        inputs:      4D BHWD input maps
        decay:       1 - momentum
        scope:       string, variable scope
    Return:
        normed:      batch-normalized maps
    """
    # inputs = ops.convert_to_tensor(inputs)
    inputs_shape = inputs.get_shape()
    n_out = inputs_shape[-1]
    beta = tf.get_variable(shape=[n_out], initializer=tf.zeros_initializer(),
                       name='beta', trainable=True)
    gamma = tf.get_variable(shape=[n_out], initializer=tf.zeros_initializer(),
                        name='gamma', trainable=True)
    batch_mean, batch_var = tf.nn.moments(inputs, [0, 1, 2], name='moments')
    ema = tf.train.ExponentialMovingAverage(decay=decay)

    def mean_var_with_update():
        ema_apply_op = ema.apply([batch_mean, batch_var])
        with tf.control_dependencies([ema_apply_op]):
            return tf.identity(batch_mean), tf.identity(batch_var)

    mean, var = tf.cond(is_training,
                        mean_var_with_update,
                        lambda: (ema.average(batch_mean), ema.average(batch_var)))
    normed = tf.nn.batch_normalization(inputs, mean, var, beta, gamma, epsilon)
    return normed

# layer where the variables are created inside
# |-------------------------------------|
# bn -> relu -> conv -> bn -> relu -> deconv
# (size remains same)
def residual_layer_conv_identity(input_x, is_training, reuse_var, filter_h_w=[3, 3]):
    depth = int(input_x.get_shape()[3])
    # first bn and relu
    with tf.variable_scope('bn1', initializer=tf.contrib.layers.xavier_initializer(), reuse=reuse_var):
        input_x_bn1 = batch_norm_conv(input_x, is_training)
    input_x_relu1 = tf.nn.relu(input_x_bn1)
    # first convolution
    weight_conv1 = tf.get_variable('conv1', shape=[filter_h_w[0], filter_h_w[1], depth, depth])
    conv1_out = tf.nn.conv2d(input_x_relu1, weight_conv1, strides=[1, 1, 1, 1], padding='SAME')
    bias_conv1 = tf.get_variable('bias1', shape=[depth], initializer=tf.zeros_initializer())
    input_x_conv1 = conv1_out + bias_conv1
    # second bn and relu
    with tf.variable_scope('bn2', initializer=tf.contrib.layers.xavier_initializer(), reuse=reuse_var):
        input_x_bn2 = batch_norm_conv(input_x_conv1, is_training)
    input_x_relu2 = tf.nn.relu(input_x_bn2)
    # second convolution
    weight_conv2 = tf.get_variable('conv2', shape=[filter_h_w[0], filter_h_w[1], depth, depth])
    conv2_out = tf.nn.conv2d(input_x_relu2, weight_conv2, strides=[1, 1, 1, 1], padding='SAME')
    bias_conv2 = tf.get_variable('bias2', shape=[depth], initializer=tf.zeros_initializer())
    input_x_conv2 = conv2_out + bias_conv2
    # residual connection, followed by activation
    output_x = input_x + input_x_conv2
    return output_x

# layer where the variables are created inside
# residual layer along with projection to double the number of channels
def residual_layer_conv_projection_identity(input_x, is_training, reuse_var, filter_h_w=[3, 3]):
    depth = int(input_x.get_shape()[3])
    # first bn and relu
    with tf.variable_scope('bn1', initializer=tf.contrib.layers.xavier_initializer(), reuse=reuse_var):
        input_x_bn1 = batch_norm_conv(input_x, is_training)
    input_x_relu1 = tf.nn.relu(input_x_bn1)
    # first convolution
    weight_conv1 = tf.get_variable('conv1', shape=[filter_h_w[0], filter_h_w[1], depth, depth*2])
    conv1_out = tf.nn.conv2d(input_x_relu1, weight_conv1, strides=[1, 2, 2, 1], padding='SAME')
    bias_conv1 = tf.get_variable('bias1', shape=[depth*2], initializer=tf.zeros_initializer())
    input_x_conv1 = conv1_out + bias_conv1
    # second bn and relu
    with tf.variable_scope('bn2', initializer=tf.contrib.layers.xavier_initializer(), reuse=reuse_var):
        input_x_bn2 = batch_norm_conv(input_x_conv1, is_training)
    input_x_relu2 = tf.nn.relu(input_x_bn2)
    # second convolution
    weight_conv2 = tf.get_variable('conv2', shape=[filter_h_w[0], filter_h_w[1], depth*2, depth*2])
    conv2_out = tf.nn.conv2d(input_x_relu2, weight_conv2, strides=[1, 1, 1, 1], padding='SAME')
    bias_conv2 = tf.get_variable('bias2', shape=[depth*2], initializer=tf.zeros_initializer())
    input_x_conv2 = conv2_out + bias_conv2
    # residual connection,
    weight_conv_projection = tf.get_variable('conv3', shape=[1, 1, depth, depth*2])
    input_x_res = tf.nn.conv2d(input_x, weight_conv_projection, strides=[1, 2, 2, 1], padding='SAME')
    # output
    output_x = input_x_res + input_x_conv2
    return output_x

# layer where the variables are created inside
# |---------------------------------------|
# bn -> relu -> deconv -> bn -> relu -> deconv 
# (size remains same)

def residual_layer_deconv_identity(input_x, is_training, reuse_var, filter_h_w=[3, 3]):
    batch = int(input_x.get_shape()[0])
    height = int(input_x.get_shape()[1])
    width = int(input_x.get_shape()[2])
    depth = int(input_x.get_shape()[3])
    # first bn and relu
    with tf.variable_scope('bn1', initializer=tf.contrib.layers.xavier_initializer(), reuse=reuse_var):
        input_x_bn1 = batch_norm_conv(input_x, is_training)
    input_x_relu1 = tf.nn.relu(input_x_bn1)
    # first convolution
    weight_deconv1 = tf.get_variable('deconv1', shape=[filter_h_w[0], filter_h_w[1], depth, depth])
    bias_deconv1 = tf.get_variable('bias1', shape=[depth], initializer=tf.zeros_initializer())
    input_x_deconv1 =  bias_deconv1 + tf.nn.conv2d_transpose(input_x_relu1, weight_deconv1, [batch, height, width, depth], 
                                                            strides=[1, 1, 1, 1], padding='SAME')
    # second bn and relu
    with tf.variable_scope('bn2', initializer=tf.contrib.layers.xavier_initializer(), reuse=reuse_var):
        input_x_bn2 = batch_norm_conv(input_x_deconv1, is_training)
    input_x_relu2 = tf.nn.relu(input_x_bn2)
    # second convolution
    weight_deconv2 = tf.get_variable('deconv2', shape=[filter_h_w[0], filter_h_w[1], depth, depth])
    bias_deconv2 = tf.get_variable('bias2', shape=[depth], initializer=tf.zeros_initializer())
    input_x_deconv2 =  bias_deconv2 + tf.nn.conv2d_transpose(input_x_relu2, weight_deconv2, [batch, height, width, depth], 
                                                            strides=[1, 1, 1, 1], padding='SAME')
    output_x = input_x + input_x_deconv2
    return output_x

# layer where the variables are created inside
# |---------------------------------------|
# bn -> relu -> deconv -> bn -> relu -> deconv 
# (size remains same)
def residual_layer_deconv_projection_identity(input_x, is_training, reuse_var, filter_h_w=[3, 3], output_shape=None):
    batch = int(input_x.get_shape()[0])
    height = int(input_x.get_shape()[1])
    width = int(input_x.get_shape()[2])
    depth = int(input_x.get_shape()[3])
    if output_shape == None:
        output_shape = [batch, 2*height, 2*width, int(depth/2)]
    # first bn and relu
    with tf.variable_scope('bn1', initializer=tf.contrib.layers.xavier_initializer(), reuse=reuse_var):
        input_x_bn1 = batch_norm_conv(input_x, is_training)
    input_x_relu1 = tf.nn.relu(input_x_bn1)
    # first convolution
    weight_deconv1 = tf.get_variable('deconv1', shape=[filter_h_w[0], filter_h_w[1], int(depth/2), depth])
    bias_deconv1 = tf.get_variable('bias1', shape=[int(depth/2)], initializer=tf.zeros_initializer())
    input_x_deconv1 =  bias_deconv1 + tf.nn.conv2d_transpose(input_x_relu1, weight_deconv1, output_shape, strides=[1, 1, 1, 1], 
                                                            padding='SAME')
    # second bn and relu
    with tf.variable_scope('bn2', initializer=tf.contrib.layers.xavier_initializer(), reuse=reuse_var):
        input_x_bn2 = batch_norm_conv(input_x_deconv1, is_training)
    input_x_relu2 = tf.nn.relu(input_x_bn2)
    # second convolution
    weight_deconv2 = tf.get_variable('deconv2', shape=[filter_h_w[0], filter_h_w[1], int(depth/2), int(depth/2)])
    bias_deconv2 = tf.get_variable('bias2', shape=[int(depth/2)], initializer=tf.zeros_initializer())
    input_x_deconv2 =  bias_deconv2 + tf.nn.conv2d_transpose(input_x_relu2, weight_deconv2, output_shape, strides=[1, 1, 1, 1], 
                                                            padding='SAME')
    # residual connection
    weight_res1 = tf.get_variable('res1', shape=[1, 1, int(depth/2), depth])
    input_x_res = tf.nn.conv2d_transpose(input_x, weight_res1, output_shape, strides=[1, 2, 2, 1], padding='SAME')
    output_x = input_x_res + input_x_deconv2
    return output_x


