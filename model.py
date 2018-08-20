import tensorflow as tf
import pdb
import numpy as np
import os
import utils.layers as layers
from utils.attention_gru_cell import AttentionGRUCell
import _pickle as cPickle
import yaml
import sys
from math import ceil

config = yaml.load(open(sys.argv[1]))

# importing the data loader depending on the dataset
if config['dataset'] == 'ShapeIntersection':
    from utils.shape_intersection_loader import Loader
elif config['dataset'] == 'FloorPlanQA':
    from utils.floor_plan_qa_loader import input_pipeline, preprocess
else:
    raise Exception("Dataset name incorrect!")

# initializing variables
num_epochs = config['num_epochs']
early_stopping_epoch = config['early_stopping_epoch']
l2_reg = config['l2_reg']
dropout_reg = config['dropout_reg']
batch_size = config['batch_size']
learning_rate = config['learning_rate']
embedding_size = config['embedding_size']
run = config['run']
num_hops = config['num_hops']
abl_number = config['abl_number']
save_path = config['save_path']
reg = config['reg']
per_inter_sup = config['per_inter_sup']
pretrained = bool(config['pretrained'])

filenames_tr = [config['data_path'] + '/data.tr']
filenames_va = [config['data_path'] + '/data.va']
filenames_te = [config['data_path'] + '/data.te']

# constant variables
lstm_size = embedding_size
num_conv_layer = 1
num_tr_samples = 12800
num_tr_batches = num_tr_samples // batch_size
num_va_samples = 12800
num_va_batches = num_va_samples // batch_size
num_te_samples = 12800
num_te_batches = num_te_samples // batch_size
if num_tr_samples % batch_size != 0:
    print("Train error is not correct")
if num_va_samples % batch_size != 0:
    print("Validation error is not correct")
if num_te_samples % batch_size != 0:
    print("Test error is not correct")
train_runs = (num_tr_samples // batch_size) * num_epochs

# initializing variables that depend on the model
samples_inter_sup = np.ones([num_tr_batches, batch_size])
if config['model'] == "DMN":
    is_dmn = True
elif config['model'] == "DSMN":
    is_dmn = False
    if reg != 1 and per_inter_sup != 1:
        per_data = config['data_path'] + '/data_' + str(per_inter_sup)
        with open(per_data, 'rb') as fp:
            samples_inter_sup = cPickle.load(fp)
        samples_inter_sup = np.reshape(samples_inter_sup, [-1, batch_size])
else:
    raise Exception("Model name incorrect")

# initializing variable depending on the dataset
if config['dataset'] == 'ShapeIntersection':
    img_size = 32
    max_num_sen = 12
    max_sen_len = 5
    max_ques_len = 1
    num_options = 1
elif config['dataset'] == 'FloorPlanQA':
    img_size = 36
    max_num_sen = 11
    max_sen_len = 19
    max_ques_len = 6
    num_options = 4
    dictionary_lookup = config['data_path'] + '/lookup.pkl'
    with open(dictionary_lookup, 'rb') as dictionary_file:
        word_to_int = cPickle.load(dictionary_file)
        vocabulary_size = len(word_to_int)
else:
    raise Exception("Dataset name incorrect!")

small_img_size = ceil(img_size / 8)


def calc_acc_FloorPlanQA(sess, num_batches, batch_size, type):
    """ To calculate the average accuracy by going through the entire data 
        type: possible values 'tr' (training), 'va' (validation) and 'te' (test)
    """

    acc_cmp = 0
    img_cmp = 0
    for i in range(num_batches):
        if type == 'tr':
            data = sess.run(tr_batch, feed_dict=None)
        elif type == 'va':
            data = sess.run(va_batch, feed_dict=None)
        elif type == 'te':
            data = sess.run(te_batch, feed_dict=None)

        temp = np.ones(batch_size)
        X_d, X_length_d, Y_d, num_sen_d, mask_d, \
        Ques_d, Ques_length_d, img_d = preprocess(data, word_to_int,
                                                  batch_size, max_num_sen,
                                                  max_sen_len)

        acc_tmp, img_tmp = sess.run([accuracy_acc, accuracy_img],
                                    feed_dict={X: X_d, X_length: X_length_d, Y: Y_d,
                                               num_sen: num_sen_d, mask: mask_d,
                                               Ques: Ques_d, Ques_length: Ques_length_d,
                                               img: img_d, is_training: False,
                                               dropout_keep: 1, img_loss_mask: temp})

        acc_cmp += acc_tmp
        img_cmp += img_tmp

    return (acc_cmp / num_batches), (img_cmp / num_batches)


def calc_acc_ShapeIntersection(sess, num_batches, loader):
    """ To calculate the average accuracy by going through the entire data"""

    loss_cmp = 0
    loss_img_cmp = 0
    correct = 0
    for i in range(num_batches):
        temp = np.ones(128)
        X_d, X_length_d, Y_d, num_sen_d, mask_d, \
        Ques_d, Ques_length_d, img_d, loss_mask_d = loader.next_batch()

        loss_tmp, Y_out_temp, loss_img_temp = sess.run([loss_acc, Y_out, loss_img],
                                                       feed_dict={X: X_d, X_length: X_length_d,
                                                                  Y: Y_d, num_sen: num_sen_d,
                                                                  mask: mask_d, Ques: Ques_d,
                                                                  Ques_length: Ques_length_d,
                                                                  img: img_d, loss_mask: loss_mask_d,
                                                                  is_training: False, dropout_keep: 1,
                                                                  img_loss_mask: temp})
        loss_cmp += loss_tmp
        loss_img_cmp += loss_img_temp

    return loss_cmp / num_batches, loss_img_cmp / num_batches


def evaluate_FloorPlanQA(sess, train_writer, test_writer, validation_writer, validation_f_writer, chkpoint_dir):
    """To train, test and validate the model on the FloorPlanQA dataset"""

    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)

    best_va_acc = 0
    best_va_img = 0
    best_va_acc_iter = 0
    if pretrained == False:
        for i in range(train_runs):
            tr_data = sess.run(tr_batch, feed_dict=None)
            X_tr, X_length_tr, Y_tr, num_sen_tr, mask_tr,
            Ques_tr, Ques_length_tr, img_tr = preprocess(tr_data, word_to_int,
                                                         batch_size, max_num_sen,
                                                         max_sen_len)
            img_loss_mask_tr = samples_inter_sup[i % num_tr_batches, 0:batch_size]
            train_dict = {X: X_tr, X_length: X_length_tr, Y: Y_tr, num_sen: num_sen_tr,
                          mask: mask_tr, Ques: Ques_tr, Ques_length: Ques_length_tr,
                          img: img_tr, is_training: True, dropout_keep: dropout_reg,
                          img_loss_mask: img_loss_mask_tr}

            summary, _ = sess.run([merged_tr, train_step], feed_dict=train_dict)
            train_writer.add_summary(summary, i)

            # Calculating validation accuracy after each epoch
            if i % num_tr_batches == 0:
                va_acc, va_img = calc_acc_FloorPlanQA(sess, num_va_batches, batch_size, 'va')
                summary = sess.run(merged_va, feed_dict={complete_accuracy_acc: va_acc,
                                                         complete_accuracy_img: va_img})
                validation_writer.add_summary(summary, i)
                print("Validation Accuracy after " + str(i) + ' steps: ' + str(va_acc))

                # Saving the best validation accuracy model
                if va_acc > best_va_acc:
                    best_va_acc_iter = i
                    best_va_acc = va_acc
                    best_va_img = va_img
                    print("===> Saving the model with best validation accuracy")
                    saver.save(sess, chkpoint_dir + '/model')

                # Early stopping
                if (i - best_va_acc_iter) // num_tr_batches > early_stopping_epoch:
                    print("===> Early stopping")
                    break

        print("===> Best Validation Accuracy:" + str(best_va_acc))
        print("===> Restoring weights for best validation accuracy")
        saver.restore(sess, chkpoint_dir + '/model')
        summary = sess.run(merged_va_f,
                           feed_dict={complete_accuracy_acc: best_va_acc, complete_accuracy_img: best_va_img})
        validation_f_writer.add_summary(summary, best_va_acc_iter)

        # Calculating test accuracy
        te_acc, te_img = calc_acc_FloorPlanQA(sess, num_te_batches, batch_size, 'te')
        summary = sess.run(merged_te, feed_dict={complete_accuracy_acc: te_acc, complete_accuracy_img: te_img})
        test_writer.add_summary(summary, best_va_acc_iter)
        print("Test Accuracy after " + str(best_va_acc_iter) + ' steps: ' + str(te_acc))

    else:
        print("===> Restoring weights for the pretrained model")
        saver.restore(sess, chkpoint_dir + '/model')
        te_acc, te_img = calc_acc_FloorPlanQA(sess, num_te_batches, batch_size, 'te')
        va_acc, va_img = calc_acc_FloorPlanQA(sess, num_va_batches, batch_size, 'va')
        print("Test Accuracy for pretrained model: " + str(te_acc))
        print("Validation Accuracy for pretrained model: " + str(va_acc))

    coord.request_stop()
    coord.join(threads)


def evaluate_ShapeIntersection(sess, train_writer, test_writer, validation_writer, validation_f_writer, chkpoint_dir):
    """To train, test and validate the model on the ShapeIntersection dataset"""

    mask_range = 15
    if per_inter_sup == 1:
        shuffle_flag = True
    else:
        shuffle_flag = False
    tr_data = Loader(filenames_tr[0], batch_size, mask_range, img_size, 1,
                     embedding_size=embedding_size, shuffle=shuffle_flag)
    va_data = Loader(filenames_va[0], batch_size, mask_range, img_size, 1,
                     embedding_size=embedding_size, shuffle=shuffle_flag)
    te_data = Loader(filenames_te[0], batch_size, mask_range, img_size, 1,
                     embedding_size=embedding_size, shuffle=shuffle_flag)

    best_va_loss = float('inf')
    best_va_iter = -1
    best_va_img_loss = float('inf')
    if pretrained == False:
        for i in range(train_runs):

            X_tr, X_length_tr, Y_tr, num_sen_tr, mask_tr, \
            Ques_tr, Ques_length_tr, img_tr, loss_mask_tr = tr_data.next_batch()
            img_loss_mask_tr = samples_inter_sup[i % num_tr_batches, 0:batch_size]
            train_dict = {X: X_tr, X_length: X_length_tr, Y: Y_tr, num_sen: num_sen_tr,
                          mask: mask_tr, Ques: Ques_tr, Ques_length: Ques_length_tr,
                          img: img_tr, loss_mask: loss_mask_tr, is_training: True,
                          dropout_keep: dropout_reg, img_loss_mask: img_loss_mask_tr}

            pred_tr, summary, _, tr_loss_acc, tr_loss_img = sess.run([Y_out, merged_tr, train_step,
                                                                      loss_acc, loss_img],
                                                                     feed_dict=train_dict)
            train_writer.add_summary(summary, i)

            # Calculating validation accuracy after each epoch
            if i % num_tr_batches == 0:
                va_loss, va_img_loss = calc_acc_ShapeIntersection(sess, num_va_batches, va_data)
                summary = sess.run(merged_va, feed_dict={complete_loss: va_loss,
                                                         complete_img_loss: va_img_loss})
                validation_writer.add_summary(summary, i)
                print('Step: {}.'.format(i))
                print('Validation RMSE: {}. Validation img RMSE: {}.'.format(np.sqrt(va_loss), np.sqrt(va_img_loss)))

                if va_loss < best_va_loss:
                    best_va_iter = i
                    best_va_loss = va_loss
                    best_va_img_loss = va_img_loss
                    print("===> Saving the model with best validation loss")
                    saver.save(sess, chkpoint_dir + '/model')

                # Early stopping
                if (i - best_va_iter) // num_tr_batches > early_stopping_epoch:
                    print("Early stopping")
                    break

        print('Best va step: {}. Best validation RMSE: {:.4f}.'.format(best_va_iter, np.sqrt(best_va_loss)))
        print("===> Restoring weights for best validation loss")
        saver.restore(sess, chkpoint_dir + '/model')
        summary = sess.run(merged_va_f, feed_dict={complete_loss: best_va_loss, complete_img_loss: best_va_img_loss})
        validation_f_writer.add_summary(summary, best_va_iter)

        te_loss, te_img_loss = calc_acc_ShapeIntersection(sess, num_te_batches, te_data)
        summary = sess.run(merged_te, feed_dict={complete_loss: te_loss, complete_img_loss: te_img_loss})
        test_writer.add_summary(summary, best_va_iter)
        print('Test RMSE: {:.4f} Test img RMSE: {:.2f}.'.format(np.sqrt(te_loss), np.sqrt(te_img_loss)))

    else:
        print('Loading the pretrained model')
        saver.restore(sess, chkpoint_dir + '/model')
        va_loss, va_img_loss = calc_acc_ShapeIntersection(sess, num_va_batches, va_data)
        print('Validation RMSE: {}.'.format(np.sqrt(va_loss)))
        te_loss, te_img_loss = calc_acc_ShapeIntersection(sess, num_te_batches, te_data)
        print('Test RMSE: {:.4f}.'.format(np.sqrt(te_loss)))

    tr_data.destruct()
    va_data.destruct()
    te_data.destruct()


# source: https://github.com/domluna/memn2n
def _position_encoding(sentence_size, embedding_size):
    """Position encoding described in section 4.1 in "End to End Memory Networks" (http://arxiv.org/pdf/1503.08895v5.pdf)"""

    encoding = np.ones((embedding_size, sentence_size), dtype=np.float32)
    ls = sentence_size + 1
    le = embedding_size + 1
    for i in range(1, le):
        for j in range(1, ls):
            encoding[i - 1, j - 1] = (i - (le - 1) / 2) * (j - (ls - 1) / 2)
    encoding = 1 + 4 * encoding / embedding_size / sentence_size
    return np.transpose(encoding)


def img_encoder(img, num_conv=1, reuse_var=None):
    """ function to extact features from an image
    the number of features returned is small_img_size * small_img_size * num_conv
    shape(features) = [batch_size, small_img_size * small_img_size * num_conv]
    """
    with tf.variable_scope('res_conv_proj1', initializer=tf.contrib.layers.xavier_initializer(), reuse=reuse_var):
        res_conv_proj1 = layers.residual_layer_conv_projection(img)

    with tf.variable_scope('res_conv1', initializer=tf.contrib.layers.xavier_initializer(), reuse=reuse_var):
        res_conv1 = layers.residual_layer_conv(res_conv_proj1)

    with tf.variable_scope('res_conv_proj2', initializer=tf.contrib.layers.xavier_initializer(), reuse=reuse_var):
        res_conv_proj2 = layers.residual_layer_conv_projection(res_conv1)

    with tf.variable_scope('res_conv2', initializer=tf.contrib.layers.xavier_initializer(), reuse=reuse_var):
        res_conv2 = layers.residual_layer_conv(res_conv_proj2)

    with tf.variable_scope('res_conv_proj3', initializer=tf.contrib.layers.xavier_initializer(), reuse=reuse_var):
        res_conv_proj3 = layers.residual_layer_conv_projection(res_conv2)

    with tf.variable_scope('conv1', initializer=tf.contrib.layers.xavier_initializer(), reuse=reuse_var):
        conv1 = layers.convolution_layer(res_conv_proj3, [3, 3, 8, num_conv], [1, 1, 1, 1], padding='SAME',
                                         relu_after=False)

    features = tf.reshape(conv1, [-1, small_img_size * small_img_size * num_conv])
    return features


def img_decoder(img, num_conv=1, reuse_var=None):
    """ to decode the img from the embedding
    img is of size batch_size * small_img_size * small_img_size * 1
    """
    with tf.variable_scope('deconv1', initializer=tf.contrib.layers.xavier_initializer(), reuse=reuse_var):
        deconv1 = layers.deconvolution_layer(img, [3, 3, 8 * num_conv, 1], deconv_strides=[1, 1, 1, 1], padding='SAME',
                                             output_size=[batch_size, small_img_size, small_img_size, 8 * num_conv],
                                             relu_after=False)
        deconv1 = tf.nn.relu(deconv1)

    with tf.variable_scope('res_deconv_proj1', initializer=tf.contrib.layers.xavier_initializer(), reuse=reuse_var):
        res_deconv_proj1 = layers.residual_layer_deconv_projection(deconv1, relu_after=False,
                                                                   output_shape=[batch_size, int(img_size / 4),
                                                                                 int(img_size / 4), 4 * num_conv])
        res_deconv_proj1 = tf.nn.relu(res_deconv_proj1)

    with tf.variable_scope('res_deconv1', initializer=tf.contrib.layers.xavier_initializer(), reuse=reuse_var):
        res_deconv1 = layers.residual_layer_deconv(res_deconv_proj1, relu_after=False)
        res_deconv1 = tf.nn.relu(res_deconv1)

    with tf.variable_scope('res_deconv_proj2', initializer=tf.contrib.layers.xavier_initializer(), reuse=reuse_var):
        res_deconv_proj2 = layers.residual_layer_deconv_projection(res_deconv1, relu_after=False)
        res_deconv_proj2 = tf.nn.relu(res_deconv_proj2)

    with tf.variable_scope('res_deconv2', initializer=tf.contrib.layers.xavier_initializer(), reuse=reuse_var):
        res_deconv2 = layers.residual_layer_deconv(res_deconv_proj2, relu_after=False)
        res_deconv2 = tf.nn.relu(res_deconv2)

    with tf.variable_scope('res_deconv_proj3', initializer=tf.contrib.layers.xavier_initializer(), reuse=reuse_var):
        res_deconv_proj3 = layers.residual_layer_deconv_projection(res_deconv2, relu_after=False)

    return res_deconv_proj3


if __name__ == '__main__':

    if config['dataset'] == 'FloorPlanQA':
        tr_batch = input_pipeline(filenames_tr, batch_size, num_options)
        va_batch = input_pipeline(filenames_va, batch_size, num_options)
        te_batch = input_pipeline(filenames_te, batch_size, num_options)
        word_embedding = tf.Variable(tf.random_uniform([vocabulary_size + 1, embedding_size],
                                                       -1 * np.sqrt(3),
                                                       np.sqrt(3)),
                                     trainable=True,
                                     collections=None,
                                     validate_shape=True)

    # Input placeholders
    if config['dataset'] == 'FloorPlanQA':
        X = tf.placeholder(tf.int64, shape=[batch_size, max_num_sen, max_sen_len])
        Y = tf.placeholder(tf.float32, shape=[batch_size, num_options, 1])
        Ques = tf.placeholder(tf.int64, shape=[batch_size, None])
        Ques_emb = tf.nn.embedding_lookup(word_embedding, Ques)
    else:
        X = tf.placeholder(tf.float32, shape=[batch_size, max_num_sen, max_sen_len])
        Y = tf.placeholder(tf.float32, shape=[batch_size, ])
        Ques = tf.placeholder(tf.float32, shape=[batch_size, None, embedding_size])
        Ques_emb = Ques

    X_length = tf.placeholder(tf.int64, shape=[batch_size, max_num_sen])
    num_sen = tf.placeholder(tf.int64, shape=[batch_size, ])
    mask = tf.placeholder(tf.float32, shape=[batch_size, max_num_sen])
    img = tf.placeholder(tf.float32, shape=[batch_size, img_size, img_size, max_num_sen])
    img_loss_mask = tf.placeholder(tf.float32, shape=[batch_size])
    Ques_length = tf.placeholder(tf.int64, shape=[batch_size])
    is_training = tf.placeholder(tf.bool)
    dropout_keep = tf.placeholder(tf.float32)

    if config['dataset'] == 'ShapeIntersection':
        loss_mask = tf.placeholder(tf.float32, shape=[batch_size, img_size, img_size, max_num_sen])

    if config['dataset'] == 'FloorPlanQA':
        X_emb = tf.nn.embedding_lookup(word_embedding, X)
        encoding = _position_encoding(max_sen_len, embedding_size)
        X_encoded = tf.reduce_sum(X_emb * encoding, 2)
    else:
        X_sen_emb_1 = tf.reshape(X, [batch_size * max_num_sen, max_sen_len])
        X_sen_emb_2 = tf.contrib.layers.fully_connected(X_sen_emb_1, embedding_size)
        X_sen_emb_3 = tf.contrib.layers.fully_connected(X_sen_emb_2, embedding_size, activation_fn=None)
        X_encoded = tf.reshape(X_sen_emb_3, [batch_size, max_num_sen, embedding_size])

    with tf.variable_scope('X_sen', initializer=tf.contrib.layers.xavier_initializer()):
        outputs, _ = tf.nn.bidirectional_dynamic_rnn(tf.contrib.rnn.GRUCell(embedding_size),
                                                     tf.contrib.rnn.GRUCell(embedding_size),
                                                     X_encoded,
                                                     dtype=np.float32,
                                                     sequence_length=num_sen)

    # f<-> = f-> + f<-
    X_des_emb = tf.reduce_sum(tf.stack(outputs), axis=0)

    # apply dropout
    X_des_emb = tf.nn.dropout(X_des_emb, keep_prob=dropout_keep)
    state_X_sen = tf.reshape(X_des_emb, [batch_size * max_num_sen, embedding_size])

    initial_val = tf.Variable(tf.zeros([batch_size, img_size, img_size, max_num_sen]), trainable=False)
    input_curr = initial_val

    if is_dmn == False:
        for i in range(max_num_sen):
            reuse_var = bool(i)

            # attention module to select the correct channels for encoding
            with tf.variable_scope('channel_attention', initializer=tf.contrib.layers.xavier_initializer(),
                                   reuse=bool(i - 1)):
                # last channel is the all zero channel
                input_curr_relevant = tf.slice(input_curr, begin=[0, 0, 0, 0],
                                               size=[batch_size, img_size, img_size, i + 1])
                if i == 0:
                    relevant_image = input_curr_relevant
                else:
                    X_des_emb_relevant = tf.slice(X_des_emb, begin=[0, 0, 0], size=[batch_size, i, embedding_size])
                    X_des_emb_extra = tf.Variable(tf.zeros([batch_size, 1, embedding_size]), trainable=False)
                    X_des_emb_relevant_extra = tf.concat([X_des_emb_relevant, X_des_emb_extra], axis=1)
                    X_des_emb_curr = tf.slice(X_des_emb, begin=[0, i, 0], size=[batch_size, 1, embedding_size])
                    features = [X_des_emb_relevant_extra * X_des_emb_curr,
                                tf.abs(X_des_emb_relevant_extra - X_des_emb_curr)]
                    feature_vec = tf.concat(features, 2)
                    feature_vec_reshaped = tf.reshape(feature_vec, [batch_size * (i + 1), -1])
                    attention_values = tf.contrib.layers.fully_connected(feature_vec_reshaped,
                                                                         embedding_size,
                                                                         activation_fn=tf.nn.tanh,
                                                                         reuse=bool(i - 1),
                                                                         scope="fc1")

                    attention_values = tf.contrib.layers.fully_connected(attention_values,
                                                                         1,
                                                                         activation_fn=None,
                                                                         reuse=bool(i - 1), scope="fc2")

                    attention_values_reshaped = tf.reshape(attention_values, [batch_size, (i + 1)])
                    attention_values_softmax = tf.nn.softmax(attention_values_reshaped)
                    attention_values_softmax_reshaped = tf.expand_dims(tf.expand_dims(attention_values_softmax, 1), 1)
                    relevant_image = tf.reduce_sum(input_curr_relevant * attention_values_softmax_reshaped, axis=3,
                                                   keep_dims=True)

            with tf.variable_scope('img_encoder', initializer=tf.contrib.layers.xavier_initializer(), reuse=reuse_var):
                conv_rep = img_encoder(relevant_image, num_conv_layer, reuse_var=reuse_var)
            text_rep = tf.squeeze(tf.slice(X_des_emb, begin=[0, i, 0], size=[batch_size, 1, embedding_size]), axis=1)
            features = tf.concat([conv_rep, text_rep], axis=1)

            with tf.variable_scope('img_small', initializer=tf.contrib.layers.xavier_initializer(), reuse=reuse_var):
                img_small = layers.fully_connected(features, small_img_size * small_img_size, relu_after=False)
                img_small = tf.reshape(img_small, [batch_size, small_img_size, small_img_size, 1])

            with tf.variable_scope('img_decoder', initializer=tf.contrib.layers.xavier_initializer(), reuse=reuse_var):
                output_temp = img_decoder(img_small, num_conv_layer, reuse_var=reuse_var)

            mask_curr = tf.slice(mask, begin=[0, i], size=[batch_size, 1])
            mask_curr = tf.expand_dims(tf.expand_dims(mask_curr, axis=2), axis=3)
            output_temp = (output_temp * mask_curr)
            padding = tf.convert_to_tensor([[0, 0], [0, 0], [0, 0], [i, max_num_sen - 1 - i]])
            output_temp_expanded = tf.pad(output_temp, paddings=padding)
            output_curr = output_temp_expanded + input_curr
            input_curr = output_curr

        img_new = output_curr

        if config['dataset'] == 'FloorPlanQA':
            with tf.variable_scope('loss_image'):
                loss_img = tf.nn.sigmoid_cross_entropy_with_logits(labels=img, logits=img_new)
                mask_modi = tf.expand_dims(tf.expand_dims(mask, 1), 1) * tf.expand_dims(
                    tf.expand_dims(tf.expand_dims(img_loss_mask, 1), 1), 1)
                loss_img = tf.cond(tf.equal((tf.reduce_sum(mask_modi) * img_size * img_size), 0),
                                   lambda: tf.zeros([]),
                                   lambda: tf.reduce_sum(loss_img * mask_modi) / (
                                               tf.reduce_sum(mask_modi) * img_size * img_size))
                ones_img = tf.ones(tf.shape(img))
                zeros_img = tf.zeros(tf.shape(img))
                predicted_img = tf.where(img_new < 0, x=zeros_img, y=ones_img)
                correct_prediction = tf.cast(tf.equal(predicted_img, img), tf.float32) * mask_modi
                accuracy_img = tf.cond(tf.equal((tf.reduce_sum(mask_modi) * img_size * img_size), 0),
                                       lambda: tf.zeros([]),
                                       lambda: tf.reduce_sum(correct_prediction) / (
                                                   tf.reduce_sum(mask_modi) * img_size * img_size))

            # adding batch normalization
            # For FloorPlanQA we observe that DSMN* performs better without this
            if reg == 1:
                img_new = tf.reshape(img_new, shape=[batch_size * img_size * img_size * max_num_sen, -1])
                img_new = layers.batch_norm(img_new, is_training)
                img_new = tf.reshape(img_new, shape=[batch_size, img_size, img_size, max_num_sen])

            # the special softmax layer
            img_new_sig = tf.sigmoid(img_new)
            mask_modi = tf.expand_dims(tf.expand_dims(mask, 1), 1)
            img_new_sig_rel = img_new_sig * mask_modi
        else:
            with tf.variable_scope('loss_image'):
                img_created_reshape = tf.reshape(img_new, [-1, 12])
                img_reshape = tf.reshape(img, [-1, 12])
                loss_mask_temp = loss_mask * tf.expand_dims(tf.expand_dims(tf.expand_dims(img_loss_mask, 1), 1), 1)
                loss_mask_reshape = tf.reshape(loss_mask_temp, [-1, 12])
                loss_img = 1e2 / per_inter_sup * tf.reduce_mean(
                    tf.multiply(tf.pow(img_created_reshape - img_reshape, 2),
                                loss_mask_reshape))
            img_new_sig_rel = img_new
            accuracy_img = tf.Variable(tf.zeros([]), trainable=False)

    else:
        loss_img = tf.Variable(tf.zeros([]), trainable=False)
        accuracy_img = tf.Variable(tf.zeros([]), trainable=False)

    # rnn for question
    with tf.variable_scope('Ques_lstm', initializer=tf.contrib.layers.xavier_initializer()):
        _, state_Ques = tf.nn.dynamic_rnn(tf.contrib.rnn.GRUCell(embedding_size),
                                          Ques_emb,
                                          dtype=np.float32,
                                          sequence_length=Ques_length
                                          )

    # attention module for tagging
    # initializing memory
    if is_dmn == False:
        initial_val = tf.Variable(tf.zeros([batch_size, img_size, img_size, 1]), trainable=False)
        mem = initial_val

    initial_val_tag = tf.Variable(tf.zeros([batch_size, embedding_size]), trainable=False)
    mem_tag = initial_val_tag

    state_Ques_expanded = tf.expand_dims(state_Ques, axis=1)

    for i in range(num_hops):
        reuse_var = bool(i)

        con_fea_text = [X_des_emb * state_Ques_expanded,
                        tf.abs(state_Ques_expanded - X_des_emb),
                        X_des_emb * tf.expand_dims(mem_tag, axis=1),
                        tf.abs(X_des_emb - tf.expand_dims(mem_tag, axis=1))]

        # extracting the visual features for context
        if is_dmn == False:
            con_fea_vis_abs = tf.transpose(tf.abs(mem - img_new_sig_rel), perm=[0, 3, 1, 2])
            con_fea_vis_abs = tf.reshape(con_fea_vis_abs, [batch_size * max_num_sen, img_size, img_size, 1])
            with tf.variable_scope('img_encoder_con_abs', initializer=tf.contrib.layers.xavier_initializer(),
                                   reuse=reuse_var):
                con_fea_vis_abs = img_encoder(con_fea_vis_abs, num_conv_layer, reuse_var=reuse_var)
            con_fea_vis_mul = tf.transpose(mem * img_new_sig_rel, perm=[0, 3, 1, 2])
            con_fea_vis_mul = tf.reshape(con_fea_vis_mul, [batch_size * max_num_sen, img_size, img_size, 1])
            with tf.variable_scope('img_encoder_con_mul', initializer=tf.contrib.layers.xavier_initializer(),
                                   reuse=reuse_var):
                con_fea_vis_mul = img_encoder(con_fea_vis_mul, num_conv_layer, reuse_var=reuse_var)
            con_fea_vis = [tf.reshape(con_fea_vis_abs,
                                      [batch_size, max_num_sen, small_img_size * small_img_size * num_conv_layer]),
                           tf.reshape(con_fea_vis_mul,
                                      [batch_size, max_num_sen, small_img_size * small_img_size * num_conv_layer])]
        else:
            con_fea_vis = []

        # getting the attention values based on the features
        con_fea_vis.extend(con_fea_text)
        con_fea = con_fea_vis
        con_fea = tf.concat(con_fea, 2)
        con_fea = tf.reshape(con_fea, [batch_size * max_num_sen, -1])
        with tf.variable_scope('att_val', initializer=tf.contrib.layers.xavier_initializer(), reuse=reuse_var):
            att_val = tf.contrib.layers.fully_connected(con_fea,
                                                        embedding_size,
                                                        activation_fn=tf.nn.tanh,
                                                        reuse=reuse_var,
                                                        scope="fc1")

            att_val = tf.contrib.layers.fully_connected(att_val,
                                                        1,
                                                        activation_fn=None,
                                                        reuse=reuse_var,
                                                        scope="fc2")

        att_val = tf.reshape(att_val, [batch_size, max_num_sen])

        # the special softmax layer for using attention to get context
        att_val = tf.exp(att_val)
        att_val = att_val * mask
        att_val_smax = att_val / tf.reduce_sum(att_val, axis=1, keep_dims=True)
        gru_inputs = tf.concat([X_des_emb, tf.expand_dims(att_val_smax, 2)], 2)

        with tf.variable_scope('attention_gru', reuse=reuse_var):
            _, con_tag = tf.nn.dynamic_rnn(AttentionGRUCell(embedding_size),
                                           gru_inputs,
                                           dtype=np.float32,
                                           sequence_length=num_sen)

        if is_dmn == False:
            con = tf.reduce_sum(img_new_sig_rel * tf.expand_dims(tf.expand_dims(att_val_smax, 1), 1),
                                3, keep_dims=True)

        mem_tag_fea_text = [mem_tag, state_Ques, con_tag]
        if is_dmn == False:
            with tf.variable_scope('img_encoder_con', initializer=tf.contrib.layers.xavier_initializer(),
                                   reuse=reuse_var):
                mem_tag_fea_vis = [img_encoder(con, num_conv_layer, reuse_var=reuse_var)]
        else:
            mem_tag_fea_vis = []
            # getting the new memory tag
        mem_tag_fea_text.extend(mem_tag_fea_vis)
        mem_tag_fea = mem_tag_fea_text
        mem_tag_fea = tf.concat(mem_tag_fea, 1)
        with tf.variable_scope('mem_tag' + str(i), initializer=tf.contrib.layers.xavier_initializer(), reuse=None):
            mem_tag = layers.fully_connected(mem_tag_fea, embedding_size, relu_after=True)
        if is_dmn == False:
            # getting new memory
            mem_fea_text = [mem_tag]
            with tf.variable_scope('img_encoder' + str(i), initializer=tf.contrib.layers.xavier_initializer(),
                                   reuse=None):
                mem_fea_vis = [img_encoder(mem, num_conv_layer, reuse_var=None)]
            mem_fea_text.extend(mem_fea_vis)
            mem_fea = mem_fea_text
            mem_fea = tf.concat(mem_fea, axis=1)
            with tf.variable_scope('mem_small', initializer=tf.contrib.layers.xavier_initializer(), reuse=reuse_var):
                mem_small = layers.fully_connected(mem_fea, small_img_size * small_img_size, relu_after=False)
            mem_small = tf.reshape(mem_small, [batch_size, small_img_size, small_img_size, 1])
            with tf.variable_scope('img_decoder' + str(i), initializer=tf.contrib.layers.xavier_initializer(),
                                   reuse=None):
                mem = img_decoder(mem_small, num_conv_layer, reuse_var=None)
            # applying batch_normalization and sigmoid to the created memory
            mem = tf.reshape(mem, shape=[batch_size * img_size * img_size * 1, -1])
            with tf.variable_scope('mem_bn' + str(i)):
                mem = layers.batch_norm(mem, is_training)
            mem = tf.reshape(mem, shape=[batch_size, img_size, img_size, 1])
            mem = tf.sigmoid(mem)

    if is_dmn == False:
        final_image = mem
        with tf.variable_scope('img_encoder_final', initializer=tf.contrib.layers.xavier_initializer(), reuse=None):
            conv_rep = img_encoder(final_image, num_conv_layer, reuse_var=None)

    if is_dmn == False:
        if abl_number == 1:
            features = tf.concat([conv_rep, mem_tag, state_Ques], axis=1)
        elif abl_number == 2:
            features = tf.concat([mem_tag, state_Ques], axis=1)
        else:
            features = tf.concat([conv_rep, state_Ques], axis=1)
    else:
        features = tf.concat([mem_tag, state_Ques], axis=1)

    # apply dropout
    features = tf.nn.dropout(features, keep_prob=dropout_keep)

    Y_squeezed = tf.squeeze(Y)
    with tf.variable_scope('Classification'):
        with tf.variable_scope('fc1', initializer=tf.contrib.layers.xavier_initializer()):
            fc1_out = layers.fully_connected(input_x=features, ouput_dim=num_options, relu_after=False)

        if config['dataset'] == 'FloorPlanQA':
            # loss caculation
            score = fc1_out
            loss_acc = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=Y_squeezed, logits=score))
            # accuracy calculation
            Y_class = tf.argmax(Y_squeezed, axis=1)
            Y_pred_class = tf.argmax(score, axis=1)
            correct_prediction = tf.equal(Y_class, Y_pred_class)
            accuracy_acc = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        else:
            # loss calculation
            Y_out = tf.reshape(fc1_out, (batch_size,))
            loss_acc = tf.reduce_mean(tf.pow(Y_out - Y, 2))
            accuracy_acc = tf.Variable(tf.zeros([]), trainable=False)

    loss_reg = 0
    # add l2 regularization for all variables except biases
    for v in tf.trainable_variables():
        if not 'bias' in v.name.lower():
            loss_reg += tf.nn.l2_loss(v)

    if is_dmn == False:
        loss = (reg * loss_acc) + ((1 - reg) * loss_img) + (l2_reg * loss_reg)
    else:
        loss = loss_acc + (l2_reg * loss_reg)

    # tensorboard summaries
    tf.summary.scalar('loss', loss, collections=['train'])
    tf.summary.scalar('loss_acc', loss_acc, collections=['train'])
    tf.summary.scalar('loss_reg', loss_reg, collections=['train'])
    tf.summary.scalar('accuracy_acc', accuracy_acc, collections=['train'])

    if is_dmn == False:
        tf.summary.scalar('loss_img', loss_img, collections=['train'])
        tf.summary.scalar('accuracy_img', accuracy_img, collections=['train'])

    train_step = tf.train.AdamOptimizer(learning_rate).minimize(loss)

    # some additional summaries
    if config['dataset'] == 'FloorPlanQA':
        complete_accuracy_acc = tf.placeholder(tf.float32, shape=[])
        complete_accuracy_img = tf.placeholder(tf.float32, shape=[])
        tf.summary.scalar("complete_accuracy_acc", complete_accuracy_acc,
                          collections=['validation', 'test', 'validation_f'])
        tf.summary.scalar("complete_accuracy_img", complete_accuracy_img,
                          collections=['validation', 'test', 'validation_f', ])
    else:
        complete_loss = tf.placeholder(tf.float32, shape=[])
        complete_img_loss = tf.placeholder(tf.float32, shape=[])
        tf.summary.scalar("complete_loss", complete_loss, collections=['validation', 'test', 'validation_f'])
        tf.summary.scalar("complete_img_loss", complete_img_loss, collections=['validation', 'test', 'validation_f', ])

    # merging all summaries in one node
    merged_tr = tf.summary.merge_all('train')
    merged_va = tf.summary.merge_all('validation')
    merged_te = tf.summary.merge_all('test')
    merged_va_f = tf.summary.merge_all('validation_f')

    saver = tf.train.Saver()
    chkpoint_dir = save_path + '/' + config['model'] + '_' + config['dataset'] + '_' \
                   + str(lstm_size) + '_' + str(embedding_size) + '_' \
                   + str(learning_rate) + '_' + str(batch_size) + '_' \
                   + str(reg) + '_' + str(num_conv_layer) + '_' \
                   + str(run) + '_' + str(num_epochs) + '_' \
                   + str(early_stopping_epoch) + '_' + str(dropout_reg) + '_' \
                   + str(l2_reg) + '_' + str(num_hops) + '_' \
                   + str(abl_number) + '_' + str(per_inter_sup)

    with tf.Session() as sess:
        # making the necessary files 
        if not os.path.exists(chkpoint_dir):
            os.mkdir(chkpoint_dir)
            os.mkdir(chkpoint_dir + '/train')
            os.mkdir(chkpoint_dir + '/test')
            os.mkdir(chkpoint_dir + '/validation')
            os.mkdir(chkpoint_dir + '/validation_f')

        # setting up tensorboard writers
        train_writer = tf.summary.FileWriter(chkpoint_dir + '/train', sess.graph)
        validation_writer = tf.summary.FileWriter(chkpoint_dir + '/validation')
        test_writer = tf.summary.FileWriter(chkpoint_dir + '/test')
        validation_f_writer = tf.summary.FileWriter(chkpoint_dir + '/validation_f')

        # graph initialization
        init_op1 = tf.global_variables_initializer()
        sess.run(init_op1)
        init_opt2 = tf.local_variables_initializer()
        sess.run(init_opt2)

        if config['dataset'] == 'FloorPlanQA':
            evaluate_FloorPlanQA(sess, train_writer, test_writer, validation_writer, validation_f_writer, chkpoint_dir)
        else:
            evaluate_ShapeIntersection(sess, train_writer, test_writer, validation_writer, validation_f_writer,
                                       chkpoint_dir)
