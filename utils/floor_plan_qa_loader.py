import numpy as np
import tensorflow as tf

# returns a numpy array which gives the channel number associated with each sentence
# from generate_question.generate_object_door_channel()
    # suppose maximum r rooms
    # 0 channel for room 0, 1 for room 1, ...
    # r channel for house door
    # r+1 channel for room 1 door, r+2 channel for room 2 door, ...
    # 2*r+1 for object 1 ..
# from house_structure
    # possible shapes for the entity
    # entity_shapes=['cube', 'cuboid', 'sphere', 'cylinder']
def generate_sentence_order(X, X_length, num_sen, word_to_int):
    channel_order = np.zeros(np.shape(X_length))-1
    # i referring to each description in batch 
    for i in range(np.shape(X_length)[0]):
        # j referring to each sentence in a description
        for j in range(num_sen[i]):
            curr_sentence = X[i,j,:]
            if curr_sentence[0]==word_to_int['room'] and curr_sentence[1]==word_to_int['1']:
                channel_order[i,j] = 0
                curr_room = 0
                continue
            elif curr_sentence[0]==word_to_int['room'] and curr_sentence[1]==word_to_int['2']:
                channel_order[i,j] = 1
                curr_room = 1
                continue
            elif curr_sentence[0]==word_to_int['room'] and curr_sentence[1]==word_to_int['3']:
                channel_order[i,j] = 2
                curr_room = 2
                continue
            elif curr_sentence[0]==word_to_int['the'] and curr_sentence[1]==word_to_int['house'] and curr_sentence[2]==word_to_int['door']:
                channel_order[i,j] = 3
                curr_room = -1
                continue
            elif np.sum(curr_sentence==word_to_int['door'])==1:
                if curr_room != -1:
                    channel_order[i,j] = curr_room + 4
                else :
                    print('Error in door assignement')
                curr_room = -1
                continue
            elif np.sum(curr_sentence==word_to_int['cube'])==1:
                channel_order[i,j] = 7
                curr_room = -1
                continue
            elif np.sum(curr_sentence==word_to_int['cuboid'])==1:
                channel_order[i,j] = 8
                curr_room = -1
                continue
            elif np.sum(curr_sentence==word_to_int['sphere'])==1:
                channel_order[i,j] = 9
                curr_room = -1
                continue
            elif np.sum(curr_sentence==word_to_int['cylinder'])==1:
                channel_order[i,j] = 10
                curr_room = -1
                continue
    return channel_order

# the channels of the image are rearranged in the order of sentence
# X is a sequence of words, shape(X)=[batch_size, maximum number of sentences is a description, maximum length of a sentece]
# X_length is length of each sentence in X, shape(length)=[batch_size, maximum number of sentences is a description]
# Y is the solution, shape(Y)=[batch_size,4,1]
# num_sen is the number of vlaid sentences in the description, shape(num_sen) = [batch_size, ]
# mask is [1,1,1 ...,1,0,...,0,0], where the last number of 1's is the number of valid sentences in the description, 
# shape(mask) = [batchsize, max_num_sen]
def preprocess(data, word_to_int, batch_size, max_num_sen, max_sen_len):
    # getting each of the array from the data list
    des_len = data[0]
    ques_len = data[1]
    des_seq = data[6]
    ques_seq = data[7]
    sol_seq = data[12]
    img = data[13]

    # defining the output variables
    # output description, length of each sentence, number of sentence in a description
    stop = np.argwhere(des_seq == word_to_int['.'])
    num_sen = np.zeros((batch_size,))
    for i in range(batch_size):
        num_sen[i] = np.sum(stop[:, 0] == i)
    num_sen = num_sen.astype(int)
    mask = np.zeros([batch_size, max_num_sen])
    for i in range(batch_size):
        mask[i, 0:num_sen[i]] = 1
    X = np.zeros((batch_size, max_num_sen, max_sen_len))
    X_length = np.zeros((batch_size, max_num_sen))
    sen_num = 0
    for i in range(len(stop)):
        # first sentence
        if i == 0 or stop[i, 0] != stop[i - 1, 0]:  
            # removing '.' from the end of the sentence
            X[stop[i, 0], 0, 0:stop[i, 1]] = des_seq[stop[i, 0], 0:stop[i, 1], 0]
            X_length[stop[i, 0], 0] = stop[i, 1]
            sen_num = 1
        # rest of the sentences
        else: 
            # removing '.' from the end of the sentence
            X[stop[i, 0], sen_num, 0:stop[i, 1]-stop[i-1, 1]-1] = des_seq[stop[i, 0],
                                                                    stop[i-1,1]+1:stop[i, 1], 0]
            X_length[stop[i, 0], sen_num] = stop[i, 1] - stop[i-1, 1] - 1
            sen_num += 1
    
    # Question
    Ques = np.squeeze(ques_seq)
    Ques_length = ques_len
    
    # Solution
    Y = sol_seq
    img = np.reshape(img, (batch_size, 36, 36, 11))
    channel_order = generate_sentence_order(X, X_length, num_sen, word_to_int)
    channel_order = channel_order.astype(int)
    for i in range(batch_size):
        img[i] = np.transpose(img[i, :, :, channel_order[i]], [1,2,0])

    return X, X_length, Y, num_sen, mask, Ques, Ques_length, img

def read_my_file_format(filename_queue, num_options):
    reader = tf.TFRecordReader()
    key, record_string = reader.read(filename_queue)
    context_features={
        "des_len":tf.FixedLenFeature([],tf.int64),
        "ques_len":tf.FixedLenFeature([],tf.int64)
        }
    for i in range(num_options):
        context_features['opt_'+str(i)+'_len']=tf.FixedLenFeature([],tf.int64)
    sequence_features={
        "des_seq":tf.FixedLenSequenceFeature([1,],tf.int64),
        "ques_seq":tf.FixedLenSequenceFeature([1,],tf.int64),
        "sol_seq":tf.FixedLenSequenceFeature([1,],tf.int64),
        #Warning: We are only selecting images of size 36 * 36
        "img_seq":tf.FixedLenSequenceFeature([36*11,],tf.int64)
        }
    for i in range(num_options):
        sequence_features['opt_'+str(i)+'_seq']=tf.FixedLenSequenceFeature([1,],tf.int64)
    context_parsed, sequence_parsed = tf.parse_single_sequence_example(
        serialized=record_string, context_features=context_features, sequence_features=sequence_features)
    return context_parsed,sequence_parsed

def input_pipeline(filenames, batch_size, num_options, num_epochs=None):
    filename_queue = tf.train.string_input_producer(
        filenames, num_epochs=num_epochs, shuffle=True)
    context_parsed, sequence_parsed = read_my_file_format(filename_queue, num_options)
    #creating list to define the structure of the batch output
    #['des_len','ques_len','opt_0_len',.....,'des_seq','ques_seq','opt_0_seq',.....,'sol_seq','img_seq']
    data_batch=[context_parsed['des_len'],context_parsed['ques_len']]
    for i in range(num_options):
        data_batch.append(context_parsed['opt_'+str(i)+'_len'])
    data_batch.extend([sequence_parsed['des_seq'],sequence_parsed['ques_seq']])
    for i in range(num_options):
        data_batch.append(sequence_parsed['opt_'+str(i)+'_seq'])
    data_batch.append(sequence_parsed['sol_seq'])
    data_batch.append(sequence_parsed['img_seq'])
    batched_data = tf.train.batch(
      data_batch, batch_size=batch_size, dynamic_pad=True)
    return batched_data