import tensorflow as tf
import numpy as np
from tensorflow.contrib import rnn

class MyLSTM(object):

    def My_LSTM(self, x, dropout, scope, embedding_size, sequence_length, batch_size, hidden_units):
        n_input = embedding_size
        n_steps = sequence_length
        n_hidden = hidden_units
        n_layers = 1

        # input : x , shape: [batch_size, n_steps, hidden_size]


        def lstm_cell():
            l_cell = tf.contrib.rnn.BasicLSTMCell(n_hidden, forget_bias=1.0, state_is_tuple=True)
            return tf.contrib.rnn.DropoutWrapper(l_cell, output_keep_prob=dropout)
        cell = tf.contrib.rnn.MultiRNNCell([lstm_cell() for _ in range(n_layers)],
                                                 state_is_tuple=True)

        self._initial_state = cell.zero_state(batch_size, dtype=tf.float32)  # 参数初始化

        outputs = []
        state = self._initial_state  # state 表示 各个batch中的状态
        with tf.variable_scope("RNN"):
            for time_step in range(n_steps):
                if time_step > 0:
                    tf.get_variable_scope().reuse_variables()
                # cell_out: [batch, hidden_size]
                (cell_output, state) = cell(x[:, time_step, :], state)
                outputs.append(cell_output)  # output: shape[num_steps][batch,hidden_size]

        return outputs

    def __init__(
      self, sequence_length, embedding_size, hidden_units, class_num, l2_reg_lambda, batch_size):

      # Placeholders for input, output and dropout
      self.input_x = tf.placeholder(tf.float32, [None, sequence_length, embedding_size], name="input_x")
      self.input_y = tf.placeholder(tf.int64, [None], name="input_y")
      self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")
      self.b_size = tf.placeholder(tf.int32, [], name='batch_size')
      # Keeping track of l2 regularization loss (optional)
      l2_loss = tf.constant(0.0, name="l2_loss")

      # Create a convolution + maxpool layer for each filter size
      with tf.name_scope("output"):
        self.out1 = self.My_LSTM(self.input_x, self.dropout_keep_prob, "lstm", embedding_size, sequence_length, self.b_size, hidden_units)


      with tf.name_scope("mean_pooling_layer"):
         # 检查维度0还是1
         out_put = tf.reduce_mean(self.out1, 0)

      with tf.name_scope("Softmax_layer_and_output"):
         softmax_w = tf.get_variable("softmax_w",[hidden_units,class_num],dtype=tf.float32)
         softmax_b = tf.get_variable("softmax_b",[class_num],dtype=tf.float32)
         self.logits = tf.matmul(out_put,softmax_w)+softmax_b

      with tf.name_scope("loss"):
        self.loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.logits+1e-10,labels=self.input_y)
        self.cost = tf.reduce_mean(self.loss)

      with tf.name_scope("accuracy"):
        self.prediction = tf.argmax(self.logits,1,name='prediction')
        correct_prediction = tf.equal(self.prediction,self.input_y)
        self.correct_num=tf.reduce_sum(tf.cast(correct_prediction,tf.float32))
        self.accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32),name="accuracy")

