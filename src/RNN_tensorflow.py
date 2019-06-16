import numpy as np
import tensorflow as tf

### implementation of each layers class
## Embedding layer
class Embedding:
    def __init__(self, vocabulary_size, embedding_size, scale=0.08):
        self.V = tf.Variable(tf.random_normal([vocabulary_size, embedding_size], stddev=scale), name='V')

    def __call__(self, x):
        return tf.nn.embedding_lookup(self.V, x)

## imprement RNN by tf.scan
# The input is converted into a vector by the Embedding layer.
class RNN_scan:
    def __init__(self, in_dim, hid_dim, seq_len=None, scale=0.08):
        self.in_dim = in_dim
        self.hid_dim = hid_dim
        
        # Since tanh is used as the activation function, the weight is initialized with the initial value of gorot.
        glorot = tf.cast(tf.sqrt(6/(in_dim + hid_dim*2)), tf.float32)
        self.W = tf.Variable(tf.random_uniform([in_dim+hid_dim, hid_dim], minval=-glorot, maxval=glorot), name='W')
        self.b = tf.Variable(tf.zeros([hid_dim]), name='b')

        self.seq_len = seq_len
        self.initial_state = None

    def __call__(self, x):
        def fn(h_prev, x_and_m):
            x_t, m_t = x_and_m
            inputs = tf.concat([x_t, h_prev], -1)
            h_t = tf.nn.tanh(tf.matmul(inputs, self.W) + self.b)
            # Apply mask: h_t is applied to m_t at 1, and m_t at 0 is ignored
            h_t = m_t * h_t + (1 - m_t) * h_prev
          
            return h_t

        # Correct inputs chronologically
        # shape: [batch_size, max_seqence_length, in_dim] -> [max_seqence_length, batch_size, in_dim]
        x_tmaj = tf.transpose(x, perm=[1, 0, 2])
        
        # Generate Mask & Correct inputs chronologically
        mask = tf.cast(tf.sequence_mask(self.seq_len, tf.shape(x)[1]), tf.float32)
        mask_tmaj = tf.transpose(tf.expand_dims(mask, axis=-1), perm=[1, 0, 2])
        
        if self.initial_state is None:
            batch_size = tf.shape(x)[0]
            self.initial_state = tf.zeros([batch_size, self.hid_dim])
        
        h = tf.scan(fn=fn, elems=[x_tmaj, mask_tmaj], initializer=self.initial_state)
        
        return h[-1]

## imprement RNN by Cell structure
class RNN:
    def __init__(self, hid_dim, seq_len = None, initial_state = None):
        self.cell = tf.nn.rnn_cell.BasicRNNCell(hid_dim)
        self.initial_state = initial_state
        self.seq_len = seq_len
    
    def __call__(self, x):
        if self.initial_state is None:
            self.initial_state = self.cell.zero_state(tf.shape(x)[0], tf.float32)
            
        # Note that outputs are 0 after the length of each series.
        outputs, state = tf.nn.dynamic_rnn(self.cell, x, self.seq_len, self.initial_state)
        return tf.gather_nd(outputs, tf.stack([tf.range(tf.shape(x)[0]), self.seq_len-1], axis=1))

## imprement LSTM by tf.scan
class LSTM:
    def __init__(self, in_dim, hid_dim, seq_len=None, initial_state=None):
        self.in_dim = in_dim
        self.hid_dim = hid_dim

        glorot = tf.cast(tf.sqrt(6/(in_dim + hid_dim*2)), tf.float32)

        # input gate
        self.W_i = tf.Variable(tf.random_uniform([in_dim + hid_dim, hid_dim], minval=-glorot, maxval=glorot), name='W_i')
        self.b_i  = tf.Variable(tf.zeros([hid_dim]), name='b_i')
        
        # forget gate
        self.W_f = tf.Variable(tf.random_uniform([in_dim + hid_dim, hid_dim], minval=-glorot, maxval=glorot), name='W_f')
        self.b_f  = tf.Variable(tf.zeros([hid_dim]), name='b_f')

        # output gate
        self.W_o = tf.Variable(tf.random_uniform([in_dim + hid_dim, hid_dim], minval=-glorot, maxval=glorot), name='W_o')
        self.b_o  = tf.Variable(tf.zeros([hid_dim]), name='b_o')

        # cell
        self.W_c = tf.Variable(tf.random_uniform([in_dim + hid_dim, hid_dim], minval=-glorot, maxval=glorot), name='W_c')
        self.b_c  = tf.Variable(tf.zeros([hid_dim]), name='b_c')

        # mask
        self.seq_len = seq_len
        
        self.initial_state = initial_state
    
    def __call__(self, x):
        def fn(prev_state, x_and_m):
            c_prev, h_prev = prev_state[0], prev_state[1]
            x_t, m_t = x_and_m
            
            inputs = tf.concat([x_t, h_prev], -1)
            
            # each gates
            i_t = tf.nn.sigmoid(tf.matmul(inputs, self.W_i) + self.b_i)
            f_t = tf.nn.sigmoid(tf.matmul(inputs, self.W_f) + self.b_f)
            o_t = tf.nn.sigmoid(tf.matmul(inputs, self.W_o) + self.b_o)

            # cell
            c_t = f_t * c_prev + i_t * tf.nn.tanh(tf.matmul(inputs, self.W_c) + self.b_c)

            # hidden state
            h_t = o_t * tf.nn.tanh(c_t)
            
            # apply mask
            c_t = m_t * c_t + (1 - m_t) * c_prev
            h_t = m_t * h_t + (1 - m_t) * h_prev

            return tf.stack([c_t, h_t])

        # Correct inputs chronologically
        x_tmaj = tf.transpose(x, perm=[1, 0, 2])
        # Generate Mask & Correct inputs chronologicallyスクの生成＆時間順化
        mask = tf.cast(tf.sequence_mask(self.seq_len, tf.shape(x)[1]), tf.float32)
        mask_tmaj = tf.transpose(tf.expand_dims(mask, axis=-1), perm=[1, 0, 2])
        
        if self.initial_state is None:
            batch_size = tf.shape(x)[0]
            self.initial_state = tf.stack([tf.zeros([batch_size, self.hid_dim]), tf.zeros([batch_size, self.hid_dim])])

        state_seq = tf.scan(fn=fn, elems=[x_tmaj, mask_tmaj], initializer=self.initial_state)
        
        return state_seq[-1][1]