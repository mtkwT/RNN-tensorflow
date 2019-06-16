import numpy as np
import tensorflow as tf
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tqdm import tqdm
from src.RNN_tensorflow import Embedding, RNN_scan, RNN, LSTM

# Calculating log (Overflow Prevention)
def tf_log(x):
    return tf.log(tf.clip_by_value(x, 1e-10, x))

def main():
    ## import dataset
    pad_index = 0
    num_words = 10000
    (x_train, t_train), (x_test, t_test) = imdb.load_data(num_words=num_words)
    x_train, x_valid, t_train, t_valid = train_test_split(x_train, t_train, test_size=0.2, random_state=42)

    # Since more padding is computationally inefficient, the data is sorted in descending order to reduce padding.
    x_train_lens = [len(com) for com in x_train]
    sorted_train_indexes = sorted(range(len(x_train_lens)), key=lambda x: -x_train_lens[x])
    x_train = [x_train[idx] for idx in sorted_train_indexes]
    t_train = [t_train[idx] for idx in sorted_train_indexes]

    tf.reset_default_graph() # initialization
    emb_dim = 100
    hid_dim = 50

    x = tf.placeholder(tf.int32, [None, None], name='x')
    t = tf.placeholder(tf.float32, [None, None], name='t')

    seq_len = tf.reduce_sum(tf.cast(tf.not_equal(x, pad_index), tf.int32), axis=1)

    ### Building Calculation Graphs & Updating Parameters
    ## Plain RNN
    # h = Embedding(num_words, emb_dim)(x)
    # h = RNN_scan(emb_dim, hid_dim, seq_len)(h)
    # h = RNN(hid_dim, seq_len)(h)
    # y = tf.layers.Dense(1, tf.nn.sigmoid)(h)

    # LSTM
    h = Embedding(num_words, emb_dim)(x)
    h = LSTM(emb_dim, hid_dim, seq_len)(h)
    h = tf.layers.Dense(20, tf.nn.relu)(h)
    h = tf.layers.Dropout(rate=0.05)(h)
    y = tf.layers.Dense(1, tf.nn.sigmoid)(h)

    cost = -tf.reduce_mean(t*tf_log(y) + (1 - t)*tf_log(1 - y))

    train = tf.train.AdamOptimizer().minimize(cost)
    test = tf.round(y)

    ## Train & Validation
    n_epochs = 5
    batch_size = 100
    n_batches_train = len(x_train) // batch_size
    n_batches_valid = len(x_valid) // batch_size

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for epoch in range(n_epochs):
            print("epoch{}:".format(epoch))
            # Train
            train_costs = []
            for i in tqdm(range(n_batches_train)):
                start = i * batch_size
                end = start + batch_size
                
                x_train_batch = np.array(pad_sequences(x_train[start:end], padding='post', value=pad_index)) # Padding per batch
                t_train_batch = np.array(t_train[start:end])[:, None]

                _, train_cost = sess.run([train, cost], feed_dict={x: x_train_batch, t: t_train_batch})
                train_costs.append(train_cost)
        
            # Valid
            valid_costs = []
            y_pred = []
            for i in range(n_batches_valid):
                start = i * batch_size
                end = start + batch_size
                
                x_valid_pad = np.array(pad_sequences(x_valid[start:end], padding='post', value=pad_index)) # Padding per batch
                t_valid_pad = np.array(t_valid[start:end])[:, None]
                
                pred, valid_cost = sess.run([test, cost], feed_dict={x: x_valid_pad, t: t_valid_pad})
                y_pred += pred.flatten().tolist()
                valid_costs.append(valid_cost)
            print('EPOCH: %i, Training Cost: %.3f, Validation Cost: %.3f, Validation F1: %.3f' % (epoch+1, np.mean(train_costs), np.mean(valid_costs), f1_score(t_valid, y_pred, average='macro')))

if __name__ == "__main__":
    main()   