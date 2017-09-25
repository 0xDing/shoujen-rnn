import tensorflow as tf
from tensorflow.contrib import rnn
from tensorflow.contrib import legacy_seq2seq

from utils.config import CONFIG
from utils.data import data



class ShoujenModel:
    """
    Shoujen neural network model.
    """

    def __init__(self, infer=False):
        if infer:
            CONFIG["batch_size"] = 1
            CONFIG["seq_length"] = 1
        with tf.name_scope('inputs'):
            self.input_data = tf.placeholder(
                tf.int32, [CONFIG["batch_size"], CONFIG["seq_length"]]
            )
            self.target_data = tf.placeholder(
                tf.int32, [CONFIG["batch_size"], CONFIG["seq_length"]]
            )

        with tf.name_scope('model'):
            self.cell = rnn.GRUCell(CONFIG["hidden_size"])
            self.cell = rnn.MultiRNNCell([self.cell] * CONFIG["num_layers"])
            self.initial_state = self.cell.zero_state(
                CONFIG["batch_size"], tf.float32
            )
            with tf.variable_scope('rnnlm'):
                softmax_w = tf.get_variable(
                    'softmax_w', [CONFIG["hidden_size"], data.vocabulary_size]
                )
                softmax_b = tf.get_variable('softmax_b', [data.vocabulary_size])
                embedding = tf.get_variable('embedding', [data.vocabulary_size, CONFIG["hidden_size"]])
                inputs = tf.nn.embedding_lookup(embedding, self.input_data)
            outputs, last_state = tf.nn.dynamic_rnn(self.cell, inputs, initial_state=self.initial_state)

        with tf.name_scope('loss'):
            output = tf.reshape(outputs, [-1, CONFIG["hidden_size"]])

            self.logits = tf.matmul(output, softmax_w) + softmax_b
            self.probs = tf.nn.softmax(self.logits)
            self.last_state = last_state

            targets = tf.reshape(self.target_data, [-1])
            loss = legacy_seq2seq.sequence_loss_by_example(
                [self.logits],
                [targets],
                [tf.ones_like(targets, dtype=tf.float32)]
            )
            self.cost = tf.reduce_sum(loss) / CONFIG["batch_size"]
            tf.summary.scalar('loss', self.cost)

        with tf.name_scope('optimize'):
            self.lr = tf.placeholder(tf.float32, [])
            tf.summary.scalar('learning_rate', self.lr)
            optimizer = tf.train.AdamOptimizer(self.lr)
            tvars = tf.trainable_variables()
            grads = tf.gradients(self.cost, tvars)
            for g in grads:
                tf.summary.histogram(g.name, g)
            grads, _ = tf.clip_by_global_norm(grads, CONFIG["grad_clip"])

            self.train_op = optimizer.apply_gradients(zip(grads, tvars))
            self.merged_op = tf.summary.merge_all()


