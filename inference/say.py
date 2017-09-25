import sys
import scipy
import time
import tensorflow as tf

from utils.data import data
from utils.config import CONFIG
from models.shoujen import ShoujenModel


def say():
    model = ShoujenModel(True)
    saver = tf.train.Saver()
    with tf.Session() as session:
        checkpoint = tf.train.latest_checkpoint(CONFIG["log_dir"])
        saver.restore(session, checkpoint)

        prime = u'盡夫天理之極而無一毫人欲之私者得之'  # warm RNN
        state = session.run(model.cell.zero_state(1, tf.float32))

        for word in prime[:-1]:
            x = scipy.zeros((1, 1))
            x[0, 0] = data.char2id(word)
            feed_dict = {model.input_data: x, model.initial_state: state}
            state = session.run(model.last_state, feed_dict)

        word = prime[-1]
        says = prime
        for i in range(CONFIG["default_paragraph_length"]):
            x = scipy.zeros([1, 1])
            x[0, 0] = data.char2id(word)
            feed_dict = {model.input_data: x, model.initial_state: state}
            probs, state = session.run([model.probs, model.last_state], feed_dict)
            p = probs[0]
            word = data.id2char(scipy.argmax(p))
            sys.stdout.flush()
            says += word
        return print(says)
