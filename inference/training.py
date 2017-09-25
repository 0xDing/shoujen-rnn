import os
import tensorflow as tf

from utils.config import CONFIG
from utils.data import data
from models.shoujen import ShoujenModel


def training():
    model = ShoujenModel()
    with tf.Session() as session:
        session.run(tf.global_variables_initializer())
        saver = tf.train.Saver()
        writer = tf.summary.FileWriter(CONFIG["log_dir"], session.graph)

        max_iter = CONFIG["n_epoch"] * \
            (data.total_length // CONFIG["seq_length"]) // CONFIG["batch_size"]
        for i in range(max_iter):
            learning_rate = CONFIG["learning_rate"] * \
                (CONFIG["decay_rate"] ** (i // CONFIG["decay_steps"]))
            x_batch, y_batch = data.next_batch()
            feed_dict = {
                model.input_data: x_batch,
                model.target_data: y_batch,
                model.lr: learning_rate
            }
            train_loss, summary, _, _ = session.run([model.cost, model.merged_op, model.last_state, model.train_op],
                                                    feed_dict)
            if i % 10 == 0:
                writer.add_summary(summary, global_step=i)
                print('Step:{}/{}, training_loss:{:4f}'.format(i, max_iter, train_loss))

            if i % 2000 == 0 or (i+1) == max_iter:
                saver.save(session, os.path.join(CONFIG["log_dir"], 'shoujen_model.ckpt'), global_step=i)
