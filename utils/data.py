import os, sys
from utils.config import CONFIG


class DataGenerator:

    def __init__(self):
        self.seq_length = CONFIG["seq_length"]
        self.batch_size = CONFIG["batch_size"]
        self.data = ""

        data_dir = os.path.join(sys.path[0], "data")
        data_files = [file for file in os.listdir(data_dir) if file.endswith(".txt")]
        for file_name in data_files:
            f = open(data_dir + "/" + file_name)
            self.data += f.read()
            f.close()

        self.total_length = len(self.data)  # total data length
        self.words = list(set(self.data))
        self.words.sort()
        self.vocabulary_size = len(self.words)
        print('Vocabulary Size: ', self.vocabulary_size)
        self.char2id_dict = {w: i for i, w in enumerate(self.words)}
        self.id2char_dict = {i: w for i, w in enumerate(self.words)}

        # pointer position to generate current batch
        self._pointer = 0

        # save metadata file
        self.save_metadata(CONFIG["metadata"])

    def char2id(self, c):
        return self.char2id_dict[c]

    def id2char(self, _id):
        return self.id2char_dict[_id]

    def save_metadata(self, file):
        with open(file, 'w') as f:
            f.write('id\tchar\n')
            for i in range(self.vocabulary_size):
                c = self.id2char(i)
                f.write('{}\t{}\n'.format(i, c))

    def next_batch(self):
        x_batches = []
        y_batches = []
        for i in range(self.batch_size):
            if self._pointer + self.seq_length + 1 >= self.total_length:
                self._pointer = 0
            bx = self.data[self._pointer: self._pointer + self.seq_length]
            by = self.data[self._pointer +
                           1: self._pointer + self.seq_length + 1]
            self._pointer += self.seq_length  # update pointer position

            # convert to ids
            bx = [self.char2id(c) for c in bx]
            by = [self.char2id(c) for c in by]
            x_batches.append(bx)
            y_batches.append(by)

        return x_batches, y_batches


data = DataGenerator()
