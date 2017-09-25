#!/usr/local/bin/python3
from fire import Fire


class Cli(object):
    """Shoujen-RNN - Shoujen speech Generator, implement a gated recurrent unit neural network."""

    @staticmethod
    def training():
        from inference.training import training
        training()

    @staticmethod
    def say():
        from inference.say import say
        say()


if __name__ == '__main__':
    Fire(Cli)
