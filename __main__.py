import sys
import os
current_file_path = os.path.dirname(__file__).split("/")[:-1]
sys.path.append("/".join(current_file_path))
from glyphgan.libs.model import *


class GlyphGan:
    def __init__(self):
        self.location = build_log_dir('training')
        specs = {'dir': "glyphs", 'batch_size': 1024, 'x_size': 64, 'y_size': 64, 'epochs': 2500, 'lr': 0.0002, 'd_loops': 5, 'embedding': 126}
        self.wpgan = WPGAN(specs, self.location)
        self.wpgan_trained = WPGAN(specs, self.location)
        self.wpgan.train()
        self.wpgan_trained.freeze_generator()


if __name__ == "__main__":
    GlyphGan()

tf.logging.set_verbosity(old_v)
