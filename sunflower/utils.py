import numpy as np
import soundfile as sf

from .song_loader import Song


def export_wav(song: Song, path):
    """TO DO: Move this function somewhere else.

    --- Just used for tests atm ---
    """

    sf.write(path, song.waveform.reshape(-1, song.channels), song.sr, subtype="FLOAT")
