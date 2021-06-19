# %%
# Imports
import sys

sys.path.insert(0, "..")

from sunflower.sunflower.song_loader import Song, load_from_disk
from sunflower.sunflower.utils import export_wav

# %%
# Loading example file

raw_audio, extension = load_from_disk("data/ololo.mp3")

song = Song(raw_audio, extension)

song.print_attributes()

export_wav(song, "data_benchmark/ooo.wav")
