# %%
# Imports
import sys

sys.path.insert(0, "..")

from sunflower.song_analyzer import SongAnalyzer
from sunflower.song_loader import Song, load_from_disk

# %%
# Loading example file

raw_audio, extension = load_from_disk("../data_benchmark/test_eq.wav")

song = Song(raw_audio, extension)

song.print_attributes()

# %%
# Analyze song

song_analyzer = SongAnalyzer(song)
