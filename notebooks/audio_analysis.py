# %%
# Imports
import sys

sys.path.insert(0, "..")

from sunflower.song_loader import Song, load_from_disk
from sunflower.song_analyzer import SongAnalyzer
from sunflower.utils import export_wav
from sunflower.benchmark import run_benchmark
from sunflower.song_visualizer import visualize_waveform,visualize_waveform_plotly
import plotly.graph_objects as go
import numpy as np
import librosa
import soundfile as sf

%load_ext autoreload
%autoreload 2

# %%
# Loading example file

raw_audio, extension = load_from_disk("../data_benchmark/chiptune.mp3")

song = Song(raw_audio, extension)

song.print_attributes()

# %% 
# Analyze song

song_analyzer = SongAnalyzer(song)
song_analyzer.detect_tempo()

print(song_analyzer.tempo)

# %%
# Visualization

visualize_waveform_plotly(song,song_analyzer)