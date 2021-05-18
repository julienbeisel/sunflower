# %%
# Imports
import sys

sys.path.insert(0, "..")

import librosa
import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
import pygame
import soundfile as sf
from moviepy.audio.AudioClip import AudioArrayClip
from moviepy.editor import *

from sunflower.benchmark import run_benchmark
from sunflower.song_analyzer import SongAnalyzer
from sunflower.song_loader import Song, load_from_disk
from sunflower.song_visualizer import visualize_waveform, visualize_waveform_plotly
from sunflower.utils import export_wav

# %%
# Loading example file

raw_audio, extension = load_from_disk("../data_benchmark/test_eq.wav")

song = Song(raw_audio, extension)

song.print_attributes()

# %%
# Analyze song

song_analyzer = SongAnalyzer(song)
