# %%
# Imports
import sys

sys.path.insert(0, "..")

from sunflower.song_loader import Song, load_from_disk
from sunflower.song_analyzer import SongAnalyzer
from sunflower.utils import export_wav
from sunflower.benchmark import run_benchmark
from sunflower.song_visualizer import visualize_waveform, visualize_waveform_plotly
from moviepy.editor import *
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import numpy as np
import librosa
import soundfile as sf
import pygame

# %%
# Loading example file

raw_audio, extension = load_from_disk("../data_benchmark/piano_loop.wav")

song = Song(raw_audio, extension)

song.print_attributes()

# %%
# Analyze song

song_analyzer = SongAnalyzer(song)

# %%
def clamp(min_value, max_value, value):

    if value < min_value:
        return min_value

    if value > max_value:
        return max_value

    return value


class AudioBar:
    def __init__(
        self,
        x,
        y,
        freq,
        decibel,
        color,
        width=50,
        min_height=10,
        max_height=100,
        min_decibel=-80,
        max_decibel=0,
    ):

        self.x, self.y, self.freq = x, y, freq

        self.color = color

        self.width, self.min_height, self.max_height = width, min_height, max_height

        self.height = min_height

        self.min_decibel, self.max_decibel = min_decibel, max_decibel

        self.__decibel_height_ratio = (self.max_height - self.min_height) / (
            self.max_decibel - self.min_decibel
        )

        desired_height = decibel * self.__decibel_height_ratio + self.max_height

        self.height = clamp(self.min_height, self.max_height, desired_height)


def draw_rectangle(audiobar, frame):
    """Draw a rectangle in the frame.
    """

    # Change (top, bottom, left, right) to your coordinates
    left = audiobar.x
    right = left + audiobar.width
    bottom = 0
    top = bottom + audiobar.height

    frame[audiobar.top, left:right] = audiobar.color
    frame[bottom, left:right] = audiobar.color
    frame[top:bottom, left] = audiobar.color
    frame[top:bottom, right] = audiobar.color

    return frame


# %%
# getting a matrix which contains amplitude values according to frequency and time indexes


def color_clip(size, duration, fps=25, color=(50, 50, 50)):
    return ColorClip(size, color, duration=duration)


size = (200, 200)
duration = 10
clip = color_clip(size, duration)

clip.preview()


# %%
pygame.quit()
# %%
clip = VideoFileClip("<path to file>")
final_clip = clip.fl_image(draw_rectangle)
# %%
# getting a matrix which contains amplitude values according to frequency and time indexes
