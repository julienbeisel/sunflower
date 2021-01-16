# %%
# Imports
import sys

sys.path.insert(0, "..")

from sunflower.song_loader import Song, load_from_disk
from sunflower.song_analyzer import SongAnalyzer
from sunflower.utils import export_wav
from sunflower.benchmark import run_benchmark
from sunflower.song_visualizer import visualize_waveform,visualize_waveform_plotly
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import numpy as np
import librosa
import soundfile as sf

%load_ext autoreload
%autoreload 2

# %%
# Loading example file

raw_audio, extension = load_from_disk("../data_benchmark/piano_loop.wav")

song = Song(raw_audio, extension)

song.print_attributes()

# %% 
# Analyze song

song_analyzer = SongAnalyzer(song)
song_analyzer.detect_tempo()

print(song_analyzer.tempo)

# %% 
# getting a matrix which contains amplitude values according to frequency and time indexes

stft = np.abs(librosa.stft(song.mono_waveform, hop_length=512, n_fft=2048*4))
spectrogram = librosa.amplitude_to_db(stft, ref=np.max)


# %% 
# viz

librosa.display.specshow(spectrogram,y_axis='log', x_axis='time')
plt.title('Spectogram')
plt.colorbar(format='%+2.0f dB')
plt.tight_layout()
plt.show()




# %%
# Visualization

visualize_waveform_plotly(song,song_analyzer)