import matplotlib.pyplot as plt
import librosa.display
import numpy as np
import plotly.graph_objects as go


def visualize_waveform(song):
    """Waveform visualization.
    """

    plt.figure(figsize=(16, 4))
    librosa.display.waveplot(song.mono_waveform, sr=song.sr)
    plt.title("Waveform Visualizer (mono)")
    plt.show()


def visualize_waveform_plotly(song, song_analyzer, stereo=False, rate=30):

    stereo = False
    rate = 30

    left = song.waveform[0][0::rate]
    right = song.waveform[1][0::rate]
    mono = song.mono_waveform[0::rate]

    x = np.arange(0, len(mono)) * rate / song.sr

    if stereo:
        data = [
            go.Scatter(x=x, y=left, name="Left"),
            go.Scatter(x=x, y=right, name="Right"),
        ]
    else:
        data = [go.Scatter(x=x, y=mono, name="Mono")]

    fig = go.Figure(data=data)

    beats = librosa.frames_to_time(song_analyzer.beat_frames, song.sr)

    fig.add_trace(
        go.Scatter(
            x=beats,
            y=np.zeros(len(beats)),
            mode="markers",
            marker_symbol="diamond-tall",
            marker_size=10,
            name="Beat Rythm",
        )
    )

    fig.update_layout(
        title=f"Waveform Visualizer <br>BPM :{round(song_analyzer.tempo,0)}"
    )

    return fig
