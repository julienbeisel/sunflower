import sys

sys.path.insert(0, "..")

from sunflower.song_loader import Song, load_from_disk
from sunflower.song_analyzer import SongAnalyzer
import soundfile as sf
import pandas as pd

df_benchmark = pd.read_csv("data/benchmark/benchmark.csv", sep=";")
df_benchmark["bpm"] = df_benchmark["bpm"].str.replace(",", ".").astype(float)


def compute_limits_detection(timestamps, detection, duration=None):
    """Compute time windows for the GIF montage."""

    # Detect bass

    limit_clips = []

    for t in range(0, len(timestamps)):

        allow_detection = True

        if detection[t] == 1 and allow_detection:

            allow_detection = False

            limit_clips.append(timestamps[t])

        if detection[t] == 0 and not allow_detection:
            allow_detection = True

    if duration:

        limit_clips.append(duration)

    return limit_clips


for row in df_benchmark.itertuples():

    raw_audio, extension = load_from_disk(f"data/benchmark/{row.name}")
    song = Song(raw_audio, extension)
    song_analyzer = SongAnalyzer(song, tempo=row.bpm)

    find_drop = song_analyzer.process_decibel_per_frequencies(
        mode="peak", sensibility=90,
    )

    drop_time = compute_limits_detection(find_drop[0], find_drop[1])[0]

    print(f"Song : {row.name}")
    print(f"Drop found : {round(drop_time,0)} - Real drop : {row.drop}")

