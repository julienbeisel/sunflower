from sunflower.song_analyzer import SongAnalyzer
from sunflower.song_loader import Song, load_from_disk
from sunflower.utils import export_wav


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


raw_audio, extension = load_from_disk(
    "data/benchmark/ZEU - LINCOLN FREESTYLE 18K (PROD. EPEK & PANDREZZ).mp3"
)

song = Song(raw_audio, extension)

export_wav(song, "data/test.wav")

song_analyzer = SongAnalyzer(song, tempo=71)

find_drop = song_analyzer.process_decibel_per_frequencies(mode="peak", sensibility=90,)

drop_time = round(compute_limits_detection(find_drop[0], find_drop[1])[0], 0)

print(drop_time)
