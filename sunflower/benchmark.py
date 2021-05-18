import json

from .song_analyzer import SongAnalyzer
from .song_loader import Song, load_from_disk


def run_benchmark(folder="../data/"):
    """Check the accuracy of the algorithms."""

    with open(f"{folder}info_benchmark.json") as json_file:
        info_benchmark = json.load(json_file)

    for filename in info_benchmark:

        raw_audio, extension = load_from_disk(f"{folder}{filename}")

        song = Song(raw_audio, extension)

        song_analyzer = SongAnalyzer(song)
        song_analyzer.detect_tempo()
        tempo = song_analyzer.tempo

        possible_values = [round(tempo / 2, 0), round(tempo, 0), round(2 * tempo, 0)]

        closest_tempo = min(
            possible_values,
            key=lambda x: abs(x - info_benchmark.get(filename).get("BPM")),
        )

        info_benchmark[filename]["Found BPM"] = closest_tempo

    return info_benchmark
