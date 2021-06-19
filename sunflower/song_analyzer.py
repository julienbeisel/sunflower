import librosa
import numpy as np

from .song_loader import Song

BASS_RANGE = {"name": "bass", "start": 50, "stop": 80}
HEAVY_RANGE = {"name": "heavy_range", "start": 101, "stop": 250}
WHOLE_SPECTRUM_RANGE = {"name": "bass", "start": 20, "stop": 12000}


class SongAnalyzer:
    def __init__(self, song: Song, tempo: float = None, low_tempo: bool = True):
        """Creates a SongAnalyzer object.

        :param song: Song object
        :param tempo: Sets the BPM of the track
        :param low_tempo: Type of song to analyze.
        Setting the mode to True ensures to have a BPM lower to a threshold (110 BPM)
        :param drop_beats: When the song drops

        """

        ######################
        # Loaded song

        self.song = song

        ######################
        # Features

        self.low_tempo = low_tempo
        self.tempo = tempo
        self.beat_frames = None

        self.spectogram = None
        self.time_index_ratio = None
        self.frequencies_index_ratio = None

        self.set_frequencies()

        ######################
        # Features

    def detect_tempo(self):
        """Detects tempo of a track."""

        if (self.song.sr is None) or (self.song.waveform is None):
            raise ValueError("No song was loaded.")

        # Detect tempo

        if self.tempo:

            self.tempo, beat_frames = librosa.beat.beat_track(
                y=self.song.mono_waveform, sr=self.song.sr, bpm=self.tempo
            )

        else:
            self.tempo, beat_frames = librosa.beat.beat_track(
                y=self.song.mono_waveform, sr=self.song.sr, tightness=100
            )

        self.adjust_tempo()
        self.beat_frames = beat_frames

    def adjust_tempo(self) -> float:
        """Adjusts the BPM for more coherence (e.g. turning 160 BPM into 80 BPM)"""

        THRESOLD = 110

        if self.low_tempo:

            while self.tempo > THRESOLD:
                self.tempo = self.tempo / 2

        self.tempo = round(self.tempo, 0)

    def set_frequencies(self):
        """Choose the value by its time and frequency."""

        # getting a matrix which contains amplitude values according to frequency and time indexes
        stft = np.abs(
            librosa.stft(
                self.song.mono_waveform_analysis, hop_length=512, n_fft=2048 * 4
            )
        )

        self.spectrogram = librosa.amplitude_to_db(
            stft, ref=np.max
        )  # converting the matrix to decibel matrix

        frequencies = librosa.core.fft_frequencies(
            n_fft=2048 * 4
        )  # getting an array of frequencies

        # getting an array of time periodic
        times = librosa.core.frames_to_time(
            np.arange(self.spectrogram.shape[1]),
            sr=self.song.sr,
            hop_length=512,
            n_fft=2048 * 4,
        )

        self.time_index_ratio = len(times) / times[len(times) - 1]

        self.frequencies_index_ratio = (
            len(frequencies) / frequencies[len(frequencies) - 1]
        )

    def get_decibel(self, target_time, freq):
        """Multiply the time and the frequency we want by the ratio to get the indexes."""
        return self.spectrogram[int(freq * self.frequencies_index_ratio)][
            int(target_time * self.time_index_ratio)
        ]

    def process_decibel_per_frequencies(
        self,
        rate_frequencies=1 / 12,
        rate_duration=1 / 16,
        mode="peak",
        freq_study: str = "bass",
        sensibility=None,
        drop_time=0,
    ):
        """Get average frequencies.

        :param mode: Options: avg, peak
        :param rate_frequencies:
        :param rate_duration: % of the bpm duration
        :param freq_range: frequence range to study

        TODO: reduce complexity
        """

        if freq_study == "bass":
            freq_range = BASS_RANGE
        elif freq_study == "whole":
            freq_range = WHOLE_SPECTRUM_RANGE
        else:
            raise ValueError("Please provide a correct frequency range to study")

        # Note: These timestramps cant be computed with np.linspace
        # We know the song starts at the beginning of a measure, but we don't know it it
        # ends at the end of a measure (reverb, delay etc.)

        # Timestamps to compute energy
        beat_duration = 60 / self.tempo
        song_duration = librosa.get_duration(self.song.waveform, sr=self.song.sr)

        timestamps = []

        timestamp_meas = 0

        while timestamp_meas < song_duration:

            timestamps.append(timestamp_meas)
            timestamp_meas += beat_duration * rate_duration

        if sensibility is None:
            raise ValueError("Automatic sensibility computing not implemented yet")

        # Timestamps corresponding to each measure start

        timestamps_measures = []

        timestamp_meas = 0

        while timestamp_meas < song_duration:

            timestamps_measures.append(timestamp_meas)
            timestamp_meas += beat_duration

        if mode == "peak":
            results = [timestamps]
        else:
            results = []

        list_db = []

        # Sub-range of frequencies to study
        step = np.ceil((freq_range["stop"] - freq_range["start"]) * rate_frequencies)

        for timestamp in timestamps:

            start = freq_range["start"]
            stop = start + step

            # Store all the db computed for the frequency range
            list_db_range = []

            # It is sometimes useful to not consider the whole freq_range, but
            # segment it

            while stop <= freq_range["stop"]:

                # Computing db at the start of the window
                db = (
                    self.get_decibel(timestamp, start)
                    + self.get_decibel(timestamp, start + (stop - start) / 2)
                ) / 2

                list_db_range.append(db)
                start = stop
                stop = start + step

            list_db.append(np.array(list_db_range).mean())

        if mode == "avg":

            results.append(np.mean(list_db))

        elif mode == "peak":

            reference_value = np.percentile(list_db, sensibility)
            list_db = np.where(list_db < reference_value, 0, 1)
            results.append(list_db)

        return results
