from .song_loader import Song
import numpy as np
import librosa


class SongAnalyzer:
    def __init__(
        self,
        song: Song,
        tempo: float = None,
        low_tempo: bool = True,
        drop_beats: int = None,
    ):
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

        #################
        # Features

        self.low_tempo = low_tempo
        self.tempo = tempo
        self.beat_frames = None

        self.spectogram = None
        self.time_index_ratio = None
        self.frequencies_index_ratio = None

        self.set_frequencies()

        self.drop_beats = drop_beats

    def detect_tempo(self):
        """Detects tempo of a track.
        """

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
        """Adjusts the BPM for more coherence (e.g. turning 160 BPM into 80 BPM)
        """

        THRESOLD = 110

        if self.low_tempo:

            while self.tempo > THRESOLD:
                self.tempo = self.tempo / 2

        self.tempo = round(self.tempo, 0)

    def set_frequencies(self):
        """Choose the value by its time and frequency.
        """

        # getting a matrix which contains amplitude values according to frequency and time indexes
        stft = np.abs(
            librosa.stft(self.song.mono_waveform, hop_length=512, n_fft=2048 * 4)
        )

        self.spectrogram = librosa.amplitude_to_db(
            stft, ref=np.max
        )  # converting the matrix to decibel matrix

        frequencies = librosa.core.fft_frequencies(
            n_fft=2048 * 4
        )  # getting an array of frequencies

        # getting an array of time periodic
        times = librosa.core.frames_to_time(
            np.arange(self.spectogram.shape[1]),
            sr=self.song.sr,
            hop_length=512,
            n_fft=2048 * 4,
        )

        self.time_index_ratio = len(times) / times[len(times) - 1]

        self.frequencies_index_ratio = (
            len(frequencies) / frequencies[len(frequencies) - 1]
        )

    def get_decibel(self, target_time, freq):
        """Multiply the time and the frequency we want by the ratio to get the indexes.
        """
        return self.spectrogram[int(freq * self.frequencies_index_ratio)][
            int(target_time * self.time_index_ratio)
        ]

