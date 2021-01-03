from .song_loader import Song
import librosa


class SongAnalyzer:
    def __init__(self, song: Song, tempo: float = None, low_tempo: bool = True):
        """Creates a SongAnalyzer object.

        :param song: Song object
        :param tempo: Sets the BPM of the track
        :param low_tempo: Type of song to analyze.
        Setting the mode to True ensures to have a BPM lower to a threshold (110 BPM)

        """

        ######################
        # Loaded song

        self.song = song

        #################
        # Features

        self.low_tempo = low_tempo
        self.tempo = tempo
        self.beat_frames = None

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

