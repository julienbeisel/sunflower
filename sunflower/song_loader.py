import io

import librosa
import numpy as np
import pydub

ALLOWED_EXTENSIONS = {"mp3", "wav"}


class Song:
    def __init__(self, filelike, extension: str):
        """Creates a Song object.

        :param filelike: Song in bytes
        :param extension: Extension of the song
        Extensions available : ['mp3','wav']

        Attributes:

        waveform: Numpy array representing the song (stereo)
        mono_waveform: Numpy array representing the song (mono)
        extension: Extension of the file
        channels: Number of channels of the song
        sr: Sample rate
        sample_width: Sample width
        """

        ######################
        # Basic audio features

        self.waveform = None
        self.mono_waveform = None

        # Waveforms with fortran style reshape
        self.waveform_analysis = None
        self.mono_waveform_analysis = None

        self.extension = None
        self.channels = None
        self.sr = None
        self.sample_width = None
        self.bitrate = None

        self.load_from_filelike(filelike, extension)

        #################
        # Processing song

        self.process_song()

    def load_from_filelike(self, filelike, extension: str):
        """Filelike to librosa.

        :param filelike: Song in bytes
        :param extension: Extension of the song
        Extensions available : ['mp3','wav']
        """

        self.extension = extension

        if extension == "mp3":
            a = pydub.AudioSegment.from_mp3(filelike)
        elif extension == "wav":
            a = pydub.AudioSegment.from_wav(filelike)
        else:
            raise ValueError("Wrong extension: Format not supported.")

        # Converting to float32 for librosa
        waveform = np.array(a.get_array_of_samples())

        self.sample_width = a.sample_width
        self.channels = a.channels

        if self.channels == 2:

            waveform_analysis = waveform.reshape(self.channels, -1, order="F").astype(
                "float32"
            )
            waveform = waveform.reshape(self.channels, -1).astype("float32")

        else:

            waveform = waveform.astype("float32")
            waveform_analysis = waveform

        # Normalization
        waveform = normalize(waveform, self.sample_width)
        waveform_analysis = normalize(waveform_analysis, self.sample_width)

        self.waveform = waveform
        self.waveform_analysis = waveform_analysis

        self.sr = a.frame_rate

        self.bitrate = compute_bitrate(self.sr, self.sample_width, self.channels)

    def print_attributes(self) -> None:
        """Print attributes of the object."""

        attrs = vars(self)
        print(", ".join("%s: %s" % item for item in attrs.items()))

    def process_song(self) -> None:
        """Removes silence at the beginning of the song.

        TO-DO: Fine-tune top_db
        """

        self.waveform, _ = librosa.effects.trim(
            self.waveform, frame_length=128, hop_length=32, top_db=40
        )

        difference = self.waveform.shape[1] - self.waveform_analysis.shape[1]

        self.waveform_analysis = self.waveform_analysis[:, difference:]

        self.mono_waveform = librosa.to_mono(self.waveform)
        self.mono_waveform_analysis = librosa.to_mono(self.waveform_analysis)


def normalize(waveform, sample_width):
    """Normalize waveform."""

    return waveform / (2 ** (8 * sample_width - 1))


def compute_bitrate(frame_rate, frame_width, channels):
    """Formula to compute bitrate.

    Source: https://stackoverflow.com/questions/33747728/how-can-i-get-the-same-bitrate-of-input-and-output-file-in-pydub
    """

    bitrate = (frame_rate * frame_width * 8 * channels) / 1000

    return bitrate


def allowed_file(filename: str) -> (bool, str):
    """Check if the file extension is allowed."""

    extension = ""
    allowed = False

    if "." in filename:
        extension = filename.rsplit(".", 1)[1].lower()

        allowed = filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS

    return (allowed, extension)


def load_from_disk(file_path: str):
    """Load a song from disk to emulate a GET request"""

    f = open(file_path, "rb")
    filename = f.name

    allowed, extension = allowed_file(filename)

    if not allowed:
        raise ValueError(
            f"File extension not allowed. Allowed extensions :{ALLOWED_EXTENSIONS}"
        )

    data_song = io.BytesIO(f.read())

    return data_song, extension
