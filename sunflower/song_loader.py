import io
import pydub
import numpy as np
import soundfile as sf
import librosa

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
        self.extension = None
        self.channels = None
        self.sr = None
        self.sample_width = None

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

            waveform = waveform.reshape((2, -1)).astype("float32")

        else:

            waveform = waveform.astype("float32")

        # Normalization
        waveform = waveform / (self.sample_width ** 15)
        self.waveform = waveform
        self.mono_waveform = librosa.to_mono(self.waveform)
        self.sr = a.frame_rate

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

        self.mono_waveform = librosa.to_mono(self.waveform)


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
