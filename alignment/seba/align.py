import os  # nowa: autoimport
import shutil
from math import floor, log2
from subprocess import DEVNULL, Popen
from typing import Tuple

import essentia.standard as esst
import fastdtw as fdtw
import librosa
import numpy as np
import pretty_midi
import scipy
from dtw import dtw

from .. import cdist, utils
from .dlnco.DLNCO import dlnco


def check_executables():
    """
    Just check that `fluidsynth` is available in the path and raise a
    RuntimeError if not
    """
    if shutil.which('fluidsynth') is None:
        raise RuntimeError(
            "Please, install fluidsynth command and make it available in your PATH environment variable"
        )


def download_soundfont():
    """
    Just download MuseScore 3 soundfont to  `./soundfont.sf2`
    """
    sf2_path = "soundfont.sf2"
    if not os.path.exists(sf2_path):
        import urllib.request  # noqa: autoimport
        url = "https://ftp.osuosl.org/pub/musescore/soundfont/MuseScore_General/MuseScore_General.sf2"
        print("downloading...")
        urllib.request.urlretrieve(url, sf2_path)


# hack to let fastdtw accept float32
def _my_prep_inputs(x, y, dist):
    return x, y


def fdtw_dist(sample1, sample2):
    """
    This functions computes the eculidean distance on the first `12` featres and
    cosine distance on the remaining. Then it sums the two distances and returns
    """
    return cdist.euclidean(sample1[:12], sample2[:12]) + cdist.cosine(
        sample1[12:], sample2[12:])


def multiple_audio_alignment(audio1,
                             sr1,
                             audio2,
                             sr2,
                             hopsize,
                             n_fft=4096,
                             merge_dlnco=True,
                             fastdtw=False):
    """
    Aligns two audio files and returns a list of lists containing the map
    between the audio frames.

    Parameters
    ----------

    audio1 : np.array
        Numpy array representing the signal.
    sr1 : int
        Sampling rate of **audio1**
    audio2 : np.array
        Numpy array representing the signal.
    sr2 : int
        Sampling rate of **audio2**
    hopsize : int
        The hopsize for the FFT. Consider to use something like `n_fft/4`
    n_fft : int
        The window size. Consider to use something like `4*hopsize`
    merge_dlnco : bool
        Unknown


    Returns
    -------
     numpy.ndarray
        A 2d array, mapping frames from :attr: `audio1` to frames in
        :attr: `audio2`. `[[frame in audio 1, frame in audio 2]]`

    """

    # chroma and DLNCO features
    # output shape is (features, frames)
    # transposed -> (frames, features)
    audio1_chroma = librosa.feature.chroma_stft(y=audio1,
                                                sr=sr1,
                                                tuning=0,
                                                norm=2,
                                                hop_length=hopsize,
                                                n_fft=n_fft).T
    audio1_dlnco = dlnco(audio1, sr1, n_fft, hopsize).T

    audio2_chroma = librosa.feature.chroma_stft(y=audio2,
                                                sr=sr2,
                                                tuning=0,
                                                norm=2,
                                                hop_length=hopsize,
                                                n_fft=n_fft).T
    audio2_dlnco = dlnco(audio2, sr2, n_fft, hopsize).T

    L = min(audio1_dlnco.shape[0], audio2_dlnco.shape[0])
    if not fastdtw:
        dlnco_mat = scipy.spatial.distance.cdist(audio1_dlnco[:L, :],
                                                 audio2_dlnco[:L, :],
                                                 'euclidean')
        chroma_mat = scipy.spatial.distance.cdist(audio1_chroma[:L, :],
                                                  audio2_chroma[:L, :],
                                                  'cosine')

        # print("Starting DTW")
        res = dtw(dlnco_mat + chroma_mat)
        wp = np.stack([res.index1, res.index2], axis=1)
        return wp
    else:
        # shape of features is still (frames, features)
        features1 = np.concatenate([audio1_chroma, audio1_dlnco], axis=1)
        features2 = np.concatenate([audio2_chroma, audio2_dlnco], axis=1)
        fdtw._fastdtw.__prep_inputs = _my_prep_inputs
        _D, path = fdtw.fastdtw(features1.astype(np.float32),
                                features2.astype(np.float32),
                                dist=fdtw_dist,
                                radius=98)
        return np.asarray(path)


def audio_to_midi_alignment(midi,
                            audio,
                            sr,
                            hopsize,
                            n_fft=4096,
                            merge_dlnco=True,
                            sf2_path="soundfont.sf2",
                            fastdtw=False):
    """
    Synthesize midi file, align it to :attr: `audio` and returns a mapping
    between midi times and audio times.

    Parameters
    ----------

    midi : :class: `pretty_midi.PrettyMIDI`
        The midi file that will be aligned
    audio : np.array
        Numpy array representing the signal.
    sr : int
        Sampling rate of **audio1**
    hopsize : int
        The hopsize for the FFT. Consider to use something like `n_fft/4`
    n_fft : int
        The window size. Consider to use something like `4*hopsize`
    merge_dlnco : bool
        Unknown
    sf2_path: str
        Path to a soundfont (default to 'soundfont.sf2')

    Returns
    -------
    numpy.ndarray
        A 2d array, mapping time from :attr: `midi` to times in
        :attr: `audio`. `[[midi time, audio time]]`

    """
    fname = str(os.getpid())
    midi.write(fname + ".mid")
    popen = Popen(
        [
            "fluidsynth", "-ni", sf2_path, fname + ".mid", "-F",
            fname + ".wav", "-r",
            str(sr)
        ],
        stderr=DEVNULL,
        stdout=DEVNULL,
    )
    popen.wait()
    if popen.returncode > 0:
        raise RuntimeError("Cannot synthesize song!")
    audio1 = esst.EasyLoader(filename=fname + ".wav", sampleRate=sr)()
    os.remove(fname + ".mid")
    os.remove(fname + ".wav")

    audio2 = audio
    wp = multiple_audio_alignment(audio1,
                                  sr,
                                  audio2,
                                  sr,
                                  hopsize,
                                  n_fft=n_fft,
                                  fastdtw=fastdtw)
    wp_times = np.asarray(wp) * hopsize / sr

    return wp_times


def audio_to_score_alignment(matscore: np.ndarray,
                             audio_data: Tuple[np.ndarray, int],
                             res=0.02,
                             merge_dlnco=True,
                             sf2_path="soundfont.sf2",
                             fastdtw=False):
    """
    Synthesize an asmd score, align it to :attr: `audio` and returns a mapping
    between midi times and audio times.

    Parameters
    ----------

    matscore : numpy.ndarray
        A matrix representing notes from which a midi file is contructed. Each
        row is a note and columns are: pitches, onsets, offsets, *something*,
        program, *anything else*.
    audio_data : np.array, int
        Numpy array representing the signal and the sample rate
    res : float
        The width of each column of the spectrogram. This will define the
        hopsize as 2**floor(log2(sr*time_precision)) and the length of the
        window size as 4*hopsize
    merge_dlnco : bool
        Unknown

    Returns
    -------
    list :
        a list of floats representing the new computed onsets
    list :
        a list of floats representing the new computed offsets

    If audio is too much short, return None

    """
    audio, sr = audio_data
    if audio.size < sr * 0.25:
        return None

    # removing trailing silence
    start, stop = utils.find_start_stop(audio, sample_rate=sr)
    _audio = audio[start:stop]
    if _audio.size > sr * 0.25:
        audio = _audio
    else:
        # trimming was unsuccessful
        # use the audio not trimmed
        start = 0
        stop = audio.shape[0]

    # force input duration to last as audio
    matscore[:, 1:3] -= np.min(matscore[:, 1:3])
    audio_duration = (stop - start) / sr
    mat_duration = np.max(matscore[:, 1:3])
    matscore[:, 1:3] *= (audio_duration / mat_duration)

    # creating one track per each different program
    programs = np.unique(matscore[:, 4])
    tracks = {}
    is_drum = False
    for program in programs:
        if program == 128:
            program = 0
            is_drum = True
        tracks[program] = pretty_midi.Instrument(program=int(program),
                                                 is_drum=is_drum)

    # creating pretty_midi.PrettyMIDI object and inserting notes
    # also inserting the program number!
    midi = pretty_midi.PrettyMIDI(resolution=1200, initial_tempo=60)
    midi.instruments = tracks.values()
    for row in matscore:
        program = row[4]
        if program == 128:
            program = 0
        tracks[program].notes.append(
            pretty_midi.Note(100, int(row[0]), float(row[1]), float(row[2])))

    # aligning midi to audio
    hopsize = 2**floor(log2(sr * res))
    try:
        path = audio_to_midi_alignment(midi,
                                       audio,
                                       sr,
                                       hopsize,
                                       4 * hopsize,
                                       merge_dlnco,
                                       fastdtw=fastdtw)
    except Exception as e:
        print(e)
        return None

    # interpolating
    new_ons = np.interp(matscore[:, 1], path[:, 0], path[:, 1])
    new_offs = np.interp(matscore[:, 2], path[:, 0], path[:, 1])
    new_ons += (start / sr)
    new_offs += (start / sr)

    return new_ons, new_offs


if __name__ == "__main__":
    check_executables()
    download_soundfont()
