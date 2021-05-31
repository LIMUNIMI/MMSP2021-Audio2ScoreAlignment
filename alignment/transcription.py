"""
A module to perform transcription using different models

"""

import logging
import os
import uuid
from pathlib import Path
from typing import Optional, Tuple, Union

import numpy as np
from essentia.standard import AudioWriter, Resample

from . import settings as s
from . import utils

# suppress tensorflow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # FATAL
logging.getLogger('tensorflow').setLevel(logging.FATAL)

# Define model and load checkpoint
# Only needs to be run once.
CHECKPOINT_DIR = './alignment/magenta/train/'


def google_sucks(samples, sr):
    """
    Google sucks and want to use audio path (raw wav) instead of decoded
    samples loosing in decoupling between file format and DSP.
    This hack overwrites their stupid loader which writed data to a temprorary
    file and reopen it
    """

    return samples


def magenta(audio: np.ndarray, sr: int, cuda=False):
    """
    input audio and sample rate, output mat like asmd with (pitch, ons, offs,
    velocity)
    """
    import tensorflow.compat.v1 as tf  # noqa: autoimport
    tf.disable_v2_behavior() 
    from magenta.models.onsets_frames_transcription import \
        train_util  # noqa: autoimport
    from magenta.models.onsets_frames_transcription import (  # noqa: autoimport
        audio_label_data_utils, configs, data, infer_util)
    from magenta.music import audio_io  # noqa: autoimport
    from magenta.music.protobuf import music_pb2  # noqa: autoimport

    # simple hack because google sucks... in this way we can accept audio data
    # already loaded and keep our reasonable interface (and decouple i/o
    # from processing)
    original_google_sucks = audio_io.wav_data_to_samples
    audio_io.wav_data_to_samples = google_sucks
    audio = np.array(audio)
    config = configs.CONFIG_MAP['onsets_frames']
    hparams = config.hparams
    hparams.use_cudnn = cuda
    hparams.batch_size = 1
    examples = tf.placeholder(tf.string, [None])

    dataset = data.provide_batch(examples=examples,
                                 preprocess_examples=True,
                                 params=hparams,
                                 is_training=False,
                                 shuffle_examples=False,
                                 skip_n_initial_records=0)

    estimator = train_util.create_estimator(config.model_fn, CHECKPOINT_DIR,
                                            hparams)

    iterator = dataset.make_initializable_iterator()
    next_record = iterator.get_next()

    example_list = list(
        audio_label_data_utils.process_record(wav_data=audio,
                                              sample_rate=sr,
                                              ns=music_pb2.NoteSequence(),
                                              example_id="fakeid",
                                              min_length=0,
                                              max_length=-1,
                                              allow_empty_notesequence=True,
                                              load_audio_with_librosa=False))
    assert len(example_list) == 1
    to_process = [example_list[0].SerializeToString()]

    sess = tf.Session()

    sess.run([
        tf.initializers.global_variables(),
        tf.initializers.local_variables()
    ])

    sess.run(iterator.initializer, {examples: to_process})

    def transcription_data(params):
        del params
        return tf.data.Dataset.from_tensors(sess.run(next_record))

    # put back the original function (it still writes and reload... stupid
    # though
    audio_io.wav_data_to_samples = original_google_sucks
    input_fn = infer_util.labels_to_features_wrapper(transcription_data)

    prediction_list = list(
        estimator.predict(input_fn, yield_single_examples=False))

    assert len(prediction_list) == 1

    notes = music_pb2.NoteSequence.FromString(
        prediction_list[0]['sequence_predictions'][0]).notes

    out = np.empty((len(notes), 4))
    for i, note in enumerate(notes):
        out[i] = [note.pitch, note.start_time, note.end_time, note.velocity]
    return out


def omnizart(audio: Union[Path, np.ndarray],
             sr: Optional[int] = None,
             mode: str = 'Stream') -> Union[Tuple[np.ndarray, np.ndarray]]:
    """
    Transcribes audio and returns a mat like ASMD.
    Also returns pedaling information.

    Parameters
    ----------

    `audio` : np.ndarray or Path-like object
        the mono input audio
    `sr` : int or None
        the sample rate for `audio` if it is an `np.ndarray`
    `mode` : str
        mode of transcription (e.g. piano, multi-instrument)
        one of 'Piano', 'Pop', 'Stream'
        default: 'Stream'

    Returns
    -------

    np.ndarray:
        2d array with columns representing in order: pitch, onset, offset,
        velocity
    """
    from omnizart.music.app import MusicTranscription  # noqa: autoimport
    if type(audio) is np.ndarray:
        assert sr is not None, "Cannot process array without sample rate!"
        # write to temporary file
        path1 = str(uuid.uuid1(clock_seq=os.getpid())) + '.wav'
        if audio.ndim == 1:  # type:ignore
            audio = np.stack([audio, audio], axis=1)  # type:ignore
        AudioWriter(filename=path1, format='wav',
                    sampleRate=sr)(audio.astype(np.float32))  # type: ignore
        audio = path1
    try:
        out_midi = MusicTranscription().transcribe(audio,
                                                   model_path=mode,
                                                   output=None)
    except Exception as e:
        print(e)
        raise RuntimeError("Error while transcribing")
    finally:
        if os.path.exists(audio):
            os.remove(audio)
    return utils.midi2mat(out_midi)


def bytedance(
        audio: np.ndarray,
        sr: int,
        return_pedal: bool = False,
        device=s.DEVICE) -> Union[Tuple[np.ndarray, np.ndarray], np.ndarray]:
    """
    Transcribes audio and returns a mat like ASMD.
    Also returns pedaling information.

    Parameters
    ----------

    `audio` : np.ndarray
        the mono input audio
    `sr` : int
        the sample rate for `audio`
    `device` : str
        'cuda' or 'cpu' as in pytorch

    Returns
    -------

    np.ndarray:
        2d array with columns representing in order: pitch, onset, offset,
        velocity
    np.ndarray:
        2d array with columns representing in order: pedling onset, pedaling
        offset. Only sustain pdaling is inferred
        the returned arrays are not sorted!
    """
    import piano_transcription_inference as pti  # noqa: autoimport

    if sr != pti.sample_rate:
        # resample
        audio = Resample(inputSampleRate=sr,
                         outputSampleRate=pti.sample_rate,
                         quality=1)(audio)
    try:
        transcriptor = pti.PianoTranscription(device=device)
        transcribed_dict = transcriptor.transcribe(audio, '')
    except Exception:
        raise RuntimeError("Error while transcribing")
    mat = [(n['midi_note'], n['onset_time'], n['offset_time'], n['velocity'])
           for n in transcribed_dict['est_note_events']]
    ped = [(n['onset_time'], n['offset_time'])
           for n in transcribed_dict['est_pedal_events']]
    if return_pedal:
        return np.asarray(mat), np.asarray(ped)
    else:
        return np.asarray(mat)


def transcribe(audio: np.ndarray, sr: int, method: str) -> np.ndarray:
    """
    A wrapper around various functions for transcribing audio

    Possible values for `method`:

    * 'bytedance'
    * 'omnizart'
    * 'magenta'

    Returns
    -------

    np.ndarray :
        2d array with columns representing in order: pitch, onset, offset,
        velocity
        the returned array is sorted by onset, pitch and offset (in this order)

    """
    mat = eval(method)(audio, sr)
    # sort by onset, pitch, offset
    mat = mat[np.lexsort([mat[:, 2], mat[:, 0], mat[:, 1]])]
    utils.mat2midi(mat, 'transcribed.mid')
    return mat
