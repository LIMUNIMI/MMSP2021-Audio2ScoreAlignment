import resource
import time
from multiprocessing import Pipe, Process
from multiprocessing.dummy import Pool as tPool
from typing import Callable, Optional

import essentia.standard as es
import numpy as np
import plotly.graph_objs as go
import pretty_midi as pm
import psutil
from fastcluster import linkage
from scipy.cluster.hierarchy import fcluster
from scipy.spatial.distance import pdist


def mat2mir_eval(mat):
    # sorting according to pitches and then onsets
    mat = mat[np.lexsort((mat[:, 1], mat[:, 0]))]
    times = mat[:, (1, 2)]
    pitches = midi_pitch_to_f0(mat[:, 0])
    vel = mat[:, 3]
    return times, pitches, vel


def _monitor_process(call, call_args, call_kwargs):
    ret = call(call_args, call_kwargs)
    ram = resource.getrusage(resource.RUSAGE_THREAD).ru_maxrss
    ttt = resource.getrusage(resource.RUSAGE_THREAD).ru_wtime
    return ret, ram, ttt


def resource_monitor(call: Callable,
                     rss: Optional[int] = None,
                     wtime: Optional[float] = None,
                     call_args: tuple = tuple(),
                     call_kwargs: dict = dict()):
    """
    Same as `parallel_monitor` but the current process is used and not stopped.
    """
    t = tPool(1)
    res = t.apply(_monitor_process, (call, call_args, call_kwargs))
    try:
        ret, ram, ttt = res.get(wtime)
    except TimeoutError:
        return None

    if ram > rss or ttt > wtime:
        return None
    return ret


def kill_procs(procs, timeout=None, on_terminate=None):
    for p in procs:
        try:
            p.kill()
        except psutil.NoSuchProcess:
            pass
    gone, alive = psutil.wait_procs(procs, timeout=None, callback=on_terminate)


def parallel_monitor(call: Callable,
                     rss: Optional[int] = None,
                     wtime: Optional[float] = None,
                     call_args: tuple = tuple(),
                     call_kwargs: dict = dict()):
    """ Runs `call` and monitors
    it; if the used RSS memory (RAM) exceeds `rss` or the execution
    time exceeds `wtime` (whichever occurs first), it kills the process and
    returns None. 

    `rss` should be in bytes
    `wtime` should be in seconds

    To monitor the callable, an external process is spawned and stopped when it
    reaches the maximum resources
    """

    print("Monitoring new process!")

    def process(fn, fn_args, fn_kw_args, pipe):
        # call the function and send the returned value
        pipe.send(fn(*fn_args, **fn_kw_args))
        pipe.close()

    # create a connection
    parent_pipe, child_pipe = Pipe()
    # create a process
    p = Process(target=process,
                args=(call, call_args, call_kwargs, child_pipe),
                daemon=False)
    p.start()
    # create psutil objecto to read memory info
    pu_proc = psutil.Process(p.pid)
    start_time = pu_proc.create_time()
    # run asynchronously
    CRASHED = False
    i = 0
    while p.is_alive():
        # check resources
        i += 1
        try:
            children = pu_proc.children(recursive=True)
            if wtime:
                _wtime = time.time() - start_time
                if _wtime > wtime:
                    CRASHED = True
                    kill_procs(children)
                    p.kill()
                    print("Maximum Whole time reached!")
                if i % 60 == 0:
                    print(f"Used whole time (sec): {_wtime:.2f}")
            if rss:
                cum_rss = sum(map(lambda x: x.memory_info().rss, children))
                if cum_rss > rss:
                    CRASHED = True
                    kill_procs(children)
                    p.kill()
                    print("Maximum RSS reached!")
                if i % 60 == 0:
                    print(f"Used RAM (MB): {cum_rss / (2**20):.2f}")
        except (psutil.NoSuchProcess, psutil.ZombieProcess):
            # one of children or parent do not exist anymore
            continue

        # sleep a bit
        time.sleep(0.25)

    if not CRASHED and p.exitcode is not None and parent_pipe.poll():
        # getting return value
        ret = parent_pipe.recv()
    else:
        # returning None
        ret = None
    p.close()
    parent_pipe.close()
    return ret


def plotly_error_bands(fig, opacity=0.3):
    """
    Given a figure with `error_y` set, converts error bars in continuous error
    bands with the specified opacity
    """
    # creating additional traces
    for d in fig.data:
        # take color of this line
        color = d.line.color[1:]
        # converting hex to rgba
        color = 'rgba' + str(
            tuple([int(color[i:i + 2], 16) for i in (0, 2, 4)] + [opacity]))

        # computing the errors
        error_y = d.error_y
        if not error_y:
            continue
        error_y_plus = error_y.array
        error_y_minus = error_y.arrayminus
        if error_y_minus is None:
            error_y_minus = error_y.array
        if error_y_plus is None:
            error_y_plus = error_y.arrayminus

        # adding the traces
        upper_bound = go.Scatter(
            name=d.name + ' UB',
            x=d.x,
            y=d.y + error_y_plus,
            mode='lines',
            # marker=dict(color=color),
            line=dict(width=0),
            showlegend=False)

        lower_bound = go.Scatter(
            name=d.name + ' LB',
            x=d.x,
            y=d.y - error_y_minus,
            mode='lines',
            # marker=dict(color=color),
            line=dict(width=0),
            fillcolor=color,
            fill='tonexty',
            showlegend=False)

        fig.add_trace(upper_bound)
        fig.add_trace(lower_bound)

        # removing the error bars
        d.error_y = go.scatter.ErrorY()
    return fig


def onsets_clusterize(score, random_generator):
    """
    clusterize onsets in score using a threshold uniformly sampled in 0.02-0.07
    and sets the average onsets of each score 

    This works `in-place` 
    """
    # get random th
    th = random_generator.uniform(0.02, 0.07)
    # compute self-distance
    p = pdist(score[:, 1, None], metric='minkowski', p=1.)
    # clusterize
    Z = linkage(p, method='single')
    clusters = fcluster(Z, t=th, criterion='distance')
    # compute the average onset of each cluster and set it in the score
    M = np.max(clusters)
    for i in range(1, M + 1):
        onsets_idx = np.where(clusters == i)[0]
        avg_onset = np.mean(score[onsets_idx, 1])
        score[onsets_idx, 1] = avg_onset
    return score


def mat_prestretch(mat_in: np.ndarray, mat_out: np.ndarray) -> None:
    """
    Stretches onsets and offsets in `mat_in` so that minimum and maximum times
    correspond to `mat_out`. Works in-place.
    """
    in_times = mat_in[:, 1:3]
    out_times = mat_out[:, 1:3]

    # normalize in [0, 1]
    in_times -= in_times.min()
    in_times /= in_times.max()

    # restretch
    new_start = out_times.min()
    in_times *= (out_times.max() - new_start)
    in_times += new_start


def midi_pitch_to_f0(midi_pitch):
    """
    Return a frequency given a midi pitch
    """
    return 440 * 2**((midi_pitch - 69) / 12)


def find_start_stop(audio, sample_rate=44100, seconds=False, threshold=-60):
    """
    Returns a tuple containing the start and the end of sound in an audio
    array.

    ARGUMENTS:
    `audio` : essentia.array
        an essentia array or numpy array containing the audio
    `sample_rate` : int
        sample rate
    `seconds` : boolean
        if True, results will be expressed in seconds (float)
    `threshold` : int
        a threshold as in `essentia.standard.StartStopSilence`

    RETURNS:
    `start` : int or float
        the sample where sound starts or the corresponding second

    `end` : int or float
        the sample where sound ends or the corresponding second
    """
    # reset parameters based on sample_rate
    ratio = sample_rate / 44100
    fs = round(1024 * ratio)
    hs = round(128 * ratio)
    processer = es.StartStopSilence(threshold=threshold)
    for frame in es.FrameGenerator(audio,
                                   frameSize=fs,
                                   hopSize=hs,
                                   startFromZero=True):
        start, stop = processer(frame)

    if seconds:
        start = specframe2sec(start, sample_rate, hs, fs)
        stop = specframe2sec(stop, sample_rate, hs, fs)
    else:
        start = int(specframe2sample(start, hs, fs))
        stop = int(specframe2sample(stop, hs, fs))

    if start == 2 * hs:
        start = 0

    return start, stop


def specframe2sec(frame, sample_rate=44100, hop_size=3072, win_len=4096):
    """
    Takes frame index (int) and returns the corresponding central time (sec)
    """

    return specframe2sample(frame, hop_size, win_len) / sample_rate


def specframe2sample(frame, hop_size=3072, win_len=4096):
    """
    Takes frame index (int) and returns the corresponding central time (sec)
    """

    return frame * hop_size + win_len / 2


def mat2midi(mat, path):
    """
    Writes a midi file from a mat like asmd:

    pitch, start (sec), end (sec), velocity

    If `mat` is empty, just do nothing.

    Also returns the pretty_midi object

    Only one instrument is used
    """
    if len(mat) > 0:
        # creating pretty_midi.PrettyMIDI object and inserting notes
        midi = pm.PrettyMIDI(resolution=1200, initial_tempo=60)
        midi.instruments = [pm.Instrument(0)]
        for row in mat:
            velocity = int(row[3])
            if velocity < 0:
                velocity = 80
            midi.instruments[0].notes.append(
                pm.Note(velocity, int(row[0]), float(row[1]), float(row[2])))

        # writing to file
        if path:
            midi.write(path)
        return midi
    else:
        return None


def midi2mat(midi):
    """

    Open a midi file (or use a pretty_midi object if provided)  with one
    instrument track and construct a mat like asmd:

    pitch, start (sec), end (sec), velocity
    """

    out = []
    if type(midi) is not pm.PrettyMIDI:
        midi = pm.PrettyMIDI(midi_file=midi)
    for instrument in midi.instruments:
        for note in instrument.notes:
            out.append([note.pitch, note.start, note.end, note.velocity])

    # sort by onset, pitch and offset
    out = np.array(out)
    ind = np.lexsort([out[:, 2], out[:, 0], out[:, 1]])

    return out[ind]


def make_pianoroll(mat,
                   res=0.25,
                   velocities=True,
                   only_onsets=False,
                   only_offsets=False,
                   basis=1,
                   attack=1,
                   basis_l=1,
                   eps=1e-15,
                   eps_range=0):
    """
    return a pianoroll starting from a mat score from asmd with shape (128, N)

    if velocities are available, it will be filled with velocity values; to
    turn this off use `velocities=False`

    if `only_onsets` is true, onle the attack is used and the other part of the
    notes are discarded (useful for aligning with amt). Similarly
    `only_offsets`

    `basis` is the number of basis for the nmf; `attack` is the attack
    duration, all other basis will be long `basis_l` column except the last one
    that will last till the end if needed

    `eps_range` defines how to much is note is enlarged before onset and after
    offset in seconds, while `eps` defines the value to use for enlargement
    """

    L = int(np.max(mat[:, 2]) / res) + 1

    pr = np.zeros((128, basis, L))

    eps_range = int(eps_range / res)

    for i in range(mat.shape[0]):
        note = mat[i]
        pitch = int(note[0])
        vel = int(note[3])
        start = int(np.round(note[1] / res))
        end = min(L - 1, int(np.round(note[2] / res)) + 1)
        if velocities:
            vel = max(1, vel)
        else:
            vel = 1

        if only_offsets:
            pr[pitch, basis - 1, end - 1] = vel
            continue

        # the attack basis
        pr[pitch, 0, start:start + attack] = vel

        # the eps_range before onset
        start_eps = max(0, start - eps_range)
        pr[pitch, 0, start_eps:start] = eps

        start += attack

        # all the other basis
        END = False
        for b in range(1, basis):
            for k in range(basis_l):
                t = start + b * basis_l + k
                if t < end:
                    pr[pitch, b, t] = vel
                else:
                    END = True
                    break
            if END:
                break

        # the ending part
        if not only_onsets:
            if start + (basis - 1) * basis_l < end:
                pr[pitch, basis - 1, start + (basis - 1) * basis_l:end] = vel
                # the eps_range after the offset
                end_eps = min(L, end + eps_range)
                pr[pitch, basis - 1, end:end_eps] = eps

    # collapse pitch and basis dimension
    pr = pr.reshape((128 * basis, -1), order='C')
    return pr


def stretch_pianoroll(pr, out_length):
    """
    Stretch a pianoroll along the second dimension.
    """
    ratio = pr.shape[1] / out_length
    return np.array(
        list(
            map(lambda i: pr[:, min(round(i * ratio), pr.shape[1] - 1)],
                range(out_length)))).T
