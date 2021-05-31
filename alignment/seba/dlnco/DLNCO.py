import numpy as np
import librosa
from librosa import display
# import matplotlib.pyplot as plt
import math


def dlnco(y, sr, n_fft, hopsize, Q=25.0):
    """Calculate DLNCO feature
    Parameters
    ----------
    y   : audio input waveform
    sr  : sampling rate
    n_fft   : fft size
    Q   : semitone subband filter Q value
    Returns
    -------
    dlnco_feature     : DLNCO feature [shape=(12, frames)]
    """
    D = librosa.stft(y, n_fft=n_fft, hop_length=hopsize)
    D, _ = librosa.magphase(D, power=1)

    center_freqs = librosa.core.midi_to_hz(np.arange(21, 109)) # semitone subbands center frequencies
    filter_bandwidths = center_freqs / float(Q)

    hann = np.hanning(5)
    mat_onset = np.zeros((88, D.shape[1]-1))
    for ii, (cur_center_freq, cur_bw) in enumerate(zip(center_freqs, filter_bandwidths)):
        bin_upper_band = math.ceil((cur_center_freq + cur_bw)*n_fft/sr)
        bin_lower_band = math.floor((cur_center_freq - cur_bw)*n_fft/sr)
        energy = np.sum(D[bin_lower_band:bin_upper_band, :], axis=0)
        energy = np.convolve(energy, hann, mode='same')
        diff_energy = np.diff(energy)
        mat_onset[ii, :] = diff_energy

    # half_wave rectified
    mat_onset[mat_onset<0] = 0

    # plt.imshow(mat_onset, origin='lower')
    # plt.show()

    mat_CO = np.zeros((12, mat_onset.shape[1])) # chroma onset feature
    mat_onset = np.log(mat_onset+0.000000000001)
    for ii in range(0, 12):
        idx = np.arange(ii, 88, 12)
        mat_CO[ii, :] = np.sum(mat_onset[idx, :], axis=0)

    # roll the CO, the first row is now A, should be C
    mat_CO = np.roll(mat_CO, -3, axis=0)
    mat_CO -= np.min(mat_CO)

    # plt.imshow(mat_CO, origin='lower')
    # plt.show()

    # calculate the time dimension norm
    norm_mat_CO = np.linalg.norm(mat_CO, axis=0)
    
    win = int(round(sr/hopsize)) # 1 second window
    max_norm_mat_CO = np.zeros((len(norm_mat_CO))) # maximum value within the norm
    for ii in range(len(norm_mat_CO)):
        lower_b = ii-win if ii-win >=0 else 0
        upper_b = ii+win if ii+win <= len(norm_mat_CO) else len(norm_mat_CO)
        norm_mat_CO_win = norm_mat_CO[lower_b:upper_b]
        max_norm_mat_CO[ii] = np.max(norm_mat_CO_win)
    max_norm_mat_CO /= np.max(max_norm_mat_CO)

    # plt.plot(norm_mat_CO)
    # plt.plot(max_norm_mat_CO)
    # plt.show()

    lnco = mat_CO / max_norm_mat_CO

    # f, axarr = plt.subplots(2, sharex=True)
    # axarr[0].imshow(mat_CO, origin='lower')
    # axarr[1].imshow(lnco, origin='lower')
    # plt.show()

    dlnco_feature = np.zeros((12, D.shape[1]+9))
    for ii in range(10):
        temp_dlnco = np.zeros((12, D.shape[1]+9))
        temp_dlnco[:, ii:D.shape[1]-1+ii] = lnco * np.sqrt(1.0-ii/10.0)
        dlnco_feature += temp_dlnco
    dlnco_feature = dlnco_feature[:, :D.shape[1]]
    dlnco_feature /= np.max(dlnco_feature)

    # f, axarr = plt.subplots(3, sharex=False)
    # axarr[0].imshow(lnco, origin='lower')
    # axarr[1].imshow(dlnco, origin='lower')
    # plt.show()

    # # Get the FFT bins, not counting the DC component
    # frequencies = np.linspace(0, sr, n_fft, endpoint=False)[1:]

    return dlnco_feature

if __name__ == "__main__":
    y, sr = librosa.load("../examples/KissTheRain_2_s_short.wav")
    D = dlnco(y, sr, n_fft=4096)
    display.specshow(librosa.amplitude_to_db(np.abs(D), ref=np.max), y_axis='log', x_axis='time')
    plt.title('Power spectrogram')
    plt.colorbar(format='%+2.0f dB')
    plt.tight_layout()
    plt.show()
