import numpy as np


def ms2smp(ms, fs):
    """
    Parameters
    ----------
    ms: float
        Time in milliseconds
    fs: float
        Sampling rate in Hz.
    """
    # return corresponding length in samples

    return (ms*fs)/1000


def compute_stride(grain_len_samp, grain_over):
    return int(grain_len_samp - int(grain_len_samp * grain_over / 2) - 1)


def win_taper(grain_len_samp, grain_over, data_type=np.int16):

    edge_over = int(grain_len_samp * grain_over / 2)
    r = np.arange(0, edge_over) / float(edge_over)
    win = np.concatenate((r,
        np.ones(int(grain_len_samp)-2*edge_over),
        r[::-1]))
    max_val = np.iinfo(data_type).max

    return (win*max_val).astype(data_type)


def build_linear_interp_table(n_samples, down_fact, data_type=np.int16):

    samp_vals = []
    amp_vals = []
    for n in range(int(n_samples)):
        # compute t, N, and a
        t = n*down_fact
        N = np.floor(t)
        a = 1 - (t - N)
        samp_vals.append(N)
        amp_vals.append(a)

    MAX_VAL = np.iinfo(data_type).max
    amp_vals =  np.array(amp_vals)
    amp_vals = (amp_vals*MAX_VAL).astype(data_type)

    return samp_vals, amp_vals

def dft_rescale(x, f):
    X = np.fft.fft(x)
    # separate even and odd lengths
    parity = (len(X) % 2 == 0)
    N_samples = len(X) / 2 + 1 if parity else (len(X) + 1) / 2
    Y = np.zeros(int(N_samples), dtype=np.complex)
    # work only in the first half of the DFT vector since input is real
    for n in range(int(N_samples)):
        # accumulate original frequency bins into rescaled bins
        ix = int(n * f)
        if ix < N_samples:
            Y[ix] += X[n]
    # now rebuild a Hermitian-symmetric DFT
    Y = np.r_[Y, np.conj(Y[-2:0:-1])] if parity else np.r_[Y, np.conj(Y[-1:0:-1])]
    return np.real(np.fft.ifft(Y))