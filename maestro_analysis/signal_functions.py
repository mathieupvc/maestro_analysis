import numpy as np
from scipy import signal
from scipy.signal import find_peaks


def OG_envelope(data, fs, freqbounds=(100, 5000), numbands=16):
    """
    Make an envelope the proper way.
    Code by Keith Doelling
    """
    # define band limits
    bands = np.logspace(*np.log10(freqbounds), numbands + 1)
    envelope = np.zeros_like(data).astype(float)
    for lfreq, hfreq in zip(bands[:-1], bands[1:]):
        # make a filter for bandpass
        taps = signal.firwin(
            512, [lfreq * 2 / fs, hfreq * 2 / fs], pass_zero="bandpass"
        )
        # filtered = signal.lfilter(taps, 1, data)
        filtered = signal.filtfilt(taps, 1, data)
        envelope += np.abs(signal.hilbert(filtered))
    return envelope


def butter_lowpass_filter(data, cutoff, fs, order=5):
    b, a = signal.butter(order, cutoff, btype='low', analog=False, output='ba', fs=fs)
    # y = signal.lfilter(b, a, data)
    y = signal.filtfilt(b, a, data)
    return y


def subsample(x, fs_in, fs_out):
    '''To use with correct values'''

    step = fs_in / fs_out
    y = np.zeros((int(x.shape[0] / step), x.shape[1]))
    for feature in range(x.shape[1]):
        y[0, feature] = x[0, feature]  # set the first sample (find_peaks does not detect the first sample as a peak)
        peaks = find_peaks(x[:, feature])
        for peak_ind in peaks[0]:
            new_ind = round(peak_ind / step)
            y[new_ind, feature] = x[peak_ind, feature]

    return y


def xcorr(x, y, scale='unbiased', center=True):
    '''cross-correlation in Matlab's way.'''
    # Remove means
    if center:
        x = x - np.mean(x)
        y = y - np.mean(y)
    std_x = np.std(x)
    std_y = np.std(y)

    # Pad shorter array if signals are different lengths
    if x.size > y.size:
        pad_amount = x.size - y.size
        y = np.append(y, np.repeat(0, pad_amount))
    elif y.size > x.size:
        pad_amount = y.size - x.size
        x = np.append(x, np.repeat(0, pad_amount))

    corr = np.correlate(x, y, mode='full')  # scale = 'none'
    lags = np.arange(-(x.size - 1), x.size)

    if scale == 'biased':
        corr = corr / x.size * std_x * std_y
    elif scale == 'unbiased':
        corr /= (x.size - abs(lags)) * std_x * std_y
    elif scale == 'normalized':
        corr /= np.sqrt(np.dot(x, x) * np.dot(y, y))

    return corr, lags


def blp_via_hilbertTAL(data, freqs, Fs):
    """
    Compute band-limited power (BLP) of a specific electrode using a Hilbert transform.

    Parameters:
    data : numpy array
        Row vector of the data.
    freqs : list or numpy array
        Vector defining band boundaries (e.g. 50:10:150).
    Fs : float
        Sampling rate.

    Returns:
    BLP : numpy array
        Band-limited power normalized as a percent of the mean signal.
    """

    # Ensure data is a row vector
    if data.ndim > 1 and data.shape[0] > 1:
        raise ValueError('Data needs to be a row vector.')

    n_band = len(freqs) - 1
    n_tpt = data.shape[-1]
    BLP = np.zeros((n_band, n_tpt))

    for j in range(n_band):
        lowF = freqs[j]
        highF = freqs[j + 1]

        # if lowF != 0 and highF != 0:
        # Design the butterworth bandpass filter
        B, A = signal.butter(4, [lowF, highF], btype='bandpass', fs=Fs)
        filtered_data = signal.filtfilt(B, A, data)
        BLP[j, :] = np.abs(signal.hilbert(filtered_data))
        # else:
        #     # A case with special filters (if needed, implement eegfilt or similar)
        #     BLP[j, :] = np.abs(signal.hilbert(data))  # Assuming eegfilt equivalent is identity for now

    # Normalize the power by dividing by the mean within each frequency band
    BLP = BLP / BLP.mean(axis=1, keepdims=True)

    # Average across frequency bands and multiply by 100 to express as percent of mean
    BLP = BLP.mean(axis=0) * 100

    return BLP