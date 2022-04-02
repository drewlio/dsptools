"""Signal plotting functions

Leverages Matplotlib axis objects.

Functions grab the current Matplotlib axis or take the axis as a parameter. If
no axis is available, one is created. 

Author: Drew Wilson
"""


import warnings
import numpy as np
import scipy.signal as signal
import scipy.fft as fft
import matplotlib.pyplot as plt


def winfft(vec, 
           fs=1, 
           full_scale=1, 
           beta=12, 
           n=None):
    """
	Apply Kaiser window and FFT


    Parameters
    ----------

    vec : 1d list-like object
        Input time domain signal

    fs : float, optional
        Sampling rate in Hz, default=1

    full_scale : float, optional
        Normalization for full scale, default=1 (0 dB)

    beta : float, optional
        Kaiser beta, default=12

    n : integer, optional
        Number of samples to FFT, will truncate or pad, and only apply window
		function to [truncated] data and not padding.


    Returns
    -------

    freqs: ndarray, float
        Frequency axis

    mags : ndarray, complex float
        scaled frequency output values

    """


    # cast input as ndarray (in case it's not already)
    vec = np.array(vec)

    # Truncate the input, if necessary
    if n and (n < len(vec)):
        vec = vec[:n]

    # Window the remaining data
    win = signal.windows.kaiser(len(vec), beta)
    winscale = np.sum(win)
    win_data = win * vec

	# Pad the windowed input, if necessary
    if n and (n > len(vec)):
        vec = np.concatenate((vec, np.zeros(n - len(vec))))

    # Perform the FFT and scale for window gain
    mags = 1/(winscale*full_scale) * fft.fft(win_data, n)

    # fftshift
    mags = fft.fftshift(mags)

    # create frequency axis
    freqs = np.array(range(len(vec))) * fs/len(vec) - fs / 2

    return freqs, mags


def plot_spectrum(freqs, 
                  amplitudes, 
                  *args,
                  **kwargs):
    """
    Plots frequency spectrum in dB from lists of frequencies and amplitudes in
    linear units.


    Parameters
    ----------

    freqs: ndarray, float
        Frequencies 

    amplitudes: ndarray, complex float
        Amplitudes in linear units (not dB)

    dynamic_range: float, optional
        dynamic range to scale vertical axis, default = 150dB

    style : {'ggplot', `default`, **matplotlib.style.available}

    divs : integer
        Number of divisions of the x-axis, default = 10

    *args : optional Matplotlib arguments

    **kwargs : optional Matplotlib keyword arguments


    Returns
    -------

    None
    """
    # This method of setting default values is necessary because of how
    # Matplotlib allows Line2D options to plt.plot() in either positional or
    # keyword form.
    # Keys for this script are used (and then removed) from kwargs and the
    # remaining items in kwargs are passed on to plt.plot().
    if 'dynamic_range' in kwargs.keys():
        dynamic_range = kwargs['dynamic_range']
        kwargs.pop('dynamic_range')
    else:
        dynamic_range = 150

    if 'style' in kwargs.keys():
        style = kwargs['style']
        kwargs.pop('divs')
    else:
        style = 'ggplot'

    if 'divs' in kwargs.keys():
        divs = kwargs['divs']
        kwargs.pop('divs')
    else:
        divs = None


    # Apply the plot style
    plt.style.use(style)


    # Scale the freqs to Hz, kHz, MHz, GHz
    if (freqs[-1] >= 1e3) and (freqs[-1] < 1e6):
        freqs = freqs/1e3
        plt.xlabel('Frequency [kHz]')
    elif (freqs[-1] >= 1e6) and (freqs[-1] < 1e9):
        freqs = freqs/1e6
        plt.xlabel('Frequency [MHz]')
    elif (freqs[-1] >= 1e9):
        freqs = freqs/1e9
        plt.xlabel('Frequency [GHz]')
    else:
        plt.xlabel('Frequency [Hz]')


    #Estimate the n+1 frequency for setting the plot limits to round values
    freq_max = np.round(freqs[-1]+(freqs[-1]-freqs[-2]), decimals=3)


    #Set the x axis limits to `divs` number of divisions
    if divs is not None:
        ax = plt.gca()
        ax.set_xticks(np.linspace(freqs[0], freq_max, divs+1))


    # Generate the plot and set axis limits and labels
    plt.plot(freqs, 20 * _safelog(np.abs(amplitudes)), *args, **kwargs)
    plt.axis([freqs[0], freq_max, -dynamic_range, 0])
    plt.ylabel('Magnitude [dBFS]')


# avoids log of 0 errors
def _safelog(x, minval=1e-10): return np.log10(x.clip(min=minval))

