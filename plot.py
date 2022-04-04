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

    Uses the current axis of the current Matplotlib.Pyplot figure. 
    Axes are not cleared to allow multiple traces on the same axis.


    Parameters
    ----------

    freqs: ndarray, float
        Frequencies 

    amplitudes: ndarray, complex float
        Amplitudes in linear units (not dB)

    dynamic_range: float, optional
        dynamic range to scale vertical axis, default = 150dB

    style : {'ggplot', `default`, **matplotlib.style.available}

    xdivs : integer
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
        kwargs.pop('xdivs')
    else:
        style = 'ggplot'

    if 'xdivs' in kwargs.keys():
        xdivs = kwargs['xdivs']
        kwargs.pop('xdivs')
    else:
        xdivs = None


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


    #Set the x axis limits to `xdivs` number of divisions
    if xdivs is not None:
        ax = plt.gca()
        ax.set_xticks(np.linspace(freqs[0], freq_max, xdivs+1))


    # Generate the plot and set axis limits and labels
    plt.plot(freqs, 20 * _safelog(np.abs(amplitudes)), *args, **kwargs)
    plt.axis([freqs[0], freq_max, -dynamic_range, 0])
    plt.ylabel('Magnitude [dBFS]')

    plt.title('Frequency Domain')


def plot_filter_response(w, 
                         h, 
                         fs=None,
                         *args,
                         **kwargs):
    """
    Plots frequency spectrum in dB and angle in radians, over frequency.

    Uses the current Matplotlib.Pyplot figure and reuses the first two
    axes (creating them if they don't exist). Axes are cleared on each call.


    Parameters
    ----------

    w: ndarray, float
        Normalized angular frequencies, as would be returned by signal.freqz()

    h: ndarray, complex float
        Complex amplitude

    fs: float, optional
        Sample rate. If this is provided, frequency axis will be scaled Hz
        instead of angular frequencies. Default = None.

    style : {'ggplot', `default`, **matplotlib.style.available}

    xdivs : integer
        Number of divisions of the x-axis. Default = 10

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
    if 'style' in kwargs.keys():
        style = kwargs['style']
        kwargs.pop('xdivs')
    else:
        style = 'ggplot'

    if 'xdivs' in kwargs.keys():
        xdivs = kwargs['xdivs']
        kwargs.pop('xdivs')
    else:
        xdivs = None


    # Apply the plot style
    plt.style.use(style)


    # get handles to the current figure
    # check to see if axes exist and create them if not
    fig = plt.gcf()
    try:
        ax1 = fig.axes[0]
        ax2 = fig.axes[1]
    except:
        ax1 = plt.gca()
        ax2 = ax1.twinx()


    # clear the axes (which also clears `xlabel`, `ylabel`)
    ax1.clear()
    ax2.clear()


    # if `fs` is provided, scale frequencies
    # 
    # Note `plt.xlabel()` should work but there are documented old bugs with
    # `plt.twinx()` subplots and using `plt.xlabel()`, however, using 
    # `Axes.set_xlabel()` works.
    if fs is not None:
        freqs = w / (2*np.pi) * fs    # Hz
    
        # Scale the freqs to Hz, kHz, MHz, GHz
        if (freqs[-1] >= 1e3) and (freqs[-1] < 1e6):
            freqs = freqs/1e3
            ax1.set_xlabel('Frequency [kHz]')
        elif (freqs[-1] >= 1e6) and (freqs[-1] < 1e9):
            freqs = freqs/1e6
            ax1.set_xlabel('Frequency [MHz]')
        elif (freqs[-1] >= 1e9):
            freqs = freqs/1e9
            ax1.set_xlabel('Frequency [GHz]')
        else:
            ax1.set_xlabel('Frequency [Hz]')

        #Estimate the n+1 frequency for setting the plot limits to round values
        freq_max = np.round(freqs[-1]+(freqs[-1]-freqs[-2]), decimals=3)
    else:
        freqs = w
        freq_max = freqs[-1]
        ax1.set_xlabel('Normalized Angular Frequency [radians/sample]')
    

    #Set the x axis limits to `xdivs` number of divisions
    if xdivs is not None:
        ax = plt.gca()
        ax.set_xticks(np.linspace(freqs[0], freq_max, xdivs+1))


    # Generate the plot and set axis limits and labels
    plt.sca(ax1)
    plt.plot(freqs, 20 * _safelog(np.abs(h)), color='b', *args, **kwargs)
    plt.ylabel('Magnitude [dBFS]', color='b')

    angles = np.unwrap(np.angle(h))
    plt.sca(ax2)
    plt.plot(freqs, angles, 'g')
    plt.ylabel('Angle [radians]', color='g')
    
    plt.xlim([freqs[0], freq_max])
    plt.title('Digital Filter Frequency Response')


def plot_timeseries(amplitudes, 
                    fs=None, 
                    *args,
                    **kwargs):
    """
    Plots time series in absolute units from lists of time offsets and 
    amplitudes in linear units.

    Uses the current axis of the current Matplotlib.Pyplot figure. 
    Axes are not cleared to allow multiple traces on the same axis.


    Parameters
    ----------

    amplitudes: ndarray, complex float
        Amplitudes in linear units (not dB)

    fs: float, optional
        Sample rate in Hz. If this is provided, x-axis is labeled in 
        nearest units (ps, ns, us, ms, s), otherwise x-axis is labeled
        in samples.

    style : {'ggplot', `default`, **matplotlib.style.available}

    xdivs : integer
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
    if 'style' in kwargs.keys():
        style = kwargs['style']
        kwargs.pop('xdivs')
    else:
        style = 'ggplot'

    if 'xdivs' in kwargs.keys():
        xdivs = kwargs['xdivs']
        kwargs.pop('xdivs')
    else:
        xdivs = None


    # Apply the plot style
    plt.style.use(style)


    # Scale the time steps to 
    #   ps - picoseconds
    #   ns - nanoseconds
    #   us - microseconds 
    #   ms - milliseconds
    #   s  - seconds
    if fs is None:
        plt.xlabel('Time [samples]')
        period = 1
    else:
        period = 1/fs
        if (period < 1) and (period >= 1e-3):
            period = period/1e-3
            plt.xlabel('Time [ms]')
        elif (period < 1e-3) and (period >= 1e-6):
            period = period/1e-6
            plt.xlabel('Time [μs]')
        elif (period < 1e-6) and (period >= 0.1e-9):
            period = period/1e-9
            plt.xlabel('Time [ns]')
        elif (period < 0.1e-9):
            period = period/1e-12
            plt.xlabel('Time [ps]')
        else:
            plt.xlabel('Time [s]')


    # create the time steps
    times = np.arange(0, period*len(amplitudes), period)


    #Estimate the n+1 time step for setting the plot limits to round values
    time_max = np.round(times[-1]+(times[-1]-times[-2]), decimals=3)


    #Set the x axis limits to `xdivs` number of divisions
    if xdivs is not None:
        ax = plt.gca()
        ax.set_xticks(np.linspace(times[0], time_max, xdivs+1))


    # Generate the plot and set axis limits and labels
    plt.plot(times, amplitudes, *args, **kwargs)
    plt.ylabel('Amplitude [linear]')

    plt.title('Time Domain Waveform')


# avoids log of 0 errors
def _safelog(x, minval=1e-10): return np.log10(x.clip(min=minval))

