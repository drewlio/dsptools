import warnings
import numpy as np
from scipy import signal


class TrapezoidalPulse:
    """Trapezoidal Pulse
    
    Synthesize a asymmetric trapezoidal pulse with independent control of pulse
    shape parameters and with optional waveform filtering.
    
    Synthesize a asymmetric trapezoidal pulse with independent control of 
     - sample rate
     - pulse width
     - rise time
     - fall time
     - low and a high state levels
     - pulse sample padding
    and with optional waveform filtering.
    
    
    Author: Drew Wilson


    Attributes
    ----------
    waveform : tuple
        Waveform samples. Length in samples is twice the width, as
        measured to `mid-percent_reference_levels`. That is, if 10/90% 
        reference levels are used, the width is measured from the 50% 
        level crossings.
 
    waveform_filtered : tuple
        Same as waveform, but after filter is applied. If filter is
        None or no filter was specified, `waveform` and `waveform_filtered`
        are the same.
 
    fs : float
        Sample rate of the waveform.
 
    width : float
        Width of the pulse as measured the same as parameter width
        above.
 
    risetime : float
        Risetime, as provided in input parameter, irrespective of any
        filtering.
 
    falltime : float
        Falltime, as provided in the input parameter, irrespecive of
        any filtering. 
 
    percent_reference_levels : list
        Reference levels used for width and waveform length
        calculations.
 
    percent_width_reference_level : list
        Reference level used to calculate the width of the pulse,
        usually midway between the percent_reference_levels.
 
    state_levels : list
        Levels of logic low and high. Units are in volts. 
 
    filter : {None, dict}
        Filter parameters applied, or None if no filter applied.
        Selection of parameters depends on method of specifying the filter.
            Filter type None returns filter None.
            Filter type (b, a) returns only the (b, a) dict key.
 
        dict{
            order : order of the synthesized elliptical filter
            passband_ripple : passband ripple (dB)
            stopband_rejection : stopband rejection (-dB)
            normalized_frequency : normalized cutoff frequency (Hz)
            frequency : unnormalized cutoff frequency (Hz)
            coefficients : (b, a)
            
        }


    Methods
    -------
    update : **kwargs
        Reset some or all parameters and recompute outputs.
                
    """
    def __init__(self, 
        fs,
        width=10e-9,
        risetime=2e-9,
        falltime=2e-9,
        percent_reference_levels=[10,90],
        percent_width_reference_level=50,
        state_levels=[0,1],
        padding_factor=2,
        filter=None,
        cutoff=None,
        timeunit=1,
        frequnit=1):
        """
        Parameters
        ----------
        fs : float
            Sample rate

        width : float, optional
            Pulse width in units of seconds, measured to 
            mid-percent_reference_levels. That is, if 10/90% reference 
            levels are used, the width is measured from the 50% level 
            crossings.
            Default is 10ns (Corresponding to 100MHz square wave with 
            50% duty cycle.)

        risetime : float, optional
            Units of seconds. 
            Default is 2ns (Corresponding to 10% of a full period of a
            default `width` at 50% duty cycle)

        falltime : float, optional
            Units of seconds. 
            Default is 2ns (Corresponding to 10% of a full period of a
            default `width` at 50% duty cycle)

        percent_reference_levels : list, optional
            List of rise/fall time reference levels in percent.
            Default is [10, 90].

        percent_width_reference_level : float, optional
            Reference level used to calculate the width of the pulse,
            usually midway between the `percent_reference_levels`.
            
        state_levels : list, optional
            Units of volts. State levels for low and high. 
            Default is [0, 1].

        padding_factor : float, optional
            Amount of padding before/after outputted pulse waveform 
            measured in of multiples of `width`.

            Total output waveform length is 
                len(width) + ( len(width) * padding_factor )

            Default is 1. When output waveforms are concatenated, this
            creates a 50% duty cycle signal, despite having asymmetric rise
            and fall times. The pulse is kept centered in the waveform by
            adding half the excess width to the front and half to the back.

            Use this parameter when your filter bandwidth results in
            extended ringing time. 
    
            For example, padding_factor=4 results in an output
            waveform of length width+4*width. If repeated, this waveform 
            would have 20% duty cycle.
            

        filter : {None, "auto", float, (b, a)}, optional
            Specifies the optional filter to be applied to the waveform
            after straight-line synthesis based off input parameters.
            Options are:

                None (default)
                    No filtering is applied.
                    Output waveform and waveform_filtered are the same.

                "auto"
                    The filter cutoff frequency is determined by the rule
                    of thumb fc = 0.35 / <fastest risetime>.
                    Example: risetime=1e-9, falltime=2e-9 results in a
                    cutoff frequency of 0.35/1e-9=350MHz
                        
                float
                    Cutoff frequency in Hertz
    
                (b, a)
                    Tuple containing pre-designed digital filter 
                    coefficients.
                    
                "cutoff"
                    The text string 'cutoff' indicates the parameter 'cutoff'
                    should be used for the cutoff frequency float value.
                    
                    
        cutoff : float, required for filter=cutoff
            Specifies the filter cutoff frequency if the parameter 'filter'
            is set to the value 'cutoff'. This pattern is to streamline use
            of ipywidgets 'interact'. 
                        
                float
                    Cutoff frequency in Hertz
                    
        timeunit : float, optional
            Specifies a multiplier that should be applied to the time values
            such as 'width', 'risetime', 'falltime'. For example, the value
            of 1e-9 indicates that 'width', 'risetime', and 'falltime' values
            are in the units of nanoseconds.
            
                float
                    Multiplier of time units
                    
        frequnit : float, optional
            Specifies a multiplier that should be applied to the frequency
            values such as 'filter' (when float), 'cutoff'. For example,
            the value of 1e6 indicates that 'filter', 'cutoff' values
            are in the units of megahertz. 
            Note 'fs' is uneffected by 'frequnit'.
            
                float
                    Multiplier of frequency units
            
        """
        self._timeunit = timeunit
        self._frequnit = frequnit
        self.fs = fs
        self._width = width * self._timeunit
        self._risetime = risetime * self._timeunit
        self._falltime = falltime * self._timeunit
        self._percent_reference_levels = percent_reference_levels
        self._percent_width_reference_level = percent_width_reference_level
        self._state_levels = state_levels
        self._padding_factor=padding_factor
        self._filter = filter
        self._cutoff = cutoff if cutoff is None else cutoff * self._frequnit

        # generate a fs multiple, then downsample before returning from the
        # class. Not currently used
        self._fs_internal = self.fs
        
        self._compute()

        
    def update(self, **kwargs):
        """Update some or all of the parameters on an instance and recompute
        """
        for key in kwargs:
            setattr(self, "_"+key, kwargs[key])
        self._compute()
        
        
    def _compute(self):
        """Synthesize the waveform"""
        
        def compute_thresholds(self):
            """Compute threshold levels in absolute units"""
            self._lower_threshold = ((np.max(self._state_levels) - np.min(self._state_levels))
                                     * np.min(self._percent_reference_levels) / 100
                                     + np.min(self._state_levels))

            self._upper_threshold = ((np.max(self._state_levels) - np.min(self._state_levels))
                                     * np.max(self._percent_reference_levels) / 100
                                     + np.min(self._state_levels))

            self._width_threshold = ((np.max(self._state_levels) - np.min(self._state_levels))
                                     * self._percent_width_reference_level / 100
                                     + np.min(self._state_levels))
        
        
        def compute_rise(self):
            """Compute the rising portion of the waveform"""
            # number of samples in rising edge (from ref levels provided, eg 10-90%)
            n = self._risetime * self._fs_internal 
            
            # anything less than 1/fs risetime is "zero risetime"
            if self._risetime < 1/self._fs_internal:
                self.rise = np.array([np.min(self._state_levels), np.max(self._state_levels)])
                self.rise_lower_ref_level_idx = 0
                self.rise_upper_ref_level_idx = 0
                self.rise_width_ref_level_idx = 0
            else:
                # n is number of samples for the, for example, 10-90% rise
                # time. Scale up n by fraction of time within reference levels, eg (90-10)/100
                n /= (np.max(self._percent_reference_levels) 
                      - np.min(self._percent_reference_levels)) / 100
                n = int(np.round(n))
                
                # calculate the ramp for the full 0-100% rise
                self.rise = np.linspace(np.min(self._state_levels), 
                                        np.max(self._state_levels), 
                                        n)

                # calculate indices of reference levels
                self.rise_lower_ref_level_idx = np.argmax(self.rise > self._lower_threshold)
                self.rise_upper_ref_level_idx = np.argmax(self.rise > self._upper_threshold)
                self.rise_width_ref_level_idx = np.argmax(self.rise > self._width_threshold)


        def compute_fall(self):
            """Compute the falling portion of the waveform"""
            # number of samples in falling edge (from ref levels provided, eg
            # 10-90%)
            n = self._falltime * self._fs_internal 
            
            # anything less than 1/fs risetime is "zero falltime"
            if self._falltime < 1/self._fs_internal:
                self.fall = np.array([np.max(self._state_levels), np.min(self._state_levels)])
                self.fall_lower_ref_level_idx = 0
                self.fall_upper_ref_level_idx = 0
                self.fall_width_ref_level_idx = 0
            else: 
                # n is just number of samples for the, for example, 10-90% fall
                # time. Scale up n by fraction of time within reference levels, eg (90-10)/100
                n /= (np.max(self._percent_reference_levels) 
                      - np.min(self._percent_reference_levels)) / 100
                n = int(np.round(n))

                # calculate the ramp for the full 0-100% rise
                self.fall = np.linspace(np.max(self._state_levels), 
                                        np.min(self._state_levels), 
                                        n)
    
                # calculate indices of reference levels
                self.fall_lower_ref_level_idx = np.argmax(self.fall < self._lower_threshold)
                self.fall_upper_ref_level_idx = np.argmax(self.fall < self._upper_threshold)
                self.fall_width_ref_level_idx = np.argmax(self.fall < self._width_threshold)


        def compute_top(self):
            """Compute the middle portion of the waveform, from top of rising
            edge to top of falling edge."""
            # compute required number of samples between rising and falling 
            # percent_width_reference_level crossings and reduce by samples already
            # implemented in the rise and fall
            n = self._width * self._fs_internal \
                - (len(self.rise) - self.rise_width_ref_level_idx) \
                - self.fall_width_ref_level_idx

            if n <= 0:
                warnings.warn(f"Pulse top is zero length because width of {self._width} is too short for given rise and fall times")
                
            self.top = np.array(int(n) * [np.max(self._state_levels)])


        def compute_head(self):
            # must adjust head length so that pulse is centered in waveform.
            n = self._width \
                * self._fs_internal \
                * self._padding_factor \
                / 2 \
                - self.rise_width_ref_level_idx

            self.head = np.array(int(n) * [np.min(self._state_levels)])

            
        def compute_tail(self):
            # must adjust tail length so that pulse is centered in waveform.
            n = self._width \
                * self._fs_internal \
                * self._padding_factor \
                / 2 \
                - (len(self.fall) - self.fall_width_ref_level_idx)

            self.tail = np.array(int(n) * [np.min(self._state_levels)])

            
            
        def assemble(self):
            """Assemble the first, middle, end partial waveforms into one"""
            self.waveform = np.concatenate((self.head,
                                            self.rise,
                                            self.top, 
                                            self.fall,
                                            self.tail))
            
            
        def filter(self):
            """Compute and apply waveform filtering"""
            
            def design_filter(self, 
                              order = 7, 
                              passband_ripple = 0.1, #dB
                              stopband_rejection = 40, #-dB
                              normalized_freq = 0.5):
                """Design the filter given parameters"""
                
                # warn if cutoff is within 20% of Nyquist frequency
                assert (normalized_freq < 0.5 * 0.8), "Sample rate too low for filter requirements."
                b, a = signal.ellip(order, 
                                    passband_ripple, 
                                    stopband_rejection, 
                                    normalized_freq*2) # ellip normalizes to Nyquist
                self.filter = {
                    "order": order,
                    "passband_ripple": passband_ripple,
                    "stopband_rejection": stopband_rejection,
                    "normalized_frequency": normalized_freq,
                    "frequency": normalized_freq * self._fs_internal,
                    "coefficients": (b, a)
                }
                 
                return(b, a)
            
            
            def apply_filter(self, b, a):
                """Apply the filter with provided coefficients"""
                try:
                    self.waveform_filtered = signal.filtfilt(b, a, self.waveform)
                except ValueError:
                    raise Exception("Waveform vector too short to filter. Consider increasing waveform length or sample rate.")
                
                
            # no filter
            if self._filter == None:
                self.waveform_filtered = self.waveform
                self.filter = None
            
            # filter="auto"
            elif self._filter == "auto":
                try:
                    fc = 0.35 * 1/min(self._risetime, self._falltime)
                except:
                    fc = 0.35 * self._fs_internal
                apply_filter(self, *design_filter(self, normalized_freq=fc/self._fs_internal))
                
            # filter cutoff frequency in Hz given in 'cutoff' parameter
            elif self._filter == 'cutoff':
                if isinstance(self._cutoff, float):
                    apply_filter(self, *design_filter(self, normalized_freq=self._cutoff/self._fs_internal))
                
            # filter=cutoff frequency in Hz
            elif isinstance(self._filter, float):
                self._filter *= self._frequnit
                apply_filter(self, *design_filter(self, normalized_freq=self._filter/self._fs_internal))
                
            # filter=(b, a)
            elif isinstance(self._filter, (tuple, list)):
                apply_filter(self, *self._filter)
                self.filter = {
                    "coefficients": (self._filter[0], self._filter[1])
                }
                
            else:
                self.waveform_filtered = self.waveform
                warnings.warn(f"filter value of {self._filter} is invalid. No filter applied.")
                
                
        compute_thresholds(self)
        compute_rise(self)
        compute_fall(self)
        compute_top(self)
        compute_head(self)
        compute_tail(self)
        assemble(self)
        filter(self)
