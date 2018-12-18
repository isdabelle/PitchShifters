import numpy as np
from utils import ms2smp, compute_stride, win_taper, dft_rescale
import sounddevice as sd

"""
Real-time pitch shifting with granular synthesis for shift factors <=1.0
"""

""" User selected parameters """
grain_len = 30
grain_over = 0.99
shift_factor = 1.2
data_type = np.int16

# derived parameters
MAX_VAL = np.iinfo(data_type).max
GRAIN_LEN_SAMP = ms2smp(grain_len, 8000)
STRIDE = compute_stride(GRAIN_LEN_SAMP, grain_over)
OVERLAP_LEN = GRAIN_LEN_SAMP-STRIDE

# allocate input and output buffers
input_buffer = np.zeros(STRIDE, dtype=data_type)
output_buffer = np.zeros(STRIDE, dtype=data_type)


# state variables and constants
def init():

    # lookup table for tapering window
    global WIN
    WIN = win_taper(GRAIN_LEN_SAMP, grain_over, data_type)

    # lookup table for linear interpolation
    global SAMP_VALS
    global AMP_VALS
    #SAMP_VALS, AMP_VALS = build_linear_interp_table(GRAIN_LEN_SAMP, shift_factor, data_type)

    # create arrays to pass between buffers (state variables)
    global grain
    grain = np.zeros(int(GRAIN_LEN_SAMP), dtype=data_type)
    global x_concat
    x_concat = np.zeros(int(GRAIN_LEN_SAMP), dtype=data_type)

    # create arrays for intermediate values
    global prev_latency, prev_grain
    prev_latency = np.zeros(int(OVERLAP_LEN), dtype=data_type)
    prev_grain = np.zeros(int(OVERLAP_LEN), dtype=data_type)


# the process function!
# the process function!
def process(input_buffer, output_buffer, buffer_len):

    # need to specify those global variables changing in this function (state variables and intermediate values)
    global grain, x_concat, prev_latency, prev_grain

    # append samples from previous buffer, construction of the grain
    for n in range(int(GRAIN_LEN_SAMP)):
        if n < int(OVERLAP_LEN):
            x_concat[n] = prev_latency[n]
        else:
            x_concat[n] = input_buffer[n-int(OVERLAP_LEN)]

    # rescale
    #for n in range(int(GRAIN_LEN_SAMP)):
        #grain[n] = float(AMP_VALS[n]/MAX_VAL)*x_concat[int(SAMP_VALS[n])] + (1-float(AMP_VALS[n]/MAX_VAL))*x_concat[int(SAMP_VALS[n])+1]
    grain = dft_rescale(x_concat,shift_factor)

    # apply window
    for n in range(int(GRAIN_LEN_SAMP)):
        grain[n] = grain[n] * float(WIN[n]/MAX_VAL)

    # write to output
    for n in range(int(GRAIN_LEN_SAMP)):
        # overlapping part
        if n < OVERLAP_LEN:
            output_buffer[n] = grain[n] + prev_grain[n]
        # non-overlapping part
        elif n < STRIDE:
            output_buffer[n] = grain[n]
        # update state variables
        else:
            prev_latency[n-int(STRIDE)] = x_concat[n]
            prev_grain[n -int(STRIDE)] = grain[n]

"""
# Nothing to touch after this!
# """
try:
    sd.default.samplerate = 8000
    sd.default.blocksize = STRIDE
    sd.default.dtype = data_type

    def callback(indata, outdata, frames, time, status):
        if status:
            print(status)
        process(indata[:,0], outdata[:,0], frames)

    init()
    with sd.Stream(channels=1, callback=callback):
        print('#' * 80)
        print('press Return to quit')
        print('#' * 80)
        input()
except KeyboardInterrupt:
    parser.exit('\nInterrupted by user')