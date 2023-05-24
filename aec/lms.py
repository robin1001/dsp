import sys

import librosa
import numpy as np
import soundfile as sf

def lms(x, d, n=256, mu = 0.1):
    """
    Args:
        x: reference signal
        d: microphone signal
        n: n-order filter
        mu: learning rate
    """
    num_steps = min(len(x), len(d)) - n
    u = np.zeros(n)
    w = np.zeros(n)
    e = np.zeros(num_steps)
    for n in range(num_steps):
        u[1: ] = u[:-1]
        u[0] = x[n]
        e_n = d[n] - np.dot(u, w)
        w = w + mu * e_n * u
        e[n] = e_n
    return e



if __name__ == '__main__':
    mix, sr = librosa.load(sys.argv[1], sr=16000)
    reference, sr = librosa.load(sys.argv[2], sr=16000)
    print(sr)
    result = lms(mix, reference, 256, 0.1)
    sf.write(sys.argv[3], result, sr, subtype='PCM_16')
