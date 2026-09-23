import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

# --------------------------------------------------
# 1. FILTER SPECIFICATIONS
# --------------------------------------------------
Fs = 1000       # Sampling frequency (Hz)
Fp = 100        # Passband frequency (Hz)
Fst = 200       # Stopband frequency (Hz)
Ap = 1          # Passband attenuation (dB)
Ast = 40        # Stopband attenuation (dB)

# --------------------------------------------------
# 2. PREWARPING
# --------------------------------------------------
wp = 2 * np.pi * Fp / Fs
ws = 2 * np.pi * Fst / Fs

Wp = 2 * Fs * np.tan(wp / 2)
Ws = 2 * Fs * np.tan(ws / 2)

# --------------------------------------------------
# 3. BUTTERWORTH FILTER ORDER
# --------------------------------------------------
N = int(np.ceil(
    np.log10((10**(Ast/10)-1) / (10**(Ap/10)-1))
    / (2 * np.log10(Ws/Wp))
))

# --------------------------------------------------
# 4. ANALOG BUTTERWORTH FILTER
# --------------------------------------------------
N, Wn = signal.buttord(Wp, Ws, Ap, Ast, analog=True)
ba, aa = signal.butter(N, Wn, btype='low', analog=True)

# --------------------------------------------------
# 5. BILINEAR TRANSFORMATION
# --------------------------------------------------
b, a = signal.bilinear(ba, aa, fs=Fs)

print("Butterworth filter order =", N)
print("Analog cutoff frequency =", Wn, "rad/s")
print("Digital numerator coefficients:")
print(b)
print("Digital denominator coefficients:")
print(a)

# --------------------------------------------------
# 6. ANALOG FREQUENCY RESPONSE
# --------------------------------------------------
Wa, Ha = signal.freqs(ba, aa, worN=1000)

plt.figure(figsize=(9, 5))
plt.semilogx(Wa, 20*np.log10(abs(Ha)))
plt.axvline(Wp, linestyle='--', label='Passband')
plt.axvline(Ws, linestyle='--', label='Stopband')
plt.xlabel("Frequency (rad/s)")
plt.ylabel("Magnitude (dB)")
plt.title("Analog Butterworth Filter Response")
plt.grid()
plt.legend()
plt.show()

# --------------------------------------------------
# 7. DIGITAL FREQUENCY RESPONSE
# --------------------------------------------------
f, H = signal.freqz(b, a, worN=2048, fs=Fs)

plt.figure(figsize=(9, 5))
plt.plot(f, 20*np.log10(np.maximum(abs(H), 1e-10)))
plt.axvline(Fp, linestyle='--', label='Passband = 100 Hz')
plt.axvline(Fst, linestyle='--', label='Stopband = 200 Hz')
plt.xlabel("Frequency (Hz)")
plt.ylabel("Magnitude (dB)")
plt.title("Digital Butterworth IIR Filter Response")
plt.xlim(0, 500)
plt.grid()
plt.legend()
plt.show()

# --------------------------------------------------
# 8. POLE-ZERO PLOT
# --------------------------------------------------
z, p, k = signal.tf2zpk(b, a)

plt.figure(figsize=(6, 6))
plt.scatter(np.real(z), np.imag(z), marker='o', label='Zeros')
plt.scatter(np.real(p), np.imag(p), marker='x', label='Poles')

theta = np.linspace(0, 2*np.pi, 500)
plt.plot(np.cos(theta), np.sin(theta), linestyle='--', label='Unit circle')

plt.axhline(0, linewidth=0.8)
plt.axvline(0, linewidth=0.8)
plt.xlabel("Real")
plt.ylabel("Imaginary")
plt.title("Pole-Zero Plot of Digital Filter")
plt.axis('equal')
plt.grid()
plt.legend()
plt.show()

# --------------------------------------------------
# 9. TEST SIGNAL
# --------------------------------------------------
t = np.arange(0, 1, 1/Fs)

# 50 Hz wanted signal + 300 Hz unwanted signal
x = np.sin(2*np.pi*50*t) + 0.5*np.sin(2*np.pi*300*t)

# Apply digital IIR filter
y = signal.lfilter(b, a, x)

# --------------------------------------------------
# 10. INPUT AND OUTPUT WAVEFORMS
# --------------------------------------------------
plt.figure(figsize=(10, 5))
plt.plot(t, x, label="Input signal")
plt.plot(t, y, label="Filtered signal")
plt.xlim(0, 0.1)
plt.xlabel("Time (s)")
plt.ylabel("Amplitude")
plt.title("Signal Before and After Filtering")
plt.grid()
plt.legend()
plt.show()

# --------------------------------------------------
# 11. FFT BEFORE FILTERING
# --------------------------------------------------
X = np.fft.rfft(x)
freq = np.fft.rfftfreq(len(x), 1/Fs)

plt.figure(figsize=(9, 5))
plt.plot(freq, np.abs(X))
plt.xlim(0, 500)
plt.xlabel("Frequency (Hz)")
plt.ylabel("Magnitude")
plt.title("Frequency Spectrum Before Filtering")
plt.grid()
plt.show()

# --------------------------------------------------
# 12. FFT AFTER FILTERING
# --------------------------------------------------
Y = np.fft.rfft(y)

plt.figure(figsize=(9, 5))
plt.plot(freq, np.abs(Y))
plt.xlim(0, 500)
plt.xlabel("Frequency (Hz)")
plt.ylabel("Magnitude")
plt.title("Frequency Spectrum After Filtering")
plt.grid()
plt.show()
