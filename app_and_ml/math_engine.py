"""
math_engine.py — Part 1: Core Math Engine (Shreyas / Person A)
"""

import numpy as np
from scipy import signal as sp_signal


def bandpass_filter(raw_signal: np.ndarray, fs: float, lowcut: float = 0.5,
                     highcut: float = 5.0, order: int = 3) -> np.ndarray:
    """
    Keep only the 0.5-5.0 Hz band (30-300 BPM) -- real heartbeats live here.
    Removes breathing drift (too slow) and hand tremor/noise (too fast).

    fs = sample rate of raw_signal, in Hz (samples per second).
    """
    nyquist = 0.5 * fs  # the highest frequency this sample rate can represent
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = sp_signal.butter(order, [low, high], btype="band")
    return sp_signal.filtfilt(b, a, raw_signal)  # filtfilt = zero phase shift


def detect_beats(clean_wave: np.ndarray, fs: float, max_bpm: float = 140) -> np.ndarray:
    """
    Find the systolic peaks (one per heartbeat) in the filtered wave.

    distance  = minimum samples between two peaks, based on the fastest
                heart rate we'll allow (max_bpm) -- prevents counting the
                same beat's noise wiggle as a second beat.
    prominence = how much a peak must stick up above its surroundings --
                filters out small noise bumps that aren't real beats.
    """
    min_distance = int(fs * 60.0 / max_bpm)
    peaks, _ = sp_signal.find_peaks(
        clean_wave,
        distance=max(min_distance, 1),
        prominence=np.std(clean_wave) * 0.3,
    )
    return peaks


def second_derivative(clean_wave: np.ndarray, dt: float) -> np.ndarray:
    """
    The APG (Acceleration Photoplethysmogram): take the derivative twice.
    Real PPG analysis uses this to sharpen subtle landmarks. We compute it
    here (useful for your demo graph / write-up), even though the peak
    search below works on the clean wave directly -- see the note in
    compute_indices about why.
    """
    first_deriv = np.gradient(clean_wave, dt)
    return np.gradient(first_deriv, dt)


def compute_indices(clean_wave: np.ndarray, beat_peaks: np.ndarray, fs: float):
    """
    For each beat (one systolic peak to the next), search the windowed
    segment for the reflection bump and compute:
      - Stiffness Index (SI):  1 / (delay as a FRACTION of that beat's
        period) -- fraction, not raw seconds, so it isn't confounded by
        heart-rate differences between beats or people.
      - Reflection Index (RI): reflection bump height / systolic height.

    NOTE on a mistake worth knowing about in advance: if you build a
    synthetic test wave by adding two sine waves at the SAME frequency to
    fake "systolic + reflection," you'll never get a real second peak --
    two same-frequency sine waves just combine into one bigger/shifted
    sine wave, mathematically. You need two separate bump shapes (see the
    test below) for a genuine two-peak waveform to test against.

    Also: if those bump shapes are too sharp/narrow, the bandpass filter
    can "ring" (oscillate) after the sharp edge and create a FAKE extra
    bump later in the cycle that looks like a reflection peak but isn't.
    We guard against that by bounding the search window to the first
    ~55% of each beat (see below) -- real reflection bumps show up in
    roughly this range; filter ringing tends to show up later. Once
    you're processing real video, always check that your extracted
    Stiffness Index moves in the right direction on a wave you already
    know the answer for, before trusting it on a real recording.
    """
    if len(beat_peaks) < 2:
        return None, None

    si_values, ri_values = [], []
    for i in range(len(beat_peaks) - 1):
        start, end = beat_peaks[i], beat_peaks[i + 1]
        period_samples = end - start
        if period_samples < 8:
            continue

        segment = clean_wave[start:end]
        systolic_amp = segment[0]

        lo = max(int(0.10 * period_samples), 1)
        hi = min(int(0.55 * period_samples), period_samples - 1)
        if hi <= lo + 1:
            continue
        window = segment[lo:hi]

        local_peaks, props = sp_signal.find_peaks(window, prominence=0.0)
        if len(local_peaks) == 0:
            continue

        # Take the peak with the highest prominence within this bounded
        # window. Bounding the window to 10%-55% of the beat (rather than
        # searching all the way to 85%) is what makes "biggest prominence"
        # safe to use -- a real reflection bump physiologically shows up
        # in roughly this range, while filter ringing tends to show up
        # later in the cycle. Search too wide, and "biggest prominence"
        # will happily grab a ringing artifact instead of the real bump.
        best = local_peaks[int(np.argmax(props["prominences"]))]
        reflection_idx = lo + best
        reflection_amp = segment[reflection_idx]

        delay_fraction = reflection_idx / period_samples
        if delay_fraction > 0:
            si_values.append(1.0 / delay_fraction)
        if systolic_amp != 0:
            ri_values.append(abs(reflection_amp / systolic_amp))

    if not si_values:
        return None, None
    return float(np.median(si_values)), (float(np.median(ri_values)) if ri_values else None)


def make_test_beat_wave(fs, duration_sec, bpm, reflection_delay_frac, reflection_amp):
    """
    Builds a wave with a GENUINE two-bump shape per beat (a systolic
    Gaussian bump + a separate reflection Gaussian bump), unlike naively
    summing two same-frequency sine waves (see the note above).
    """
    n_samples = int(duration_sec * fs)
    wave = np.zeros(n_samples)
    t_cursor = 0.0
    while t_cursor < duration_sec:
        period = 60.0 / bpm
        start_idx = int(t_cursor * fs)
        end_idx = min(int((t_cursor + period) * fs), n_samples)
        if start_idx >= n_samples:
            break
        local_t = np.arange(end_idx - start_idx) / fs

        tau1 = period * 0.08   # systolic bump width -- wide enough to avoid filter ringing
        tau2 = period * 0.10   # reflection bump width
        delay = reflection_delay_frac * period

        systolic = np.exp(-(local_t / tau1) ** 2)
        reflection = reflection_amp * np.exp(-((local_t - delay) / tau2) ** 2)
        wave[start_idx:end_idx] += (systolic + reflection)[: end_idx - start_idx]
        t_cursor += period
    return wave


if __name__ == "__main__":
    # ---- Test 1: the filter ----
    fs = 100.0
    duration = 5.0
    t = np.arange(0, duration, 1 / fs)

    true_heart_hz = 1.2
    heartbeat = np.sin(2 * np.pi * true_heart_hz * t)
    breathing_drift = 0.5 * np.sin(2 * np.pi * 0.25 * t)
    fast_noise = 0.3 * np.sin(2 * np.pi * 20 * t)

    noisy = heartbeat + breathing_drift + fast_noise
    clean = bandpass_filter(noisy, fs=fs)

    zero_crossings = np.sum(np.diff(np.sign(clean)) != 0)
    estimated_hz = zero_crossings / 2 / duration
    print(f"[Filter test] true={true_heart_hz}Hz  estimated={estimated_hz:.2f}Hz")

    # ---- Test 2: peak detection ----
    peaks = detect_beats(clean, fs=fs)
    print(f"\n[Peak detection test] Expected ~6 beats in {duration}s at {true_heart_hz*60:.0f} BPM")
    print(f"Detected {len(peaks)} beats at sample indices: {peaks}")

    if len(peaks) >= 2:
        intervals_sec = np.diff(peaks) / fs
        estimated_bpm = 60.0 / np.mean(intervals_sec)
        print(f"Estimated BPM from peak spacing: {estimated_bpm:.1f} "
              f"(true: {true_heart_hz*60:.0f})")

    # ---- Test 3: Stiffness Index on a HEALTHY vs STIFF fake wave ----
    print("\n[Stiffness Index test] SI should be HIGHER for the 'stiff' profile")
    for label, delay_frac, refl_amp in [("healthy", 0.45, 0.40), ("stiff", 0.22, 0.15)]:
        wave = make_test_beat_wave(fs=100.0, duration_sec=10.0, bpm=75,
                                    reflection_delay_frac=delay_frac, reflection_amp=refl_amp)
        clean_wave = bandpass_filter(wave, fs=100.0)
        beats = detect_beats(clean_wave, fs=100.0)
        si, ri = compute_indices(clean_wave, beats, fs=100.0)
        print(f"  {label:8s}: SI={si}  RI={ri}  (beats detected: {len(beats)})")