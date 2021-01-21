import os
import tqdm
import pickle
import numpy as np
import time
import glob
import pdb
import librosa
import matplotlib.pyplot as plt
import speech_recognition as sr
from moviepy.editor import *
import moviepy.editor as moviepyeditor
import paramiko
from speech_recognition import RequestError
from PIL import Image


def extract_asr(audio_dir, save_dir, key, primary_language='en', secondary_language='es'):
    """
    Uses google speech API to download text recognition for the audio files.

    If there are multiple languages, you may choose two to decide between.

    See function `extract_asr` for more information on getting Google Speech API Key.

    :param str audio_dir: The directory where audio files are stored in `.wav` format.
    :param str save_dir: The directory to save the pickle text files.
    :param str key: The Google Speech API key
    :param str primary_language:
    :param str secondary_language:
    :return:
    """
    subdirs = len([dirs for _, dirs, _ in os.walk(audio_dir) if len(dirs) > 0])
    append = '/'.join(['*' for _ in range(subdirs + 1)])
    audio_dir = os.path.join(audio_dir, append)
    audio_pths = glob.glob(audio_dir)
    for pth in tqdm.tqdm(audio_pths, total=len(audio_pths), desc='Extracting ASR'):
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        identifier = pth.split('/')[-1].replace('.wav', '')
        save_pth = os.path.join(os.path.join(save_dir, identifier + '.pickle'))
        if not os.path.exists(save_pth):
            for attempt in range(10):
                try:
                    text = extract_asr(pth, key, primary_language, secondary_language)
                    with open(save_pth, 'wb') as f:
                        pickle.dump(text, f)
                except RequestError:
                    time.sleep(5)
                    continue
                else:
                    break
    print("Done")


def extract_audio_features(audio_pth):
    """
    Info from: https://www.kdnuggets.com/2020/02/audio-data-analysis-deep-learning-python-part-1.html

    1. Spectral Centroid
    The spectral features (frequency-based features), which are obtained by converting the time-based signal into the
        frequency domain using the Fourier Transform, like fundamental frequency, frequency components, spectral
        centroid, spectral flux, spectral density, spectral roll-off, etc.

    Spectral centroid indicates at which frequency the energy of a spectrum is centered upon or in other words.
    It indicates where the ” center of mass” for a sound is located (like a weighted mean).

    2. Spectral Rolloff
    It is a measure of the shape of the signal. It represents the frequency at which high frequencies decline to 0.
    To obtain it, we have to calculate the fraction of bins in the power spectrum where 85% of its power is at lower frequencies.

    3. Spectral Bandwidth
    The spectral bandwidth is defined as the width of the band of light at one-half the peak maximum (or full width at
        half maximum [FWHM]) and is represented by the two vertical red lines and λSB on the wavelength axis.

    4. Zero-Crossing Rate
    A very simple way for measuring the smoothness of a signal is to calculate the number of zero-crossing within a
        segment of that signal.

    6. Chroma feature
    A chroma feature or vector is typically a 12-element feature vector indicating how much energy of each pitch class,
        {C, C#, D, D#, E, …, B}, is present in the signal. In short, It provides a robust way to describe a similarity
        measure between music pieces.

    Generate a Google Speech API key following this tutorial: http://nerdvittles.com/?page_id=21210
    """
    results = dict()
    y, sr = librosa.load(audio_pth, mono=True)
    # RMSE
    rmse = librosa.feature.rms(y=y)

    # Chroma Feature
    chroma_stft = librosa.feature.chroma_stft(y=y, sr=sr)

    # Spectral Centroid
    spec_cent = librosa.feature.spectral_centroid(y=y, sr=sr)

    # Spectral Bandwidth
    spec_bw = librosa.feature.spectral_bandwidth(y=y, sr=sr)

    # Spectral Rolloff
    rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)

    # Zero Crossings
    zcr = librosa.feature.zero_crossing_rate(y)
    mfcc = librosa.feature.mfcc(y=y, sr=sr)

    results['chrome_stft'] = np.mean(chroma_stft)
    results['rmse'] = np.mean(rmse)
    results['spec_cent'] = np.mean(spec_cent)
    results['spec_bw'] = np.mean(spec_bw)
    results['rolloff'] = np.mean(rolloff)
    results['zcr'] = np.mean(zcr)
    results['mfcc'] = mfcc
    return results


def extract_spectogram(audio_pth, save_dir=None, duration=None, imaging=False):
    """
    This will extract audio features similar to:
        Koutini, K., Eghbal-zadeh, H., & Widmer, G. (2019).
        Receptive-field-regularized CNN variants for acoustic scene classification.
        arXiv preprint arXiv:1909.02859.
    :param str audio_pth: The location of the `.wav` file.
    :param str save_dir: If generating an image, where to save it.
    :param int duration: Duration of audio to limit, if None will extract all
    :param bool imaging: Whether or not to generate a visual
    :return:
    """
    n_fft = 2048  # 2048
    sr = 22050  # 22050  # 44100  # 32000
    mono = True
    n_mels = 256

    hop_length = 512
    fmax = None

    save_name = audio_pth.split('/')[-1].split('.')[0]
    sig, sr = librosa.load(audio_pth, mono=mono, duration=duration)
    sig = sig[np.newaxis]

    spectrograms = list()
    for y in sig:
        # compute stft
        stft = librosa.stft(np.asfortranarray(y), n_fft=n_fft, hop_length=hop_length, win_length=None, window='hann',
                            center=True, pad_mode='reflect')
        # Keep only amplitures
        stft = np.abs(stft)

        # Spectogram weighting
        freqs = librosa.core.fft_frequencies(sr=sr, n_fft=n_fft)
        stft = librosa.perceptual_weighting(stft ** 2, freqs, ref=1.0, amin=1e-10, top_db=80.0)

        spectrogram = librosa.feature.melspectrogram(S=stft, sr=sr, n_mels=n_mels, fmax=fmax)
        spectrograms.append(np.asarray(spectrogram, dtype=np.float32))
    spectrograms = np.asarray(spectrograms, dtype=np.float32)

    # Retired for imaging and visualization
    if imaging:
        cmap = plt.get_cmap('inferno')
        figure = plt.figure(figsize=(3, 3))
        plt.specgram(y, NFFT=n_fft, Fs=2, Fc=0, noverlap=128, cmap=cmap, sides='default', mode='default', scale='dB')
        if save_dir is not None:
            plt.axis('off')
            plt.savefig(os.path.join(save_dir, save_name + '.png'))
            plt.clf()
            return spectrograms
        else:
            plt.axis('off')
            figure.canvas.draw()
            w, h = figure.canvas.get_width_height()
            buf = np.fromstring(figure.canvas.tostring_argb(), dtype=np.uint8)
            buf.shape = (w, h, 4)
            buf = np.roll(buf, 3, axis=2)
            w, h, d = buf.shape
            image = Image.fromarray(buf)
            plt.clf()
            return spectrograms, buf
    return spectrograms


def norm_func(x):
    return (x - tr_mean) / tr_std


def extract_asr(audio_pth, key, primary_language='en', secondary_language='es'):
    """
    Will use the Google Speech API to extract automatic speech recognition in text form.
    Requires a Google Speech API key.

    Can extract up to two languages, a primary and secondary.
    If there is no recognizable speech in the audio file, `SIL` will be returned.

    Instructions can be found here:
        http://nerdvittles.com/?page_id=21210
    Depends on package: https://github.com/Uberi/speech_recognition
    :param str audio_pth: Path to `.wav` file
    :param str key: Google Speech API key
    :return:
    """
    # Text
    r = sr.Recognizer()
    audio = sr.AudioFile(audio_pth)
    results = list()
    with audio as source:
        duration = source.FRAME_COUNT / source.SAMPLE_RATE
        for segment in np.arange(0, duration, step=10):
            audio_file = r.record(source, duration=10)
            try:
                text_primary = r.recognize_google(audio_file, key=key, show_all=True, language=primary_language)
                if secondary_language is not None:
                    text_secondary = r.recognize_google(audio_file, key=key, language=secondary_language, show_all=True)

                if len(text_primary) < 1 and len(text_secondary) < 1:
                    raise sr.UnknownValueError
                elif len(text_primary) < 1 and len(text_secondary) > 1:
                    results.append(text_secondary['alternative'][0]['transcript'])
                elif len(text_primary) > 1 and len(text_secondary) < 1:
                    results.append(text_primary['alternative'][0]['transcript'])
                elif len(text_primary) > 1 and len(text_secondary) > 1:
                    if 'confidence' not in text_primary['alternative'][0].keys():
                        text_primary['alternative'][0]['confidence'] = 0.0
                    if 'confidence' not in text_secondary['alternative'][0].keys():
                        text_secondary['alternative'][0]['confidence'] = 0.0

                    if text_primary['alternative'][0]['confidence'] < text_secondary['alternative'][0]['confidence']:
                        results.append(text_secondary['alternative'][0]['transcript'])
                    else:
                        results.append(text_primary['alternative'][0]['transcript'])
                else:
                    pdb.set_trace()

            except sr.UnknownValueError or TypeError:
                results.append('SIL')
    return results
