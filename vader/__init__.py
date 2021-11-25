import os

import numpy as np
import scipy.io.wavfile as wavfile
from sonopy import filterbanks, mel_spec, mfcc_spec, power_spec

import vader.train

path = os.path.dirname(__file__)


def mfcc_from_file(filename):
    sr, y = wavfile.read(filename)
    mfcc = mfcc_spec(y, sr)
    duration = y.shape[0] / sr
    return mfcc, duration


def sample_to_training_data(item):
    audio_fn = item["audio"]
    sr, y = wavfile.read(audio_fn)
    duration = y.shape[0] / sr

    mfcc = mfcc_spec(y, sr)
    activity = np.zeros((mfcc.shape[0]))
    N = mfcc.shape[0]
    for utt in item["json"]:
        if utt["word"] == "sil":
            continue
        start = utt["start"]
        end = utt["end"]
        start_index = round(N * start / duration)
        end_index = round(N * end / duration)
        activity[start_index:end_index] = 1.
    return mfcc, activity


def to_training_data(data):
    from tqdm import tqdm

    mfccs = []
    activities = []
    for item in tqdm(data):
        mfcc, activity = sample_to_training_data(item)
        mfccs.append(mfcc)
        activities.append(activity)
    return np.array(mfccs), np.array(activities)


def save(model, filename):
    from joblib import dump
    dump(model, filename)


def load(filename):
    from joblib import load as jload
    return jload(filename)


def rollavg(y, n=10):
    from scipy import convolve
    return convolve(y, np.ones(n, dtype="float") / n, "same")


def vad(
    filename,
    window=10,
    threshold=.4,
    min_duration=.5,
    method="nn",
    raw=False
):
    # load model
    model_fn = os.path.join(path, f"pretrained/{method}.joblib")
    if os.path.exists(model_fn):
        model = load(model_fn)
    else:
        raise NameError(f"Unknown method: {method}")

    # compute features
    mfccs, duration = mfcc_from_file(filename)

    # predict
    y = model.predict(mfccs)
    if raw:
        return y

    # quantize
    y = (y >= .5).astype(int)

    # rolling average
    avg = vader.rollavg(y, window)
    N = avg.shape[0]

    # find segments
    chain = []
    last_index = None
    indices = np.where(avg > threshold)[0]
    for index in indices:
        index_time = duration * index / N
        if last_index is None:
            chain.append([index_time, None])
            last_index = index
        else:
            if last_index + 1 == index:
                last_index = index
            else:
                chain[-1][1] = duration * last_index / N
                chain.append([index_time, None])
                last_index = index

    # remove short audio samples
    chain_fix = []
    for start, end in chain:
        if end is None:
            end = duration
        if end - start > min_duration:
            chain_fix.append((start, end))
    return chain_fix


def vad_to_files(chain, filename, out_folder):
    sr, y = wavfile.read(filename)
    duration = y.shape[0] / sr

    if not os.path.exists(out_folder):
        os.makedirs(out_folder)

    chunk = 0
    N = y.shape[0]
    for start, end in chain:
        start_index = int(round(N * start / duration))
        end_index = int(round(N * end / duration))
        fn = os.path.join(out_folder, f"{chunk}.wav")
        wavfile.write(fn, sr, y[start_index:end_index])
        chunk += 1
