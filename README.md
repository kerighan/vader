# Voice Activity Detection with Python

### Installing

```
pip install vader
```

### Basic usage

```python
import vader

# use your own mono, preferably 16kHz .wav file
filename = "audio.wav"

# returns segments of vocal activity (unit: seconds)
# note: it uses a pre-trained NN by default
segments = vader.vad(filename)

# where to dump audio files
out_folder = "segments"
# write segments into .wav files
vader.vad_to_files(segments, filename, out_folder)
```

You can also use different pre-trained models by specifying the method parameter

```python
# logistic method
segments = vader.vad(filename, threshold=.1, window=20, method="logistic")

# multi-layer perceptron method
segments = vader.vad(filename, threshold=.1, window=20, method="nn")

# Naive Bayes method
segments = vader.vad(filename, threshold=.5, window=10, method="nb")

# Random Forest method
segments = vader.vad(filename, threshold=.5, window=10, method="rf")
```
The `threshold` parameter is the ratio of voice frames above which a window of frames is counted as a voiced sample. The `window` parameter controls the number of frames considered, and thus the length of the voiced samples.

You can also train your own models:

```python
import vader
model = vader.train.logistic_regression(mfccs, activities)
model = vader.train.random_forest_classifier(mfccs, activities)
model = vader.train.NN(mfccs, activities)
model = vader.train.NB(mfccs, activities)
```
The variable `mfccs` is a list of varying length mfcc features (num_samples, *varying_lengths*, 13), while `activities` is a list of binary vectors whose lengths match those of the mfcc features (num_samples, *varying_lengths*), equal to 1 when a frame is voiced, and 0 otherwise.

## Authors

Maixent Chenebaux