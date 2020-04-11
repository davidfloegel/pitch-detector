from flask import Flask
import crepe
from scipy.io import wavfile
import pandas as pd
import base64
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as ss
from flask import request
from flask import jsonify


app = Flask(__name__)

freq_note_mapping = pd.read_csv("./freq_note_mapping.tsv", delimiter="\t")

# start with the frequency below the lowest note that has the same distance as the next higher note
lowest_frequency = freq_note_mapping['Frequency (Hz)'][0] - (freq_note_mapping['Frequency (Hz)'][1] - freq_note_mapping['Frequency (Hz)'][0])

last_frequency = lowest_frequency
note_gaussians = {}
for index, row in freq_note_mapping.iterrows():
    note = row['Note']
    freq = row['Frequency (Hz)']
    note_gaussians[note] = {
        "mu": freq,
        "sigma": (freq - last_frequency)/4
    }
    last_frequency = freq

def base64_to_wave(base64data):
    """
    Converts a base64 string to wav data

    Parameters
    ----------
    base64data : string
        base 64 audio data

    Returns
    -------
    rate : int
        Sample rate of wav file.
    data : numpy array
        Data read from wav file.  Data-type is determined from the file;
    """
    base64audio_bytes = base64data.encode('utf-8')
    with open('./analyseme.wav', 'wb') as file_to_save:
        decoded_data = base64.decodebytes(base64audio_bytes)
        file_to_save.write(decoded_data)
    sr, audio = wavfile.read('./analyseme.wav')

    return sr, audio


def independent_ttest(data1, data2, alpha):
    # calculate means
    mean1, mean2 = np.mean(data1), np.mean(data2)
    # calculate standard errors
    se1, se2 = ss.stats.sem(data1), ss.stats.sem(data2)
    # standard error on the difference between the samples
    sed = np.sqrt(se1**2.0 + se2**2.0)
    # calculate the t statistic
    t_stat = (mean1 - mean2) / sed
    # degrees of freedom
    df = len(data1) + len(data2) - 2
    # calculate the critical value
    cv = ss.t.ppf(1.0 - alpha, df)
    # calculate the p-value
    p = (1.0 - ss.t.cdf(abs(t_stat), df)) * 2.0
    # return everything
    return t_stat, df, cv, p


def run_t_test(data, expected_mu, expected_sigma, alpha):

    # generate two independent samples
    expected_sampled = expected_sigma * np.random.randn(len(data)) + expected_mu

    # calculate the t test
    t_stat, df, cv, p = independent_ttest(data, expected_sampled, alpha)
#     print('t=%.3f, df=%d, cv=%.3f, p=%.3f' % (t_stat, df, cv, p))

    # interpret via p-value
    if p > alpha:
        accept_null_hypothesis_that_means_are_equal = True
    else:
        accept_null_hypothesis_that_means_are_equal = False

    return accept_null_hypothesis_that_means_are_equal, p


def run_analysis(base64data):
    # convert to audio and load as wav data
    sr, audio = base64_to_wave(base64data)
    # detect frequencies
    time, frequency, confidence, activation = crepe.predict(audio, sr)
    # run p test against all notes
    alpha = 0.05
    results = []
    for note, note_data in note_gaussians.items():
        means_are_equal, p = run_t_test(frequency, note_data['mu'], note_data['sigma'], alpha)
        if (means_are_equal):
            results.append({
                "note": note,
                "p": p
            })
    # return confidences
    return results


@app.route('/identify', methods=['POST'])
def identify():
    base64data = request.json['data']

    return jsonify(run_analysis(base64data))
