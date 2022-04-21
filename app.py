from flask import Flask, request, render_template
import pandas as pd
import tensorflow as tf
import keras
from keras.models import load_model
import wfdb
import numpy as np
from scipy.signal import butter, filtfilt
from sklearn.preprocessing import StandardScaler
import os
import neurokit2 as nk

# app = Flask(__name__)
# path = r"/content/gdrive/My Drive/arrhythmia/"

app = Flask(__name__)


import sys

@app.route('/', methods=['GET', 'POST'])
def main():
    arrhythmic = 0
    condition = ""
    ratio = 0
    output = {}
    
    # If a form is submitted
    if request.method == "POST":

        myfiledat = request.files["myfiledat"]
        myfilehea = request.files["myfilehea"]


        myfiledat.save(myfiledat.filename)
        myfilehea.save(myfilehea.filename)



        file_name_array = myfiledat.filename.split('.')
        record = wfdb.rdrecord(file_name_array[0])

        model = load_model("splitonpatients.h5")
        # record = wfdb.rdrecord("100")

        def resamplerecord(signal, frequency, targetfrequency):
            from wfdb import processing
            ratio = targetfrequency / frequency
            if ratio == 1.0:
              return signal

            newsignal = []
            channels = np.shape(signal)[1]
            for channel in range(channels):
              ns, _ = wfdb.processing.resample_sig(signal[:,channel], frequency, targetfrequency)
              newsignal.append(ns)
            

            return np.column_stack(newsignal)
          

        def filtersignal(signal, order, cutoff, sample_rate):
            nyquist_rate = 0.5 * sample_rate
            lowcut, highcut = cutoff[0] / nyquist_rate, cutoff[1] / nyquist_rate
            b, a = butter(order, [lowcut,highcut], btype = 'band')

            newsignal = []
            channels = np.shape(signal)[1]
            for channel in range(channels):
              ns = filtfilt(b, a, signal[:,channel])
              newsignal.append(ns)

            return np.column_stack(newsignal)



        frequency = record.fs
        sequences = []

        record.p_signal = resamplerecord(record.p_signal, frequency, 360)
        record.p_signal = filtersignal(record.p_signal, order = 5, cutoff = [0.5,15], sample_rate = 360)

        scaler = StandardScaler()
        record.p_signal = scaler.fit_transform(record.p_signal)

        _, rpeaks = nk.ecg_peaks(record.p_signal[:,0], sampling_rate = 360)
        rpeaks = rpeaks['ECG_R_Peaks'].tolist()

        window = 256
        for i, sample in enumerate(rpeaks):
            sequence = np.array([])
            start, end = sample - window // 2, sample + window // 2
            
            if 0 < start < end < record.p_signal.shape[0]:
            
              sequence = record.p_signal[start:end, 0]
              sequence.reshape(1, -1, 1)

              if (sequence.size > 0) :
                sequences.append(sequence)

        x = np.vstack(sequences)
        x = np.reshape(x, (x.shape[0], x.shape[1], 1))
        predictions = model.predict(x)

        
        normal = 0
        
        for (x,y), value in np.ndenumerate(predictions):
          # print(value)
          if(value >= 0.5):
            arrhythmic += 1
          else:
            normal += 1

        print(f"Hello world! {arrhythmic}", file=sys.stderr)
        print("Arrhythmic", arrhythmic)

        print("Normal", normal)

        ratio = arrhythmic/(arrhythmic+normal)
        print("Ratio", ratio)


        if(ratio < 0.001):
          condition = "No arrhythmia detected"
        elif (ratio < 0.002):
          condition = "Mild"
        elif (ratio < 0.005):
          condition = "Moderate"
        elif (ratio < 0.007):
          condition = "Severe"
        else :
          condition = "Critical"

        print("Condition", condition)

        # Get values through input bars
        
        
        # weight = request.form.get("weight")
        
        # Put inputs to dataframe
        # X = pd.DataFrame([[height, weight]], columns = ["Height", "Weight"])
        
        # Get prediction
        output = {"arrhythmic": arrhythmic, "condition": condition}
        print(output)
        
    else:
        output = {"arrhythmic": arrhythmic, "arrythmic": condition}
        # arrhythmic = 0
        
    return render_template("website.html", output = output)

# Running the app

if __name__ == "__main__":
    app.secret_key = os.urandom(100)
    app.run(host='0.0.0.0', port=8080, debug = True )
