import speechmetrics
import numpy as np
import tqdm
import glob
import os
wav_files = glob.glob(os.path.join("F:\\dataset\\LJSpeech-1.1\\wavs", '*.wav'), recursive=True)
metrics = speechmetrics.load('absolute.mosnet')
scores=[]
for wavpath in tqdm.tqdm(wav_files, desc='score'):
    scores.append(metrics(wavpath))
avg=0
for score in scores:
    avg=avg+np.mean(score['mosnet'])
avg=avg/len(scores)
print("原音频MOS：{}".format(avg))

# wav_files = glob.glob(os.path.join("output/result/AISHELL3", '*.wav'), recursive=True)
# wav_files = glob.glob(os.path.join("F:\\speech\\vits-main\\sample", '*.wav'), recursive=True)
# metrics = speechmetrics.load('absolute.mosnet')
# scores=[]
# for wavpath in tqdm.tqdm(wav_files, desc='score'):
#     scores.append(metrics(wavpath))
# avg=0
# for score in scores:
#     avg=avg+np.mean(score['mosnet'])
# avg=avg/len(scores)
# print("合成音频MOS：{}".format(avg))