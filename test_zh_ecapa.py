from posixpath import basename
from synthesize import synthesize,preprocess_english,preprocess_mandarin
import torch
from play import playwav
import numpy as np
import yaml
import sys
import json
from record import record
import os
from utils.model import get_model, get_vocoder
import time
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

language="zh"
read="yes"
if language=="zh":
    preprocess_config="config/AISHELL3_ecapa/preprocess.yaml"
    model_config="config/AISHELL3_ecapa/model.yaml"
    train_config="config/AISHELL3_ecapa/train.yaml"
    restore_step=2000100
    source="./text_zh.txt"
    mode = "batch"
elif language=="en":
    preprocess_config="config/LJSpeech/preprocess.yaml"
    model_config="config/LJSpeech/model.yaml"
    train_config="config/LJSpeech/train.yaml"
    restore_step=160000
    source="./text_en.txt"
    mode = "batch"
    speaker_id=0
pitch_control=1
energy_control=1
duration_control=1



# Read Config
preprocess_config = yaml.load(
    open(preprocess_config, "r",encoding='UTF-8'), Loader=yaml.FullLoader
)
model_config = yaml.load(open(model_config, "r"), Loader=yaml.FullLoader)
train_config = yaml.load(open(train_config, "r"), Loader=yaml.FullLoader)
with open(os.path.join(preprocess_config["path"]["preprocessed_path"], "speakers.json")) as f:
    speaker_map = json.load(f)
configs = (preprocess_config, model_config, train_config)

# Get model
model = get_model(restore_step, configs, device, train=False)

# Load vocoder
vocoder = get_vocoder(model_config, device)

batchs=[]
ids=[]
start_time=time.time()

# name = "pjt"
# wav="usr_embed/"+name+"_recording.wav"

# 录制声音
# name = "test"
# wav="usr_embed/"+name+".wav"
# n=1
# embeddings=[]
# record(wav)

# 生成嵌入
# from model.speaker_embedding import PreDefinedEmbedder
# import audio as Audio
# import librosa
# for i in range(n):
#     filename=wav.split('.')
#     wav=filename[0]+str(i)+'.'+filename[1]
#     wav, _ = librosa.load(wav)
#     STFT = Audio.stft.TacotronSTFT(
#                 preprocess_config["preprocessing"]["stft"]["filter_length"],
#                 preprocess_config["preprocessing"]["stft"]["hop_length"],
#                 preprocess_config["preprocessing"]["stft"]["win_length"],
#                 preprocess_config["preprocessing"]["mel"]["n_mel_channels"],
#                 preprocess_config["preprocessing"]["audio"]["sampling_rate"],
#                 preprocess_config["preprocessing"]["mel"]["mel_fmin"],
#                 preprocess_config["preprocessing"]["mel"]["mel_fmax"],
#             )
#     mel_spectrogram, energy = Audio.tools.get_mel_from_wav(wav, STFT)
#     mel_spectrogram = mel_spectrogram.T # T, C
#     speaker_emb=PreDefinedEmbedder(preprocess_config,model_config)
#     embeddings.append(speaker_emb(torch.from_numpy(mel_spectrogram)).numpy())
# np.save(wav.split('.')+".npy",np.mean(embeddings, axis=0), allow_pickle=False)

#     np.save(filename[0]+str(i)+".npy",embeddings)

embeddings=np.load("preprocessed_data/test/spker_embed/ECAPA-TDNN/pjt-spker_embed.npy")
embeddings=embeddings[None,:]

# Preprocess texts
with open(source, "r", encoding="utf-8") as f:
    for line in f.readlines():
        line = line.strip() 
        if language=="zh":
            words=line.split("|")
            speaker_id=int(words[0])
            line=words[1]
        spker_embed = embeddings
        # if speaker_id == -1:
        #     spker_embed = embeddings
        # else:
        #     speaker=list(speaker_map.keys())[list(speaker_map.values()).index(speaker_id)]
        #     spker_embed = np.load(
        #         os.path.join(
        #             preprocess_config["path"]["preprocessed_path"],
        #             "spker_embed",
        #             "ECAPA-TDNN",
        #             "{}-spker_embed.npy".format(speaker),
        #         ))
        #     spker_embed=spker_embed[None,:]
        id = raw_texts = [line[:100]]
        ids.append(id[0])
        speakers = np.array([speaker_id])
        if preprocess_config["preprocessing"]["text"]["language"] == "en":
            texts = np.array([preprocess_english(line, preprocess_config)])
        elif preprocess_config["preprocessing"]["text"]["language"] == "zh":
            texts = np.array([preprocess_mandarin(line, preprocess_config)])
        text_lens = np.array([len(texts[0])])
        batchs.append((id, raw_texts, speakers, texts, text_lens, max(text_lens),spker_embed))

control_values = pitch_control, energy_control, duration_control

synthesize(model, restore_step, configs, vocoder, batchs, control_values)
end_time=time.time()
print("共耗时{}秒".format(end_time-start_time))
if read=="no":
    sys.exit()
for id in ids:
    playwav(os.path.join(train_config["path"]["result_path"],"{}.wav".format(id)))
