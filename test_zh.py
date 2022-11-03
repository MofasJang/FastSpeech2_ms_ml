import gc
from synthesize import preprocess_engnese, synthesize,preprocess_english, preprocess_mandarin
import torch
from play import playwav
import numpy as np
import yaml
import sys
import os
from utils.model import get_model, get_vocoder
import time
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = torch.device("cpu")

language="zh-en"
read="no"
multi_speaker=False
if language=="zh":
    if multi_speaker:
        preprocess_config="config/AISHELL3/preprocess.yaml"
        model_config="config/AISHELL3/model.yaml"
        train_config="config/AISHELL3/train.yaml"
        restore_step=1200000
    else:
        preprocess_config="config/baker/preprocess.yaml"
        model_config="config/baker/model.yaml"
        train_config="config/baker/train.yaml"
        restore_step=1000000
    source="./text_zh.txt"
    mode = "batch"
elif language=="en":
    preprocess_config="config/LJSpeech/preprocess.yaml"
    model_config="config/LJSpeech/model.yaml"
    train_config="config/LJSpeech/train.yaml"
    restore_step=900000
    source="./text_en.txt"
    mode = "batch"
    speaker_id=0
elif language=="zh-en":
    preprocess_config="config/baker-LJSpeech/preprocess.yaml"
    model_config="config/baker-LJSpeech/model.yaml"
    train_config="config/baker-LJSpeech/train.yaml"
    restore_step=310000
    source="./text_zh_en.txt"
    mode = "batch"
pitch_control=1
energy_control=1
duration_control=1



# Read Config
preprocess_config = yaml.load(
    open(preprocess_config, "r",encoding='UTF-8'), Loader=yaml.FullLoader
)
model_config = yaml.load(open(model_config, "r"), Loader=yaml.FullLoader)
train_config = yaml.load(open(train_config, "r"), Loader=yaml.FullLoader)

configs = (preprocess_config, model_config, train_config)

# Get model
model = get_model(restore_step, configs, device, train=False)

# Load vocoder
vocoder = get_vocoder(model_config, device)

batchs=[]
ids=[]
# Preprocess texts
with open(source, "r", encoding="utf-8") as f:
    for line in f.readlines():
        line = line.strip() 
        words=line.split("|")
        if multi_speaker:
            speaker_id=int(words[0])
        else:
            speaker_id=0
        line=words[1]
        
        
        id = raw_texts = [line[:100]]
        ids.append(id[0])
        speakers = np.array([speaker_id])
        if preprocess_config["preprocessing"]["text"]["language"] == "en":
            texts = np.array([preprocess_english(line, preprocess_config)])
        elif preprocess_config["preprocessing"]["text"]["language"] == "zh":
            texts = np.array([preprocess_mandarin(line, preprocess_config)])
        elif preprocess_config["preprocessing"]["text"]["language"] == "zh-en":
            texts = np.array([preprocess_engnese(line, preprocess_config)])
        text_lens = np.array([len(texts[0])])
        spker_embed=np.array([])
        batchs.append((id, raw_texts, speakers, texts, text_lens, max(text_lens),spker_embed))

control_values = pitch_control, energy_control, duration_control

synthesize(model, restore_step, configs, vocoder, batchs, control_values)

if read=="no":
    sys.exit()
for id in ids:
    playwav(os.path.join(train_config["path"]["result_path"],"{}.wav".format(id)))
