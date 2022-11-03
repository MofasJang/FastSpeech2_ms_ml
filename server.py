from flask import Flask, request,Response
from synthesize import synthesize,preprocess_english,preprocess_mandarin
import torch
import numpy as np
import yaml
import os
from utils.model import get_model, get_vocoder
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
language="zh"
# preprocess_config="./config/AISHELL3/preprocess.yaml"
# model_config="./config/AISHELL3/model.yaml"
# train_config="./config/AISHELL3/train.yaml"
# restore_step=1200000
# speaker_id=12

preprocess_config="./config/baker/preprocess.yaml"
model_config="./config/baker/model.yaml"
train_config="./config/baker/train.yaml"
restore_step=1000000
speaker_id=0


pitch_control=1
energy_control=1
duration_control=1


# Read Config
preprocess_config = yaml.load(
    open(preprocess_config, "r", encoding='UTF-8'), Loader=yaml.FullLoader
)
model_config = yaml.load(open(model_config, "r"), Loader=yaml.FullLoader)
train_config = yaml.load(open(train_config, "r"), Loader=yaml.FullLoader)
configs = (preprocess_config, model_config, train_config)

# Get model
model = get_model(restore_step, configs, device, train=False)

# Load vocoder
vocoder = get_vocoder(model_config, device)


app = Flask(__name__)
@app.route("/TTS", methods=["POST"])
def TTS():
    global language,preprocess_config,model_config,train_config,configs,configs,model,vocoder,restore_step
    if request.form["language"]!=language:
        if request.form["language"]=="en":
            preprocess_config="config/LJSpeech/preprocess.yaml"
            model_config="config/LJSpeech/model.yaml"
            train_config="config/LJSpeech/train.yaml"
            restore_step=900000
            speaker_id=0
        else:
            preprocess_config="./config/baker/preprocess.yaml"
            model_config="./config/baker/model.yaml"
            train_config="./config/baker/train.yaml"
            restore_step=200000
            speaker_id=0

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
        
        language=request.form["language"]
        
    text = request.form["text"]
    speed = request.form["speed"]
    high = request.form["high"]
    speed_list={"慢速":1.5,"标准":1,"快速":0.7}
    high_list={"低音":0.5,"标准":1,"高亢":2}
    pitch_control=high_list[high]
    duration_control=speed_list[speed]
    control_values = pitch_control, energy_control, duration_control
    
    line = text.strip() 
    id="t"
    raw_texts = [line[:100]]
    if request.form["speaker"]:
        speaker_id=int(request.form["speaker"])
    speakers = np.array([speaker_id])
    if preprocess_config["preprocessing"]["text"]["language"] == "en":
        texts = np.array([preprocess_english(line, preprocess_config)])
    elif preprocess_config["preprocessing"]["text"]["language"] == "zh":
        texts = np.array([preprocess_mandarin(line, preprocess_config)])
    text_lens = np.array([len(texts[0])])
    batchs=[(id, raw_texts, speakers, texts, text_lens, max(text_lens))]
    print(model, restore_step, configs, vocoder, batchs, control_values)
    synthesize(model, restore_step, configs, vocoder, batchs, control_values)
    f=open(os.path.join(train_config["path"]["result_path"],"{}.wav".format(id)),"rb")
    return Response(f,mimetype="audio/wav")


app.run("0.0.0.0", debug=True)
