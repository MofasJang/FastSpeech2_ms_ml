import os
import numpy as np
import torch
from tqdm import tqdm
import yaml

from model.speaker_embedding import PreDefinedEmbedder

in_dir="./raw_data/test"
out_dir="./preprocessed_data/test"
preprocess_config = yaml.load(open("config/AISHELL3_ecapa/preprocess.yaml", "r"), Loader=yaml.FullLoader)
model_config = yaml.load(open("config/AISHELL3_ecapa/model.yaml", "r"), Loader=yaml.FullLoader)
speaker_emb = PreDefinedEmbedder(preprocess_config, model_config)
speakers = {}
embedding_dir="./preprocessed_data/test/spker_embed/ECAPA-TDNN"
if not os.path.exists(embedding_dir):
    os.makedirs(embedding_dir)
embedding=[]
for i, speaker in enumerate(tqdm(os.listdir(in_dir))):
    speakers[speaker] = i
    for wav_name in tqdm(os.listdir(os.path.join(in_dir, speaker))):
        basename = wav_name.split(".")[0]
        mel_path=os.path.join(
            out_dir,
            "mel",
            "{}-mel-{}.npy".format(speaker, basename),
        )
        if not os.path.exists(mel_path):
            continue
        mel_spectrogram=np.load(mel_path)
        embedding.append(speaker_emb(torch.from_numpy(mel_spectrogram).to("cuda")).to("cpu").numpy() ) 
        
    spker_embed_filename = '{}-spker_embed.npy'.format(speaker)   
    np.save(os.path.join(embedding_dir, spker_embed_filename), \
        np.mean(embedding, axis=0), allow_pickle=False)