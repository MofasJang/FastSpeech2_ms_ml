import os
import json

import torch
import torch.nn as nn
import torch.nn.functional as F

from .modules import VarianceAdaptor
from utils.tools import get_mask_from_lengths
from model.speaker_embedding import SpeakerEmbedding
from memory_profiler import profile

class FastSpeech2(nn.Module):
    """ FastSpeech2 """

    def __init__(self, preprocess_config, model_config):
        super(FastSpeech2, self).__init__()
        if model_config["block_type"] == "transformer":
            from transformer import Encoder, Decoder
            from transformer import PostNet
        elif model_config["block_type"] == "conformer":
            from transformer.conformer import Encoder, Decoder
            from transformer.blocks import PostNet
        self.model_config = model_config
        self.encoder = Encoder(model_config)
        self.variance_adaptor = VarianceAdaptor(preprocess_config, model_config)
        self.decoder = Decoder(model_config)
        self.finetune = model_config["finetune"]
        self.mel_linear = nn.Linear(
            model_config["transformer"]["decoder_hidden"],
            preprocess_config["preprocessing"]["mel"]["n_mel_channels"],
        )
        self.embedder=preprocess_config["preprocessing"]["speaker_embedder"]
        self.postnet = PostNet()
        self.speaker_emb = None
        self.multi_speaker=model_config["multi_speaker"]
        if model_config["multi_speaker"] and (self.embedder == 'ECAPA-TDNN' or self.embedder == "dvector"):
            self.speaker_emb = SpeakerEmbedding(model_config)
        # self.make_emb = PreDefinedEmbedder(preprocess_config, model_config)
        # self.speaker_classifier_1=SpeakerClassifier(preprocess_config, model_config)
        # self.speaker_classifier_2=SpeakerClassifier(preprocess_config, model_config)
        # self.liner=nn.Linear(256,256)
        
        # self.speaker_emb=nn.Linear(192,256)
        
        elif model_config["multi_speaker"]:
            with open(
                os.path.join(
                    preprocess_config["path"]["preprocessed_path"], "speakers.json"
                ),
                "r",
            ) as f:
                n_speaker = len(json.load(f))
            self.speaker_emb = nn.Embedding(
                n_speaker,
                model_config["transformer"]["encoder_hidden"],
            )

    def forward(
        self,
        speakers,
        texts,
        src_lens,
        max_src_len,
        spker_embed=None,
        mels=None,
        mel_lens=None,
        max_mel_len=None,
        p_targets=None,
        e_targets=None,
        d_targets=None,
        p_control=1.0,
        e_control=1.0,
        d_control=1.0,
    ):
        src_masks = get_mask_from_lengths(src_lens, max_src_len)
        mel_masks = (
            get_mask_from_lengths(mel_lens, max_mel_len)
            if mel_lens is not None
            else None
        )
        output = self.encoder(texts, src_masks)
        
        # if self.multi_speaker:
        #     output = output + self.speaker_emb1(speakers).unsqueeze(1).expand(
        #         -1, max_src_len, -1
        #     )
            # spker_embed=self.speaker_emb(spker_embed)
            # # speaker_classifier_output_1 = self.speaker_classifier_1(spker_embed)
            # output=output+self.add_speaker_embedding(output,spker_embed)
            # speaker_emb=spker_embed.unsqueeze(1).expand(-1, max_src_len, -1)
        
        if self.embedder=='ECAPA-TDNN' or self.embedder=="dvector":
            output = output + self.speaker_emb(spker_embed).unsqueeze(1).expand(
                -1, max_src_len, -1
            )
        elif self.speaker_emb is not None:
            output = output + self.speaker_emb(speakers).unsqueeze(1).expand(
                -1, max_src_len, -1
            )
        #     speaker_classifier_output_1=self.speaker_classifier_1(spker_embed)
            
        (
            output,
            p_predictions,
            e_predictions,
            log_d_predictions,
            d_rounded,
            mel_lens,
            mel_masks,
        ) = self.variance_adaptor(
            output,
            src_masks,
            speakers,
            mel_masks,
            max_mel_len,
            p_targets,
            e_targets,
            d_targets,
            p_control,
            e_control,
            d_control,
        )
        if self.embedder=='ECAPA-TDNN' or self.embedder=="dvector" and self.model_config["block_type"] != "conformer":
            output = output + self.speaker_emb(spker_embed).unsqueeze(1).expand(
                -1, max(mel_lens), -1
            )
        # if self.model_config["block_type"] == "conformer":
        #     output, mel_masks = self.decoder(output, mel_masks, spker_embed)
        # else:  
        output, mel_masks = self.decoder(output, mel_masks)
        output = self.mel_linear(output)

        postnet_output = self.postnet(output) + output
        # speaker_classifier_output_2 = self.speaker_classifier_2(self.make_emb(postnet_output))
        
        return (
            output,
            postnet_output,
            p_predictions,
            e_predictions,
            log_d_predictions,
            d_rounded,
            src_masks,
            mel_masks,
            src_lens,
            mel_lens,
            # speaker_classifier_output_1,
            # speaker_classifier_output_2,
        )
        
    # def add_speaker_embedding(self, x, speaker_embedding):
    #         # SV2TTS
    #     # The input x is the encoder output and is a 3D tensor with size (batch_size, num_chars, tts_embed_dims)
    #     # When training, speaker_embedding is also a 2D tensor with size (batch_size, speaker_embedding_size)
    #     #     (for inference, speaker_embedding is a 1D tensor with size (speaker_embedding_size))
    #     # This concats the speaker embedding for each char in the encoder output

    #     # Save the dimensions as human-readable names
    #     batch_size = x.size()[0]
    #     num_chars = x.size()[1]

    #     if speaker_embedding.dim() == 1:
    #         idx = 0
    #     else:
    #         idx = 1

    #     # Start by making a copy of each speaker embedding to match the input text length
    #     # The output of this has size (batch_size, num_chars * tts_embed_dims)
    #     speaker_embedding_size = speaker_embedding.size()[idx]
    #     e = speaker_embedding.repeat_interleave(num_chars, dim=idx)

    #     # Reshape it and transpose
    #     e = e.reshape(batch_size, speaker_embedding_size, num_chars)
    #     e = e.transpose(1, 2)

    #     # Concatenate the tiled speaker embedding with the encoder output
    #     return e