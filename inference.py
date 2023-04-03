import json
import math
import os

import IPython.display as ipd
import torch
import torchaudio
import typer
from scipy.io.wavfile import write
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader

import commons
import utils
from data_utils import (
    TextAudioCollate,
    TextAudioLoader,
    TextAudioSpeakerCollate,
    TextAudioSpeakerLoader,
)
from models import SynthesizerTrn
from text import text_to_sequence
from text.symbols import symbols


def get_text(text, hps):
    text_norm = text_to_sequence(text, hps.data.text_cleaners)
    if hps.data.add_blank:
        text_norm = commons.intersperse(text_norm, 0)
    text_norm = torch.LongTensor(text_norm)
    return text_norm


def main(model_path: str):
    hps = utils.get_hparams_from_file("./configs/lidan_base.json")

    net_g = SynthesizerTrn(
        len(symbols),
        hps.data.filter_length // 2 + 1,
        hps.train.segment_size // hps.data.hop_length,
        n_speakers=hps.data.n_speakers,
        **hps.model
    ).cuda()
    _ = net_g.eval()

    _ = utils.load_checkpoint(model_path, net_g, None)

    stn_tst = get_text("我是许多", hps)
    with torch.no_grad():
        x_tst = stn_tst.cuda().unsqueeze(0)
        x_tst_lengths = torch.LongTensor([stn_tst.size(0)]).cuda()
        sid = torch.LongTensor([0]).cuda()
        audio = (
            net_g.infer(
                x_tst,
                x_tst_lengths,
                sid=sid,
                noise_scale=0.667,
                noise_scale_w=0.8,
                length_scale=1,
            )[0][0, 0]
            .data.cpu()
            .float()
            # .numpy()
        )
    ipd.display(ipd.Audio(audio, rate=hps.data.sampling_rate, normalize=False))
    save_audio("nihao.wav", audio, hps.data.sampling_rate)


def save_audio(path: str, tensor: torch.Tensor, sampling_rate: int = 16000):
    torchaudio.save(path, tensor.unsqueeze(0), sampling_rate, bits_per_sample=16)
    # dataset = TextAudioSpeakerLoader(hps.data.validation_files, hps.data)
    # collate_fn = TextAudioSpeakerCollate()
    # loader = DataLoader(
    #     dataset,
    #     num_workers=8,
    #     shuffle=False,
    #     batch_size=1,
    #     pin_memory=True,
    #     drop_last=True,
    #     collate_fn=collate_fn,
    # )
    # data_list = list(loader)


if __name__ == "__main__":
    typer.run(main)
