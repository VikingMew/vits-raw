import asyncio
import hashlib
import io
import json
import logging
import os
from datetime import datetime

import scipy.io.wavfile as wavfile
import torch
import torchaudio
import tornado
import tornado.web

import commons
import utils
from models import SynthesizerTrn
from text import text_to_sequence
from text.symbols import symbols

MODEL_PATH = "G_3363000.pth"
OUTPUT_DIR = "output/"
hps = utils.get_hparams_from_file("./configs/lidan_base.json")

net_g = SynthesizerTrn(
    len(symbols),
    hps.data.filter_length // 2 + 1,
    hps.train.segment_size // hps.data.hop_length,
    n_speakers=hps.data.n_speakers,
    **hps.model,
).cuda()
_ = net_g.eval()
_ = utils.load_checkpoint(MODEL_PATH, net_g, None)

app_log = logging.getLogger("tornado.application")
tornado.log.enable_pretty_logging()


def get_text(text, hps):
    text_norm = text_to_sequence(text, hps.data.text_cleaners)
    if hps.data.add_blank:
        text_norm = commons.intersperse(text_norm, 0)
    text_norm = torch.LongTensor(text_norm)
    return text_norm


def generate(utterance: str):
    stn_tst = get_text(utterance, hps)
    now = datetime.now()

    current_time = now.strftime("%Y%m%d-%H%M%S")
    output = OUTPUT_DIR + f"{current_time}-{utterance}.wav"
    with torch.no_grad():
        x_tst = stn_tst.cuda().unsqueeze(0)
        x_tst_lengths = torch.LongTensor([stn_tst.size(0)]).cuda()
        sid = torch.LongTensor([1]).cuda()
        audio = (
            net_g.infer(
                x_tst,
                x_tst_lengths,
                sid=sid,
                noise_scale=0.667,
                noise_scale_w=0.8,
                length_scale=1.1,
            )[0][0, 0]
            .data.cpu()
            .float()
        )
    save_audio(output, audio, hps.data.sampling_rate)
    return (output, audio.unsqueeze(0).numpy())


def save_audio(path: str, tensor: torch.Tensor, sampling_rate: int = 16000):
    torchaudio.save(path, tensor.unsqueeze(0), sampling_rate, bits_per_sample=16)


class SoundHandler(tornado.web.RequestHandler):
    async def post(self):
        utterance = self.json_args["utterance"]
        output, audio = generate(utterance)
        (rate, data) = wavfile.read(output)
        buffer = io.BytesIO()
        wavfile.write(buffer, hps.data.sampling_rate, data)
        self.set_header("Content-Type", "audio/wav")
        self.set_header("Content-Length", str(len(buffer.getvalue())))
        self.write(buffer.getvalue())

    async def prepare(self):
        if self.request.headers.get("Content-Type", "").startswith("application/json"):
            self.json_args = json.loads(self.request.body)
        else:
            self.json_args = None


def make_app():
    return tornado.web.Application(
        [
            (r"/sound", SoundHandler),
        ]
    )


async def main():
    app = make_app()
    app.listen(8801)
    await asyncio.Event().wait()


if __name__ == "__main__":
    # cli_main()
    asyncio.run(main())
