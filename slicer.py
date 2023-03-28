import json
import subprocess

import typer
from tqdm import tqdm


def main(wav_path: str, json_path: str, output_dir: str):
    with open(json_path, "r") as r:
        j = json.load(r)
    ext = wav_path.split(".")[-1]
    sample_list = []
    for item in tqdm(j["segments"]):
        """
        {
          "id": 0,
          "seek": 0,
          "start": 0,
          "end": 9,
          "text": "Zither Harp",
          "tokens": [
            50364,
            57,
            1839,
            3653,
            79,
            50814
          ],
          "temperature": 0,
          "avg_logprob": -0.1725857459892661,
          "compression_ratio": 1.0737704918032787,
          "no_speech_prob": 0.5328633189201355
        }
        """
        duration = item["end"] - item["start"]

        if len(item["text"]) == 0 or duration < 2:
            continue
        # slice
        start = item["start"]
        output_path = f"{output_dir}/{item['id']}.{ext}"
        subprocess.run(
            f"ffmpeg -y -i {wav_path} -ss {start} -t {duration} -vn -acodec copy {output_path}",
            stdout=subprocess.DEVNULL,
            stderr=subprocess.STDOUT,
            shell=True,
        )
        # add item
        sample_list.append(f"{output_path}|0|{item['text']}")
    with open(f"{output_dir}/filelist.txt", "w") as w:
        for l in sample_list:
            print(l, file=w)


if __name__ == "__main__":
    typer.run(main)
