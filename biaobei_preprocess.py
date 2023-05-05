import os
import re
import sys

import typer


def main(dir_path: str):
    annoation_path = f"{dir_path}/ProsodyLabeling/000001-010000.txt"
    wave_dir = f"{dir_path}/Wave"
    candidates = []
    with open(annoation_path, "r") as r:
        for line in r:
            if line.startswith("\t"):
                continue
            compoents = line.strip().split("\t")
            if len(compoents) != 2:
                continue
            file_id = compoents[0]
            raw_annotation = compoents[1]
            modified_annotation = re.sub("""#\d""", """""", raw_annotation)
            candidates.append(f"{wave_dir}/{file_id}.wav|1|{modified_annotation}")
    with open("biaobei.filelist.txt", "w") as w:
        for item in candidates:
            print(item, file=w)


if __name__ == "__main__":
    typer.run(main)
