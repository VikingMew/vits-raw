#!/bin/bash

mkdir -p lidan_data
# Change to the folder containing the audio files
cd lidan_data_44100

# Loop through all the files in the folder with .wav extension
for file in *.wav
do
  # Use sox to change the sample rate to 22050 and save the output file in the output folder
  sox "$file" -r 22050 -e signed -b 16 -c 1 "../lidan_data/${file%.*}.wav"
done






