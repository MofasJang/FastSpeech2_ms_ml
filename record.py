import pyaudio
import wave
from pypinyin import pinyin,Style
import yaml
import os

framerate = 16000
NUM_SAMPLES = 2000
channels = 1
sampwidth = 2
TIME = 10


def save_wave_file(filename, data):
    wf = wave.open(filename, "wb")
    wf.setnchannels(channels)
    wf.setsampwidth(sampwidth)
    wf.setframerate(framerate)
    wf.writeframes(b"".join(data))
    wf.close()


def record(f, time=5):
    p = pyaudio.PyAudio()
    stream = p.open(
        format=pyaudio.paInt16,
        channels=1,
        rate=framerate,
        input=True,
        frames_per_buffer=NUM_SAMPLES,
    )
    my_buf = []
    count = 0
    print("录音中(5s)")
    while count < TIME * time:
        string_audio_data = stream.read(NUM_SAMPLES)
        my_buf.append(string_audio_data)
        count += 1
        print(".", end="", flush=True)
    print()
    save_wave_file(f, my_buf)
    stream.close()

def make_wavs(f,t,n):
    if not os.path.exists(os.path.dirname(f)):
        os.mkdir(os.path.dirname(f))
    lines=[]
    source="./text_zh.txt"
    with open(source, "r", encoding="utf-8") as file:
        for line in file.readlines():
            line = line.strip() 
            words=line.split("|")
            lines.append(words[1])
    filename=f.split('.')
    for i in range(n):
        pinyins = [p[0] for p in pinyin(lines[i], style=Style.TONE3, strict=False, neutral_tone_with_five=True)]
        with open(filename[0]+str(i)+".lab","w",) as f1:
            f1.write(" ".join(pinyins))
    for i in range(n):
        record(filename[0]+str(i)+"."+filename[1],t)
        
        
if __name__=="__main__":
    f="raw_data/test/pjt"
    make_wavs(f+'.wav',5,10)

    