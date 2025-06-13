import whisper
import os
import ffmpeg
import subprocess

ffmpeg_executable = r"E:\Đồ án chuyên ngành\resource\ffmpeg-n7.1-39-g64e2864cb9-win64-lgpl-7.1\bin\ffmpeg.exe"
video_path = "E:\\Đồ án chuyên ngành\\resource\\test_subtitles\\videotesst.mp4"
audio_dir = r"E:\Đồ án chuyên ngành\resource\test_subtitles"
audio_filename = "output_audio.wav"
audio_path = os.path.join(audio_dir, audio_filename)
subprocess.run([ffmpeg_executable, "-i", video_path, "-ac", "1", "-ar", "16000", audio_path], check=True)
if os.path.exists(audio_path):
    print("Audio file exists and ready for processing.")
else:
    print("Audio file not found.")
model = whisper.load_model("base")
result = model.transcribe(audio_path, language="vi")

output_dir = "E:\\tttt"
output_srt_path = os.path.join(output_dir, "output_subtitles.srt")

os.makedirs(output_dir, exist_ok=True)

with open(output_srt_path, "w", encoding="utf-8") as srt_file:
    for i, segment in enumerate(result["segments"]):
        start = segment["start"]
        end = segment["end"]
        text = segment["text"]
        srt_file.write(f"{i+1}\n{start:.2f} --> {end:.2f}\n{text}\n\n")
print(f"Hoàn thành! Phụ đề đã lưu tại {output_srt_path}")
os.remove(audio_path)
# from pywhispercpp.model import Model

# model = Model('large', n_threads=2)
# segments = model.transcribe(r"E:\tttt\tmpvb05klil.wav",language="vi")
# for segment in segments:
#     print(segment.text)