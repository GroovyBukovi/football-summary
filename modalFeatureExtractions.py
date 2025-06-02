import os
import cv2
import librosa
import numpy as np
import pandas as pd
import torch
import moviepy as mp
from scipy.io import wavfile
from tqdm import tqdm
from transformers import BertTokenizer, BertModel
from whisper import load_model as load_whisper
from torchvision import models, transforms
from pyAudioAnalysis import ShortTermFeatures
from PIL import Image


# -------------------------------
# CONFIGURATION
# -------------------------------
SEGMENT_DURATION = 10  # seconds per segment
AUDIO_FEATURE_DIM = 40
VISUAL_FEATURE_DIM = 2048
TEXT_FEATURE_DIM = 768
FRAME_SAMPLE_RATE = 1  # fps
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# -------------------------------
# AUDIO FEATURES
# -------------------------------

from datetime import timedelta

def seconds_to_hhmmss(seconds):
    return str(timedelta(seconds=int(seconds)))


"""def extract_audio_features(video_path, segment_duration, total_duration, output_csv):
    audio_path = 'temp_audio.wav'
    mp.VideoFileClip(video_path).audio.write_audiofile(audio_path, logger=None)

    y, sr = librosa.load(audio_path, sr=None)
    segment_samples = int(segment_duration * sr)
    num_segments = int(np.ceil(total_duration / segment_duration))

    data = []
    for i in range(num_segments):
        timestamp = i * segment_duration
        start_sample = i * segment_samples
        end_sample = start_sample + segment_samples
        segment = y[start_sample:end_sample] if end_sample <= len(y) else y[start_sample:]

        if len(segment) > 0:
            mfcc = librosa.feature.mfcc(y=segment, sr=sr, n_mfcc=AUDIO_FEATURE_DIM).mean(axis=1)
        else:
            mfcc = np.zeros(AUDIO_FEATURE_DIM)

        data.append([timestamp] + list(mfcc))

    os.remove(audio_path)

    columns = ['timestamp'] + [f'audio_{i}' for i in range(AUDIO_FEATURE_DIM)]
    pd.DataFrame(data, columns=columns).to_csv(output_csv, index=False)
    print(f"[AUDIO] Saved to {output_csv}")"""



"""def extract_audio_features(video_path, segment_duration, total_duration, output_csv):
    audio_path = 'temp_audio.wav'
    mp.VideoFileClip(video_path).audio.write_audiofile(audio_path, logger=None)

    # Read full audio signal
    sr, signal = wavfile.read(audio_path)
    signal = signal.astype(float)

    # Normalize if stereo
    if len(signal.shape) == 2:
        signal = np.mean(signal, axis=1)

    # Parameters
    segment_samples = int(segment_duration * sr)
    num_segments = int(np.ceil(total_duration / segment_duration))

    # pyAudioAnalysis uses short-term windowing internally
    win_size = 0.05  # 50 ms window
    step_size = 0.05  # 50 ms step

    data = []
    for i in range(num_segments):
        timestamp = i * segment_duration
        start_sample = i * segment_samples
        end_sample = start_sample + segment_samples

        if end_sample > len(signal):
            segment = signal[start_sample:]
        else:
            segment = signal[start_sample:end_sample]

        if len(segment) == 0:
            continue

        # Extract features using pyAudioAnalysis
        [features, feature_names] = ShortTermFeatures.feature_extraction(
            segment, sr, win_size * sr, step_size * sr
        )

        # Mean and std of each feature across short windows
        mean_features = np.mean(features, axis=1)
        std_features = np.std(features, axis=1)
        full_vector = np.concatenate([mean_features, std_features])
        data.append([timestamp] + list(full_vector))

    os.remove(audio_path)

    # Construct feature column names
    columns = ['timestamp'] + [f"{name}_mean" for name in feature_names] + [f"{name}_std" for name in feature_names]
    df = pd.DataFrame(data, columns=columns)
    df.to_csv(output_csv, index=False)
    print(f"[AUDIO] Saved to {output_csv} with {len(columns)-1} features per segment")"""

def extract_audio_features(video_path, segment_duration, total_duration, output_csv):
    audio_path = 'temp_audio.wav'
    mp.VideoFileClip(video_path).audio.write_audiofile(audio_path, logger=None)

    sr, signal = wavfile.read(audio_path)
    signal = signal.astype(float)

    if len(signal.shape) == 2:
        signal = np.mean(signal, axis=1)

    segment_samples = int(segment_duration * sr)
    num_segments = int(np.ceil(total_duration / segment_duration))

    win_size = 0.05  # 50ms
    step_size = 0.05

    data = []
    for i in range(num_segments):
        timestamp = i * segment_duration
        timecode = seconds_to_hhmmss(timestamp)
        start_sample = i * segment_samples
        end_sample = start_sample + segment_samples
        segment = signal[start_sample:end_sample] if end_sample <= len(signal) else signal[start_sample:]

        if len(segment) == 0:
            continue

        features, feature_names = ShortTermFeatures.feature_extraction(
            segment, sr, win_size * sr, step_size * sr
        )
        mean_features = np.mean(features, axis=1)
        std_features = np.std(features, axis=1)
        full_vector = np.concatenate([mean_features, std_features])
        row = [i, timestamp, timecode] + list(full_vector)
        data.append(row)

    os.remove(audio_path)

    columns = ['segment_index', 'timestamp', 'timecode'] + \
              [f"{name}_mean" for name in feature_names] + [f"{name}_std" for name in feature_names]

    pd.DataFrame(data, columns=columns).to_csv(output_csv, index=False)
    print(f"[AUDIO] Saved to {output_csv}")

# -------------------------------
# VISUAL FEATURES
# -------------------------------
"""def extract_visual_features(video_path, segment_duration, total_duration, output_csv):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    num_segments = int(np.ceil(total_duration / segment_duration))
    frames_per_segment = int(segment_duration * fps)

    resnet = models.resnet50(pretrained=True).to(DEVICE)
    resnet = torch.nn.Sequential(*list(resnet.children())[:-1])
    resnet.eval()

    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    data = []
    for seg in tqdm(range(num_segments), desc="Extracting Visual Features"):
        timestamp = seg * segment_duration
        segment_frames = []

        for f in range(frames_per_segment):
            ret, frame = cap.read()
            if not ret:
                break
            if f % int(fps / FRAME_SAMPLE_RATE) == 0:
                img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                img = Image.fromarray(img)
                img_tensor = preprocess(img).unsqueeze(0).to(DEVICE)
                with torch.no_grad():
                    feat = resnet(img_tensor).cpu().squeeze().numpy()
                segment_frames.append(feat)

        if segment_frames:
            segment_feat = np.mean(segment_frames, axis=0)
        else:
            segment_feat = np.zeros(VISUAL_FEATURE_DIM)

        data.append([timestamp] + list(segment_feat))

    cap.release()

    columns = ['timestamp'] + [f'visual_{i}' for i in range(VISUAL_FEATURE_DIM)]
    pd.DataFrame(data, columns=columns).to_csv(output_csv, index=False)
    print(f"[VISUAL] Saved to {output_csv}")"""

def extract_visual_features(video_path, segment_duration, total_duration, output_csv):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frames_per_segment = int(segment_duration * fps)
    num_segments = int(np.ceil(total_duration / segment_duration))

    resnet = models.resnet50(pretrained=True).to(DEVICE)
    resnet = torch.nn.Sequential(*list(resnet.children())[:-1])
    resnet.eval()

    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    data = []
    for seg in range(num_segments):
        timestamp = seg * segment_duration
        timecode = seconds_to_hhmmss(timestamp)
        frames = []

        for f in range(frames_per_segment):
            ret, frame = cap.read()
            if not ret:
                break
            if f % int(fps) == 0:  # 1 fps sampling
                img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                img = Image.fromarray(img)
                img_tensor = preprocess(img).unsqueeze(0).to(DEVICE)
                with torch.no_grad():
                    feat = resnet(img_tensor).cpu().squeeze().numpy()
                frames.append(feat)

        if frames:
            pooled = np.mean(frames, axis=0)
        else:
            pooled = np.zeros(2048)

        row = [seg, timestamp, timecode] + list(pooled)
        data.append(row)

    cap.release()

    columns = ['segment_index', 'timestamp', 'timecode'] + \
              [f'resnet50_feat_{i}' for i in range(2048)]

    pd.DataFrame(data, columns=columns).to_csv(output_csv, index=False)
    print(f"[VISUAL] Saved to {output_csv}")

# -------------------------------
# TEXT FEATURES
# -------------------------------
"""def extract_text_features(video_path, segment_duration, total_duration, output_csv):
    audio_path = 'temp_audio.wav'
    mp.VideoFileClip(video_path).audio.write_audiofile(audio_path, logger=None)

    whisper_model = load_whisper("base")
    whisper_result = whisper_model.transcribe(audio_path)

    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    bert_model = BertModel.from_pretrained('bert-base-uncased').to(DEVICE)
    bert_model.eval()

    num_segments = int(np.ceil(total_duration / segment_duration))
    text_feats = [np.zeros(TEXT_FEATURE_DIM) for _ in range(num_segments)]

    for segment in whisper_result['segments']:
        start_time = segment['start']
        text = segment['text']
        seg_idx = int(start_time // segment_duration)

        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=128).to(DEVICE)
        with torch.no_grad():
            outputs = bert_model(**inputs)
            cls_embedding = outputs.last_hidden_state[:, 0, :].cpu().squeeze().numpy()

        if 0 <= seg_idx < num_segments:
            text_feats[seg_idx] = cls_embedding

    os.remove(audio_path)

    data = []
    for i in range(num_segments):
        timestamp = i * segment_duration
        data.append([timestamp] + list(text_feats[i]))

    columns = ['timestamp'] + [f'text_{i}' for i in range(TEXT_FEATURE_DIM)]
    pd.DataFrame(data, columns=columns).to_csv(output_csv, index=False)
    print(f"[TEXT] Saved to {output_csv}")"""


def extract_text_features(video_path, segment_duration, total_duration, output_csv):
    audio_path = 'temp_audio.wav'
    mp.VideoFileClip(video_path).audio.write_audiofile(audio_path, logger=None)

    # Whisper transcription
    whisper_model = load_whisper("base")
    whisper_result = whisper_model.transcribe(audio_path)

    # BERT model
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    bert_model = BertModel.from_pretrained('bert-base-uncased').to(DEVICE)
    bert_model.eval()

    num_segments = int(np.ceil(total_duration / segment_duration))
    text_feats = [np.zeros(TEXT_FEATURE_DIM) for _ in range(num_segments)]
    transcript_strings = [""] * num_segments
    segment_counts = [0] * num_segments

    for segment in whisper_result['segments']:
        start_time = segment['start']
        text = segment['text']
        seg_idx = int(start_time // segment_duration)

        if 0 <= seg_idx < num_segments:
            inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=128).to(DEVICE)
            with torch.no_grad():
                outputs = bert_model(**inputs)
                cls_embedding = outputs.last_hidden_state[:, 0, :].cpu().squeeze().numpy()

            if segment_counts[seg_idx] == 0:
                text_feats[seg_idx] = cls_embedding
            else:
                text_feats[seg_idx] = (text_feats[seg_idx] * segment_counts[seg_idx] + cls_embedding) / (segment_counts[seg_idx] + 1)

            segment_counts[seg_idx] += 1
            transcript_strings[seg_idx] += " " + text.strip()

    os.remove(audio_path)

    # Final data structure
    data = []
    for i in range(num_segments):
        timestamp = i * segment_duration
        timecode = seconds_to_hhmmss(timestamp)
        transcript = transcript_strings[i].strip()
        row = [i, timestamp, timecode] + list(text_feats[i]) + [transcript]
        data.append(row)

    # Columns
    columns = ['segment_index', 'timestamp', 'timecode'] + \
              [f'text_{i}' for i in range(TEXT_FEATURE_DIM)] + ['transcript']

    # Save CSV
    pd.DataFrame(data, columns=columns).to_csv(output_csv, index=False)
    print(f"[TEXT] Saved to {output_csv} with timecode and segment_index columns.")

# -------------------------------
# MAIN RUNNER
# -------------------------------
if __name__ == "__main__":
    video_file = "Crystal Palace v Manchester City | Final | Emirates FA Cup 2024-25.mp4"

    clip = mp.VideoFileClip(video_file)
    duration = clip.duration

    extract_audio_features(video_file, SEGMENT_DURATION, duration, "C-M-audio_features.csv")
    extract_visual_features(video_file, SEGMENT_DURATION, duration, "C-M-visual_features.csv")
    extract_text_features(video_file, SEGMENT_DURATION, duration, "C-M-text_features.csv")
