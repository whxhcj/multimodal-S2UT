import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import argparse
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC
import torch, torchaudio, tqdm
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def generate_transcription(
    model_path: str, 
    tts_wav_dir: str, 
    transcript_txt: str, 
):
    transcriptions = []
    # load model and processor
    processor = Wav2Vec2Processor.from_pretrained(model_path)
    model = Wav2Vec2ForCTC.from_pretrained(model_path)
    model.to(device)
    tts_wav_files = os.listdir(tts_wav_dir)
    tts_wav_files.sort(key=lambda x: int(x.split('_')[0]))
    for wav_file in tqdm.tqdm(tts_wav_files):
        wav_path = os.path.join(tts_wav_dir, wav_file)
        audio, sr = torchaudio.load(wav_path)
        audio = audio[0]
        # tokenize
        input_values = processor(audio, sampling_rate=sr, return_tensors="pt", padding="longest").input_values
        input_values = input_values.to(device)
        # retrieve logits
        # print(wav_file)
        # if wav_file in ("128_pred.wav"):
        #     transcriptions.append("None")
        #     continue
        logits = model(input_values).logits
        logits = logits.cpu()
        # take argmax and decode
        predicted_ids = torch.argmax(logits, dim=-1)
        transcription = processor.batch_decode(predicted_ids)
        transcription = transcription[0]
        transcriptions.append(transcription)
    with open(transcript_txt, mode="w+") as f:
        f.write('\n'.join(transcriptions))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="transcript from generated tts")
    parser.add_argument("--model-path", type=str)
    parser.add_argument("--tts-wav-dir", type=str)
    parser.add_argument("--transcript-txt", type=str)
    args = parser.parse_args()
    generate_transcription(
        model_path=args.model_path, 
        tts_wav_dir=args.tts_wav_dir, 
        transcript_txt=args.transcript_txt, 
    )
