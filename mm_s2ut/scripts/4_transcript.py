import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC
import torch, torchaudio, tqdm
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# inference_dir = "/opt/data/private/dsy/project/model/multimodal_S2UT/mm_s2ut/checkpoints/enhanced_fr-en/inference"
# inference_dir = "/opt/data/private/dsy/project/model/multimodal_S2UT/mm_s2ut/checkpoints/textless_fr-en/inference"
# inference_dir = "/opt/data/private/dsy/project/model/multimodal_S2UT/mm_s2ut/checkpoints/textless_fr-en_mm/inference"
inference_dir = "/opt/data/private/dsy/project/model/multimodal_S2UT/mm_s2ut/checkpoints/textless_fr-en_mm_dropout/inference"
# inference_dir = "/opt/data/private/dsy/project/model/multimodal_S2UT/mm_s2ut/checkpoints/textless_fr-en_mm_2/inference"

# model_path = "/opt/data/private/dsy/project/model/transformer_ckpt/wav2vec2-large-xlsr-53-french"
# model_path = "/opt/data/private/dsy/project/model/transformer_ckpt/wav2vec2-large-fr-voxpopuli-french"
# model_path = "/opt/data/private/dsy/project/model/transformer_ckpt/wav2vec2-large-960h-lv60-self"
model_path = "/opt/data/private/dsy/project/checkpoint/wav2vec2-large-960h-lv60-self"

# -------------------------------------------------------------------------------------------------
def generate_transcription(
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
    generate_transcription(
        tts_wav_dir=os.path.join(inference_dir, "tts"), 
        transcript_txt=os.path.join(inference_dir, "tts_transcript.txt"), 
    )
