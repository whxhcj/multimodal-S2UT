import sys
f = open(
    "/opt/data/private/dsy/project/model/multimodal_S2UT/mm_s2ut/scripts/print.txt", 
    mode="w+"
)
sys.stdout = f

import os, json, tqdm
import pandas as pd
import sacrebleu
from sacrebleu.metrics import BLEU
from speech_to_speech_translation.text_cleaner.cleaners import english_cleaners

src_lang, tgt_lang = "fr", "en"
gen_subset = "valid"
# gen_subset = "train"
# transcript_txt_path = transcript_txt_path = "/opt/data/private/dsy/project/model/multimodal_S2UT/mm_s2ut/checkpoints/enhanced_fr-en/inference/tts_transcript.txt"
# transcript_txt_path = "/opt/data/private/dsy/project/model/multimodal_S2UT/mm_s2ut/checkpoints/textless_fr-en/inference/tts_transcript.txt"
# transcript_txt_path = "/opt/data/private/dsy/project/model/multimodal_S2UT/mm_s2ut/checkpoints/textless_fr-en_mm/inference/tts_transcript.txt"
transcript_txt_path = "/opt/data/private/dsy/project/model/multimodal_S2UT/mm_s2ut/checkpoints/textless_fr-en_mm_dropout/inference/tts_transcript.txt"
# transcript_txt_path = "/opt/data/private/dsy/project/model/multimodal_S2UT/mm_s2ut/checkpoints/textless_fr-en_mm_2/inference/tts_transcript.txt"
wav_dir = f"/opt/data/private/dsy/project/dataset/multi30k-dataset/data/speech/16khz_wav/{tgt_lang}/{gen_subset}"

ref_txt = f"/opt/data/private/dsy/project/dataset/multi30k-dataset/data/text-clean/{gen_subset}.{tgt_lang}"
tsv_path = f"/opt/data/private/dsy/project/dataset/multi30k-dataset/data/speech/format_data/fr-en/{gen_subset}.tsv"
# ref_txt = f"/opt/data/private/dsy/project/dataset/multi30k-dataset/data/text-clean/test.2016.en"
# tsv_path = f"/opt/data/private/dsy/project/dataset/multi30k-dataset/data/speech/format_data/fr-en/test.2016.tsv"
# ref_txt = f"/opt/data/private/dsy/project/dataset/multi30k-dataset/data/text-clean/test.2017.en"
# tsv_path = f"/opt/data/private/dsy/project/dataset/multi30k-dataset/data/speech/format_data/fr-en/test.2017.tsv"
# ref_txt = f"/opt/data/private/dsy/project/dataset/multi30k-dataset/data/text-clean/test.coco.en"
# tsv_path = f"/opt/data/private/dsy/project/dataset/multi30k-dataset/data/speech/format_data/fr-en/test.coco.tsv"

tsv = pd.read_csv(tsv_path, sep='\t')
ref_id_list = tsv["id"].tolist()
ref_list = []
with open(ref_txt, mode="r+") as f:
    for line in f.readlines():
        line = line.strip()
        if len(line) == 0:
            continue
        ref_list.append(line)
hyp_list = []
hyp_ref_list = []
with open(transcript_txt_path, mode="r+") as f:
    for line in f.readlines():
        line = line.strip()
        if len(line) == 0:
            continue
        hyp_list.append(line)
        hyp_ref_list.append([line, ref_list[ref_id_list[len(hyp_list) - 1] - 1]])


def remove_end_punc(line):
    if line.endswith(" ."):
        line = line[: len(line) - 2]
    return line

# print(len(ref_list), len(hyp_list))
assert len(ref_list) == len(hyp_list)
for i in range(len(ref_list)):
    hyp, ref = hyp_ref_list[i]
    hyp = english_cleaners(hyp)
    ref = english_cleaners(ref)
    hyp, ref = remove_end_punc(hyp), remove_end_punc(ref)
    hyp_ref_list[i] = [hyp, ref]
    print(ref_id_list[i])
    print("hyp: ", hyp)
    print("ref: ", ref)
    print()
bleu_score = sacrebleu.corpus_bleu(
    [hyp for hyp, _ in hyp_ref_list], 
    # [ref for _, ref in hyp_ref_list], 
    [[ref for _, ref in hyp_ref_list]]
)
print(bleu_score)