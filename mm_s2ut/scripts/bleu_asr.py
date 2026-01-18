import argparse
parser = argparse.ArgumentParser(description="compute BLEU score")
parser.add_argument("--src-lang", type=str)
parser.add_argument("--tgt-lang", type=str)
parser.add_argument("--gen-subset", type=str)
parser.add_argument("--transcript-txt-path", type=str)
parser.add_argument("--output-txt", type=str)
parser.add_argument("--ref-txt", type=str)
parser.add_argument("--tsv-path", type=str)
args = parser.parse_args()

import sys
f = open(args.output_txt, mode="w+")
sys.stdout = f

import os, json, tqdm
import pandas as pd
import sacrebleu
from sacrebleu.metrics import BLEU
from speech_to_speech_translation.text_cleaner.cleaners import (
    english_cleaners,
    transliteration_cleaners,
)

src_lang, tgt_lang = args.src_lang, args.tgt_lang
gen_subset = args.gen_subset
transcript_txt_path = args.transcript_txt_path

ref_txt = args.ref_txt
tsv_path = args.tsv_path

tsv = pd.read_csv(tsv_path, sep='\t')
ref_id_list = tsv["id"].tolist()
ref_list = []
with open(ref_txt, mode="r+") as f:
    for line in f.readlines():
        line = line.strip()
        if len(line) == 0:
            # continue
            line = ""
        ref_list.append(line)
hyp_list = []
hyp_ref_list = []
with open(transcript_txt_path, mode="r+") as f:
    for line in f.readlines():
        line = line.strip()
        if len(line) == 0:
            # continue
            line = ""
        hyp_list.append(line)
        hyp_ref_list.append([line, ref_list[ref_id_list[len(hyp_list) - 1] - 1]])


def remove_end_punc(line):
    if line.endswith(" ."):
        line = line[: len(line) - 2]
    if line.endswith("."):
        line = line[: len(line) - 1]
    return line


# print(len(ref_list), len(hyp_list))
assert len(ref_list) == len(hyp_list)
for i in range(len(ref_list)):
    hyp, ref = hyp_ref_list[i]
    if args.tgt_lang == "en":
        hyp = english_cleaners(hyp)
        ref = english_cleaners(ref)
    else:
        hyp = transliteration_cleaners(hyp)
        ref = transliteration_cleaners(ref)
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
