#! /bin/bash

# cd /mnt/liuwenrui/dataset/multi30k-dataset/data/speech/format_data/fr-en_enhanced/fr_source_unit
# SPLIT=train
# var="id\taudio\tn_frames\ttgt_text\ttgt_n_frames"
# sed -i "1s/.*/$var/" ${SPLIT}.tsv

bash /root/autodl-tmp/liuwenrui/project/model/multimodal_S2UT/mm_s2ut/scripts/enhanced/2_inference.sh
bash /mnt/liuwenrui/model/multimodal_S2UT/mm_s2ut/scripts/3_generate_waveform.sh
python /mnt/liuwenrui/model/multimodal_S2UT/mm_s2ut/scripts/4_transcript.py