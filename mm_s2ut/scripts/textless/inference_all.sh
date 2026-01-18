#! /bin/bash

# cd /mnt/liuwenrui/dataset/multi30k-dataset/data/speech/format_data/fr-en_enhanced/fr_source_unit
# SPLIT=train
# var="id\taudio\tn_frames\ttgt_text\ttgt_n_frames"
# sed -i "1s/.*/$var/" ${SPLIT}.tsv

bash /opt/data/private/dsy/project/model/multimodal_S2UT/mm_s2ut/scripts/textless/2_inference.sh
bash /opt/data/private/dsy/project/model/multimodal_S2UT/mm_s2ut/scripts/3_generate_waveform.sh
python /opt/data/private/dsy/project/model/multimodal_S2UT/mm_s2ut/scripts/4_transcript.py