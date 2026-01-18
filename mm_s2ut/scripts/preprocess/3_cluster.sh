#! /bin/bash

# subset=train
# subset=valid
# subset=test.2016
# subset=test.2017
subset=test.coco
lang=es

manifest_root=/opt/data/private/dsy/project/dataset/multi30k-dataset/data/speech/manifest
manifest_dir=$manifest_root/$subset.$lang
MANIFEST=$manifest_dir/$subset.$lang.tsv
OUT_QUANTIZED_FILE=$manifest_dir/$subset.$lang.txt

KM_MODEL_PATH=/opt/data/private/dsy/project/checkpoint/mHuBERT/mhubert_base_vp_en_es_fr_it3_L11_km1000.bin
CKPT_PATH=/opt/data/private/dsy/project/checkpoint/mHuBERT/mhubert_base_vp_en_es_fr_it3.pt

fairseq_root=/opt/data/private/dsy/project/model/multimodal_S2UT/fairseq
cd $fairseq_root
PYTHONPATH=:$fairseq_root python examples/textless_nlp/gslm/speech2unit/clustering/quantize_with_kmeans.py \
    --feature_type hubert \
    --kmeans_model_path $KM_MODEL_PATH \
    --acoustic_model_path $CKPT_PATH \
    --layer 11 \
    --manifest_path $MANIFEST  \
    --out_quantized_file_path $OUT_QUANTIZED_FILE \
    --extension .wav