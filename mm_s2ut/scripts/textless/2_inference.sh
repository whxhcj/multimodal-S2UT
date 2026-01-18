#! /bin/bash

export CUDA_VISIBLE_DEVICES=0
DATA_ROOT=/opt/data/private/dsy/project/dataset/multi30k-dataset/data/speech/format_data/fr-en
GEN_SUBSET=valid
# GEN_SUBSET=test.coco
# GEN_SUBSET=train

# MODEL_DIR=/opt/data/private/dsy/project/model/multimodal_S2UT/mm_s2ut/checkpoints/textless_fr-en
# MODEL_DIR=/opt/data/private/dsy/project/model/multimodal_S2UT/mm_s2ut/checkpoints/textless_fr-en_mm
MODEL_DIR=/opt/data/private/dsy/project/model/multimodal_S2UT/mm_s2ut/checkpoints/textless_fr-en_mm_dropout
# MODEL_DIR=/opt/data/private/dsy/project/model/multimodal_S2UT/mm_s2ut/checkpoints/textless_fr-en_mm_2
MODEL_PATH=$MODEL_DIR/checkpoint_best.pt
RESULTS_PATH=$MODEL_DIR/inference

multimodal_translation_config_yaml=/opt/data/private/dsy/project/model/multimodal_S2UT/mm_s2ut/config/multimodal_s2ut_transformer.yaml
mhubert_ckpt_path=/opt/data/private/dsy/project/checkpoint/wav2vec2-FR-7K-base/checkpoint_best.pt
# mhubert_ckpt_path=/opt/data/private/dsy/project/checkpoint/wav2vec2-FR-7K-base/checkpoint_best.pt
# --mhubert-ckpt-path $wav2vec2_ckpt_path \
wav2vec2_model_dir=/opt/data/private/dsy/project/checkpoint/wav2vec2-base-fr-voxpopuli-v2
# --wav2vec2-model-dir $wav2vec2_model_dir \
CODE_ROOT=/opt/data/private/dsy/project/model/multimodal_S2UT/mm_s2ut

TASK=multimodal_speech_to_speech
# TASK=speech_to_speech
# max_tokens=16000
# max_tokens=4000
max_tokens=8000
# max_tokens=280000

if [ ! -d $RESULTS_PATH ]; then
  mkdir -p $RESULTS_PATH
fi
fairseq-generate $DATA_ROOT \
  --config-yaml config.yaml \
  --task $TASK --target-is-code --target-code-size 1000 --vocoder code_hifigan \
  --path $MODEL_PATH  --gen-subset $GEN_SUBSET \
  --max-tokens $max_tokens \
  --beam 10 --max-len-a 1 \
  --required-batch-size-multiple 1 \
  --multitask-config-yaml config_multitask.yaml \
  --multimodal-translation-config-yaml $multimodal_translation_config_yaml \
  --user-dir $CODE_ROOT \
  --results-path ${RESULTS_PATH}

  # --multitask-config-yaml config_multitask.yaml \
  # --multimodal-translation-config-yaml $multimodal_translation_config_yaml \
  # --mhubert-ckpt-path $mhubert_ckpt_path \