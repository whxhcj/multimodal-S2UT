#! /bin/bash

export CUDA_VISIBLE_DEVICES=0

DATA_ROOT=/root/autodl-tmp/liuwenrui/project/dataset/multi30k-dataset/data/speech/format_data/fr-en_enhanced
TASK=multimodal_speech_to_text
config_yaml_path=/root/autodl-tmp/liuwenrui/project/model/multimodal_S2UT/mm_s2ut/config/xm_transformer.yaml
multimodal_translation_config_yaml=/root/autodl-tmp/liuwenrui/project/model/multimodal_S2UT/mm_s2ut/config/multimodal_s2ut_transformer.yaml
GEN_SUBSET=valid

MODEL_DIR=/root/autodl-tmp/liuwenrui/project/model/multimodal_S2UT/mm_s2ut/checkpoints/enhanced_fr-en
# MODEL_DIR=/root/autodl-tmp/liuwenrui/project/model/multimodal_S2UT/mm_s2ut/checkpoints/enhanced_fr-en_mm
# MODEL_DIR=/root/autodl-tmp/liuwenrui/project/model/multimodal_S2UT/mm_s2ut/checkpoints/enhanced_fr-en_mm_dropout

MODEL_PATH=$MODEL_DIR/checkpoint_best.pt
RESULTS_PATH=$MODEL_DIR/inference

CODE_ROOT=/root/autodl-tmp/liuwenrui/project/model/multimodal_S2UT/mm_s2ut
# --user-dir $CODE_ROOT
# --multitask-config-yaml config_multitask.yaml

if [ ! -d $RESULTS_PATH ]; then
  mkdir -p $RESULTS_PATH
fi
fairseq-generate $DATA_ROOT \
  --config-yaml $config_yaml_path \
  --task $TASK  \
  --path $MODEL_PATH  --gen-subset $GEN_SUBSET \
  --max-tokens 10000 --max-source-positions 10000 --max-target-positions 10000\
  --beam 10 --max-len-a 1 --max-len-b 200 \
  --required-batch-size-multiple 1 \
  --user-dir $CODE_ROOT \
  --multimodal-translation-config-yaml $multimodal_translation_config_yaml \
  --results-path ${RESULTS_PATH}

# --multimodal-translation-config-yaml $multimodal_translation_config_yaml
