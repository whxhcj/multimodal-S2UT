#! /bin/bash
# https://github.com/facebookresearch/fairseq/blob/main/examples/speech_to_speech/docs/enhanced_direct_s2st_discrete_units.md#training

# export CUDA_VISIBLE_DEVICES=0
# gpu_num=1
export CUDA_VISIBLE_DEVICES=0
gpu_num=1

# max_tokens=4000
# max_tokens_valid=4000
max_tokens=8000
max_tokens_valid=8000

DATA_ROOT=/root/autodl-tmp/liuwenrui/project/dataset/multi30k-dataset/data/speech/format_data/fr-en_enhanced
config_yaml_path=/root/autodl-tmp/liuwenrui/project/model/multimodal_S2UT/mm_s2ut/config/xm_transformer.yaml
multimodal_translation_config_yaml=/root/autodl-tmp/liuwenrui/project/model/multimodal_S2UT/mm_s2ut/config/multimodal_s2ut_transformer.yaml
wav2vec2_ckpt_path=/root/autodl-tmp/liuwenrui/project/model/transformer_ckpt/wav2vec2-FR-7K-large/checkpoint_best.pt
# --w2v-path ${wav2vec2_ckpt_path}
# --normalize for some kinds of wav2vec2 checkpoints

MODEL_DIR=/root/autodl-tmp/liuwenrui/project/model/multimodal_S2UT/mm_s2ut/checkpoints/enhanced_fr-en
# MODEL_DIR=/root/autodl-tmp/liuwenrui/project/model/multimodal_S2UT/mm_s2ut/checkpoints/enhanced_fr-en_mm
# MODEL_DIR=/root/autodl-tmp/liuwenrui/project/model/multimodal_S2UT/mm_s2ut/checkpoints/enhanced_fr-en_mm_dropout

CODE_ROOT=/root/autodl-tmp/liuwenrui/project/model/multimodal_S2UT/mm_s2ut
# --user-dir $CODE_ROOT
# --multitask-config-yaml config_multitask.yaml

# TASK=speech_to_text
# ARCH=xm_transformer

TASK=multimodal_speech_to_text
ARCH=mm_xm_transformer

if [ ! -d $MODEL_DIR ]; then
  mkdir -p $MODEL_DIR
fi
cp ${BASH_SOURCE[0]} $MODEL_DIR/train.sh

cmd="fairseq-train $DATA_ROOT
  --config-yaml $config_yaml_path
  --task $TASK --arch $ARCH
  --criterion speech_to_unit --label-smoothing 0.2
  --share-decoder-input-output-embed --adaptor-n-layers 1
  --dropout 0.1 --attention-dropout 0.1 --relu-dropout 0.1
  --train-subset train --valid-subset valid
  --mask-prob 0.3 --mask-channel-length 32 --mask-channel-prob 0.25
  --save-dir ${MODEL_DIR} --checkpoint-activations --encoder-proj
  --lr 0.0005 --dropout 0.1 --attention-dropout 0.1 --lr-scheduler inverse_sqrt
  --warmup-init-lr 1e-7 --warmup-updates 10000
  --optimizer adam --adam-betas '(0.9,0.98)' --clip-norm 10.0
  --max-update 20000 --max-tokens $max_tokens --max-tokens-valid $max_tokens_valid --max-source-positions 4000
  --max-target-positions 4000 --update-freq 16
  --required-batch-size-multiple 1
  --tensorboard-logdir $MODEL_DIR
  --w2v-path ${wav2vec2_ckpt_path}
  --normalize
  --apply-mask
  --multimodal-translation-config-yaml $multimodal_translation_config_yaml
  --user-dir $CODE_ROOT
  --seed 1 --fp16 --num-workers $gpu_num"

cmd="nohup "${cmd}" > $MODEL_DIR/train.log 2>&1 &"
eval $cmd
tail -f $MODEL_DIR/train.log
