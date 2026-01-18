#! /bin/bash
# https://github.com/facebookresearch/fairseq/blob/main/examples/speech_to_speech/docs/direct_s2st_discrete_units.md

export CUDA_VISIBLE_DEVICES=0
gpu_num=1

# lang_task=fr-en
lang_task=es-en
# lang_task=en-fr
# lang_task=en-es
DATA_ROOT=/opt/data/private/dsy/project/dataset/multi30k-dataset/data/speech/format_data/${lang_task}
CKPT_DIR_ROOT=/opt/data/private/dsy/project/model/multimodal_S2UT/mm_s2ut/checkpoints

# MODEL_NAME=textless_${lang_task}
# MODEL_NAME=textless_${lang_task}_vit_2
# MODEL_NAME=textless_${lang_task}_ex_vit
# MODEL_NAME=textless_${lang_task}_wav2vec2_vit_ex_1
# MODEL_NAME=textless_${lang_task}_wav2vec2_vit_in_mask_3
# MODEL_NAME=textless_${lang_task}_wav2vec2_vit-patch16_2_2
# MODEL_NAME=textless_${lang_task}_wav2vec2_vit-patch16_3_0
# MODEL_NAME=textless_${lang_task}_wav2vec2_vit-patch16_qformer_3_0
# MODEL_NAME=textless_${lang_task}_wav2vec2_vit-openai
# MODEL_NAME=textless_${lang_task}_wav2vec2_vit-opanai_ex-6
# MODEL_NAME=textless_${lang_task}_wav2vec2
# MODEL_NAME=textless_${lang_task}_wav2vec2_music
# MODEL_NAME=textless_${lang_task}_caption
# MODEL_NAME=textless_${lang_task}_wav2vec2_music_qformer_3_3
MODEL_NAME=textless_${lang_task}_qformer_3_3
MODEL_DIR=$CKPT_DIR_ROOT/$MODEL_NAME

config_yaml=/opt/data/private/dsy/project/dataset/multi30k-dataset/data/speech/format_data/${lang_task}/config.yaml
config_multitask_yaml=/opt/data/private/dsy/project/dataset/multi30k-dataset/data/speech/format_data/${lang_task}/config_multitask.yaml
multimodal_translation_config_yaml=/opt/data/private/dsy/project/dataset/multi30k-dataset/data/speech/format_data/${lang_task}/multimodal_s2ut_transformer.yaml

mhubert_ckpt_path=/opt/data/private/dsy/project/checkpoint/mHuBERT/mhubert_base_vp_en_es_fr_it3.pt
# --mhubert-ckpt-path $mhubert_ckpt_path
# wav2vec2_model_dir=/opt/data/private/dsy/project/checkpoint/wav2vec2-base-10k-voxpopuli-ft-fr
wav2vec2_model_dir=/opt/data/private/dsy/project/checkpoint/wav2vec2-base-10k-voxpopuli-ft-es
# wav2vec2_model_dir=/opt/data/private/dsy/project/checkpoint/wav2vec2-base-960h
# --wav2vec2-model-dir $wav2vec2_model_dir
# use_audio_input

PROJECT_ROOT=/opt/data/private/dsy/project/model/multimodal_S2UT
CODE_ROOT=$PROJECT_ROOT/mm_s2ut
ARCH=mm_s2ut_transformer
TASK=multimodal_speech_to_speech
# ARCH=s2ut_transformer
# TASK=speech_to_speech
# max_tokens=160000 # textless at A100, may be 28_000 best?
# max_tokens=200000 # textless at A100, may be 28_000 best?
# max_tokens=280000 # textless at A100, may be 28_000 best?
# max_tokens=15000

# max_tokens=48000
# max_tokens=16000
max_tokens=8000
lr=0.0005
update_freq=16

max_update=400000
# max_update=5500
# max_update=6500
# max_update=11000

# best
# max_tokens=4000
# lr=0.0002
# update_freq=8

# copy config, data, source code, etc.
if [ ! -d $MODEL_DIR ]; then
  mkdir -p $MODEL_DIR
fi
cp ${BASH_SOURCE[0]} $MODEL_DIR/train.sh

if [ ! -d $MODEL_DIR/data ]; then
  mkdir -p $MODEL_DIR/data
fi
cp $DATA_ROOT -ar $MODEL_DIR/data/
DATA_ROOT=$MODEL_DIR/data/${lang_task}
multimodal_translation_config_yaml=$DATA_ROOT/multimodal_s2ut_transformer.yaml
noise_config_yaml=$DATA_ROOT/noise.yaml

if [ ! -d $MODEL_DIR/code ]; then
  mkdir -p $MODEL_DIR/code
fi
cp -ar ${PROJECT_ROOT}/fairseq $MODEL_DIR/code/

if [ ! -d $MODEL_DIR/code/mm_s2ut ]; then
  mkdir -p $MODEL_DIR/code/mm_s2ut
fi
cd ${PROJECT_ROOT}/mm_s2ut
cp -ar `ls | grep -v checkpoints | xargs` $MODEL_DIR/code/mm_s2ut
CODE_ROOT=$MODEL_DIR/code/mm_s2ut
cp $CODE_ROOT/scripts/textless/2_inference_all.sh $MODEL_DIR/2_inference_all.sh

if [ ! -d $MODEL_DIR/checkpoints ]; then
  mkdir -p $MODEL_DIR/checkpoints
fi

echo DATA_ROOT=$DATA_ROOT
echo user-dir=$CODE_ROOT
echo save-dir=$MODEL_DIR/checkpoints

cmd="fairseq-train $DATA_ROOT
  --distributed-world-size $gpu_num
  --tensorboard-logdir $MODEL_DIR/checkpoints
  --config-yaml config.yaml 
  --task $TASK --target-is-code --target-code-size 1000 --vocoder code_hifigan
  --criterion speech_to_unit --label-smoothing 0.2
  --arch $ARCH --share-decoder-input-output-embed
  --dropout 0.1 --attention-dropout 0.1 --relu-dropout 0.1
  --train-subset train --valid-subset valid
  --save-dir $MODEL_DIR/checkpoints
  --lr $lr --lr-scheduler inverse_sqrt --warmup-init-lr 1e-7 --warmup-updates 10000
  --optimizer adam --adam-betas '(0.9,0.98)' --clip-norm 10.0
  --max-update $max_update --max-tokens $max_tokens --max-target-positions 3000 --update-freq $update_freq
  --required-batch-size-multiple 1
  --multitask-config-yaml config_multitask.yaml
  --multimodal-translation-config-yaml $multimodal_translation_config_yaml
  --encoder-embed-dim 768
  --encoder-ffn-embed-dim 3072
  --gen-subset test
  --user-dir $CODE_ROOT
  --seed 1 --fp16 --num-workers 8"
# --multitask-config-yaml config_multitask.yaml
# --multimodal-translation-config-yaml $multimodal_translation_config_yaml
# --noise-config-yaml $noise_config_yaml
# --mhubert-ckpt-path $mhubert_ckpt_path
# --wav2vec2-model-dir $wav2vec2_model_dir
# --encoder-embed-dim 768
# --encoder-ffn-embed-dim 3072
# --no-epoch-checkpoints
# --gen-subset test.2016,test.2017,test.coco
# --freezing-updates 0
# --encoder-embed-dim 512
# --encoder-ffn-embed-dim 2048

cmd="nohup "${cmd}" > $MODEL_DIR/train.log 2>&1 &"
eval $cmd
tail -f $MODEL_DIR/train.log
