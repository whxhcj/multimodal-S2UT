#! /bin/bash

export CUDA_VISIBLE_DEVICES=0
fairseq_root=/opt/data/private/dsy/project/model/multimodal_S2UT/fairseq
GEN_SUBSET=valid
VOCODER_CKPT=/opt/data/private/dsy/project/checkpoint/unit_hifigan_mhubert_vp_en_es_fr_it3_400k_layer11_km1000_lj_dur/model.pt
VOCODER_CFG=/opt/data/private/dsy/project/checkpoint/unit_hifigan_mhubert_vp_en_es_fr_it3_400k_layer11_km1000_lj_dur/config.json

# MODEL_DIR=/root/autodl-tmp/liuwenrui/project/model/multimodal_S2UT/mm_s2ut/checkpoints/enhanced_fr-en
# MODEL_DIR=/opt/data/private/dsy/project/model/multimodal_S2UT/mm_s2ut/checkpoints/textless_fr-en
# MODEL_DIR=/opt/data/private/dsy/project/model/multimodal_S2UT/mm_s2ut/checkpoints/textless_fr-en_mm
MODEL_DIR=/opt/data/private/dsy/project/model/multimodal_S2UT/mm_s2ut/checkpoints/textless_fr-en_mm_dropout
# MODEL_DIR=/opt/data/private/dsy/project/model/multimodal_S2UT/mm_s2ut/checkpoints/textless_fr-en_mm_2
RESULTS_PATH=$MODEL_DIR/inference
generate_txt=$MODEL_DIR/inference/generate-valid.txt
tts_wav_dir=$MODEL_DIR/inference/tts

cd $fairseq_root

if [ ! -d $tts_wav_dir ]; then
  mkdir -p $tts_wav_dir
fi

grep "^D\-" $generate_txt | \
  sed 's/^D-//ig' | sort -nk1 | cut -f3 \
  > ${RESULTS_PATH}/generate-${GEN_SUBSET}.unit

python examples/speech_to_speech/generate_waveform_from_code.py \
  --in-code-file ${RESULTS_PATH}/generate-${GEN_SUBSET}.unit \
  --vocoder $VOCODER_CKPT --vocoder-cfg $VOCODER_CFG \
  --results-path ${tts_wav_dir} --dur-prediction