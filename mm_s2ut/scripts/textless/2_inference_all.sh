#! /bin/bash

export CUDA_VISIBLE_DEVICES=0

# lang_task=fr-en
# lang_task=es-en
lang_task=en-fr
# lang_task=en-es

MODEL_NAME=textless_${lang_task}
CKPT_DIR_ROOT=/opt/data/private/dsy/project/model/multimodal_S2UT/mm_s2ut/checkpoints

# config for inference
TASK=multimodal_speech_to_speech
# TASK=speech_to_speech
# max_tokens=16000
# max_tokens=4000
max_tokens=8000
# max_tokens=280000

MODEL_DIR=$CKPT_DIR_ROOT/$MODEL_NAME
DATA_ROOT=$MODEL_DIR/data/${lang_task}
CODE_ROOT=$MODEL_DIR/code/mm_s2ut
saved_dir=$MODEL_DIR/checkpoints
MODEL_PATH=$saved_dir/checkpoint_best.pt
RESULTS_PATH=$MODEL_DIR/inference
multimodal_translation_config_yaml=$DATA_ROOT/multimodal_s2ut_transformer.yaml

if [ ! -d $RESULTS_PATH ]; then
  mkdir -p $RESULTS_PATH
fi

# config for generate waveform
fairseq_root=$MODEL_DIR/code/fairseq
# VOCODER_CKPT=/opt/data/private/dsy/project/checkpoint/unit_hifigan_mhubert_vp_en_es_fr_it3_400k_layer11_km1000_lj_dur/model.pt
# VOCODER_CFG=/opt/data/private/dsy/project/checkpoint/unit_hifigan_mhubert_vp_en_es_fr_it3_400k_layer11_km1000_lj_dur/config.json
VOCODER_CKPT=/opt/data/private/dsy/project/checkpoint/unit_hifigan/fr/g_00500000
VOCODER_CFG=/opt/data/private/dsy/project/checkpoint/unit_hifigan/fr/config.json
# VOCODER_CKPT=/opt/data/private/dsy/project/checkpoint/unit_hifigan/es/g_00500000
# VOCODER_CFG=/opt/data/private/dsy/project/checkpoint/unit_hifigan/es/config.json

# config for transcript
# asr_model_path=/opt/data/private/dsy/project/checkpoint/wav2vec2-large-960h-lv60-self
asr_model_path=/opt/data/private/dsy/project/checkpoint/jonatasgrosman/wav2vec2-large-fr-voxpopuli-french
# asr_model_path=/opt/data/private/dsy/project/checkpoint/jonatasgrosman/wav2vec2-large-xlsr-53-spanish

# config for BLEU
src_lang=en
tgt_lang=fr
ref_root=/opt/data/private/dsy/project/dataset/multi30k-dataset/data/text-clean

gen_subsets=("valid" "test.2016" "test.2017" "test.coco")
# gen_subsets=("valid")

function inference() {
    echo "inferencing $GEN_SUBSET"
    mhubert_ckpt_path=/opt/data/private/dsy/project/checkpoint/mHuBERT/mhubert_base_vp_en_es_fr_it3.pt
    # --mhubert-ckpt-path $wav2vec2_ckpt_path \
    # wav2vec2_model_dir=/opt/data/private/dsy/project/checkpoint/wav2vec2-base-10k-voxpopuli-ft-es
    wav2vec2_model_dir=/opt/data/private/dsy/project/checkpoint/wav2vec2-base-10k-voxpopuli-ft-en
    # --wav2vec2-model-dir $wav2vec2_model_dir \

    if [ ! -d $RESULTS_PATH/$GEN_SUBSET ]; then
        mkdir -p $RESULTS_PATH/$GEN_SUBSET
    fi
    fairseq-generate $DATA_ROOT \
        --config-yaml config.yaml \
        --task $TASK --target-is-code --target-code-size 1000 --vocoder code_hifigan \
        --path $MODEL_PATH  --gen-subset $GEN_SUBSET \
        --max-tokens $max_tokens \
        --beam 10 --max-len-a 1 \
        --required-batch-size-multiple 1 \
        --multitask-config-yaml config_multitask.yaml \
        --user-dir $CODE_ROOT \
        --results-path $RESULTS_PATH/$GEN_SUBSET

    # --multitask-config-yaml config_multitask.yaml \
    # --multimodal-translation-config-yaml $multimodal_translation_config_yaml \
    # --mhubert-ckpt-path $mhubert_ckpt_path \
}


function generate_waveform() {
    echo "generating waveform for $GEN_SUBSET"
    generate_txt=$RESULTS_PATH/$GEN_SUBSET/generate-${GEN_SUBSET}.txt
    tts_wav_dir=$RESULTS_PATH/$GEN_SUBSET/tts
    if [ ! -d $tts_wav_dir ]; then
        mkdir -p $tts_wav_dir
    fi
    cd $fairseq_root
    grep "^D\-" $generate_txt | \
        sed 's/^D-//ig' | sort -nk1 | cut -f3 \
        > $RESULTS_PATH/$GEN_SUBSET/generate-${GEN_SUBSET}.unit
    python examples/speech_to_speech/generate_waveform_from_code.py \
        --in-code-file $RESULTS_PATH/$GEN_SUBSET/generate-${GEN_SUBSET}.unit \
        --vocoder $VOCODER_CKPT --vocoder-cfg $VOCODER_CFG \
        --results-path ${tts_wav_dir} --dur-prediction
}


function transcript() {
    echo "transcript for $GEN_SUBSET"
    python $MODEL_DIR/code/mm_s2ut/scripts/transcript.py \
        --model-path $asr_model_path \
        --tts-wav-dir $RESULTS_PATH/$GEN_SUBSET/tts \
        --transcript-txt $RESULTS_PATH/$GEN_SUBSET/tts_transcript.txt
}


function bleu_asr() {
    echo "compute BLEU for $GEN_SUBSET"
    python $MODEL_DIR/code/mm_s2ut/scripts/bleu_asr.py \
        --src-lang $src_lang --tgt-lang $tgt_lang --gen-subset $GEN_SUBSET \
        --transcript-txt-path $RESULTS_PATH/$GEN_SUBSET/tts_transcript.txt \
        --output-txt $RESULTS_PATH/$GEN_SUBSET/bleu.txt \
        --ref-txt $ref_root/$GEN_SUBSET.$tgt_lang \
        --tsv-path $DATA_ROOT/$GEN_SUBSET.tsv
    # tail -n 1 $RESULTS_PATH/$GEN_SUBSET/bleu.txt
    # bleu_score="tail -n 1 $RESULTS_PATH/$GEN_SUBSET/bleu.txt"
    # eval $bleu_score
    bleu_score=$(tail -n 1 $RESULTS_PATH/$GEN_SUBSET/bleu.txt)
    echo "$GEN_SUBSET: $bleu_score" >> $RESULTS_PATH/bleu.txt
}


for GEN_SUBSET in ${gen_subsets[@]}; do
    inference
    generate_waveform
    transcript
    bleu_asr
done
