#! /bin/bash
# modify /opt/data/private/dsy/project/model/multimodal_S2UT/fairseq/examples/speech_to_speech/preprocessing/prep_s2ut_data.py

# subset=train
# subset=valid
# subset=test.2016
# subset=test.2017
subset=test.coco
lang=es

wav_root=/opt/data/private/dsy/project/dataset/multi30k-dataset/data/speech/16khz_wav/es

manifest_root=/opt/data/private/dsy/project/dataset/multi30k-dataset/data/speech/manifest
manifest_dir=$manifest_root/$subset.$lang
cp -ar $manifest_dir/$subset.$lang.txt $manifest_dir/$subset.txt

DATA_ROOT=/opt/data/private/dsy/project/dataset/multi30k-dataset/data/speech/format_data/es-en/es_source_unit

VOCODER_CKPT=/opt/data/private/dsy/project/checkpoint/unit_hifigan_mhubert_vp_en_es_fr_it3_400k_layer11_km1000_lj_dur/model.pt
VOCODER_CFG=/opt/data/private/dsy/project/checkpoint/unit_hifigan_mhubert_vp_en_es_fr_it3_400k_layer11_km1000_lj_dur/config.json

fairseq_root=/opt/data/private/dsy/project/model/multimodal_S2UT/fairseq
cd $fairseq_root
PYTHONPATH=:$fairseq_root python examples/speech_to_speech/preprocessing/prep_s2ut_data.py \
  --source-dir $wav_root --target-dir $manifest_dir --data-split $subset \
  --output-root $DATA_ROOT --reduce-unit \
  --vocoder-checkpoint $VOCODER_CKPT --vocoder-cfg $VOCODER_CFG

rm -rf $manifest_dir/$subset.txt
