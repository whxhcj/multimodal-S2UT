#! /bin/bash

subset=train
# subset=valid
# subset=test.2016
# subset=test.2017
# subset=test.coco
lang=es

wav_root=/opt/data/private/dsy/project/dataset/multi30k-dataset/data/speech/16khz_wav/$lang
wav_dir=$wav_root/$subset

manifest_root=/opt/data/private/dsy/project/dataset/multi30k-dataset/data/speech/manifest
manifest_dir=$manifest_root/$subset.$lang
if [ ! -d $manifest_dir ]; then
  mkdir -p $manifest_dir
fi

ext=wav
fairseq_root=/opt/data/private/dsy/project/model/multimodal_S2UT/fairseq
cd $fairseq_root
PYTHONPATH=:$fairseq_root python examples/wav2vec/wav2vec_manifest.py $wav_dir \
    --dest $manifest_dir \
    --ext $ext \
    --valid-percent 0

mv $manifest_dir/train.tsv $manifest_dir/$subset.$lang.tsv
