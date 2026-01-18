import torch, torchaudio
import pydub
import numpy as np
import fairseq
from typing import *


def collate_embeddings(tensor_list: List[torch.Tensor], pad=0.0):
    lengths = [tensor.shape[0] for tensor in tensor_list]
    max_length = max(lengths)
    padding_mask = torch.zeros(size=(len(tensor_list), max_length))
    for i in range(len(tensor_list)):
        length = lengths[i]
        if length < max_length:
            # padding_mask[i, length:] = True
            padding_mask[i, length:] = 1
            if len(tensor_list[i].shape) == 1 or tensor_list[i].shape[1] == 1:
                tensor_list[i] = torch.cat((tensor_list[i], tensor_list[i].new_full(size=(max_length - length, ), fill_value=pad)), dim=0)
            else:
                embed_dim = tensor_list[i].shape[1]
                tensor_list[i] = torch.cat((tensor_list[i], tensor_list[i].new_full(size=(max_length - length, embed_dim), fill_value=pad)), dim=0)
    tensor_list = torch.stack(tensor_list, dim=0)
    padding_mask = padding_mask == 1
    return tensor_list, padding_mask, 


def get_audio_from_path(path_list: List, backend: str = "pydub") -> torch.Tensor:
    assert backend in ("pydub", "torchaudio")
    source = []
    for audio_path in path_list:
        if backend == "pydub":
            audio = pydub.AudioSegment.from_wav(audio_path)
            data = audio.get_array_of_samples()
            data = torch.tensor(np.array(data))
            source.append(data)
        elif backend == "torchaudio":
            data, sr = torchaudio.load(audio_path)
            source.append(data[0])
    source, padding_mask = collate_embeddings(source)
    source = source.float()
    return source, padding_mask, 
