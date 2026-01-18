from pathlib import Path
import torch
import numpy as np
import wavfile
import soundfile as sf
from typing import BinaryIO, List, Optional, Tuple, Union
# from speechbrain.processing.signal_processing import (
#     compute_amplitude,
#     dB_to_amplitude,
#     convolve1d,
#     notch_filter,
#     reverberate,
# )
from fairseq.data.audio.audio_utils import (
    convert_waveform, 
    get_features_or_waveform_from_stored_zip, 
    parse_path, 
    _get_kaldi_fbank, 
    _get_torchaudio_fbank, 
)
from fairseq.data.audio.waveform_transforms import CompositeAudioWaveformTransform

SF_AUDIO_FILE_EXTENSIONS = {".wav", ".flac", ".ogg"}
FEATURE_OR_SF_AUDIO_FILE_EXTENSIONS = {".npy", ".wav", ".flac", ".ogg"}


def select_noise(noise_wav, noise_num):
    # rand_indexes = torch.randint(0, len(noise_wav), size=(noise_num, ))
    # rand_indexes = rand_indexes.cpu().numpy()
    rand_indexes = np.random.randint(0, len(noise_wav), size=noise_num)
    noise_wav_np = []
    for x in rand_indexes:
        x = int(x)
        # noise_wav_np.append(wavfile.read(noise_wav[x])[1].astype(np.float32))
        noise_wav_np.append(sf.read(noise_wav[x])[0].astype(np.float32)[:,None])
    if noise_num == 1:
        return noise_wav_np[0]
    else:
        min_len = min([len(x) for x in noise_wav_np])
        noise_wav_np = [x[:min_len] for x in noise_wav_np]
        noise_wav_np = np.floor(np.stack(noise_wav_np).mean(axis=0))
        return noise_wav_np


def add_noise(clean_wav, noise_snr, noise_wav=None, select_noise_func=None):
    clean_wav = clean_wav.astype(np.float32)
    assert (noise_wav is not None) ^ (select_noise_func is not None)
    if noise_wav is not None and select_noise_func is None:
        pass
    elif noise_wav is None and select_noise_func is not None:
        noise_wav = select_noise_func()
    if type(noise_snr) == int or type(noise_snr) == float:
        snr = noise_snr
    elif type(noise_snr) == tuple or type(noise_snr) == list:
        # snr = torch.randint(noise_snr[0], noise_snr[1] + 1, size=(1,)).cpu().item()
        snr = np.random.randint(noise_snr[0], noise_snr[1] + 1)
    else:
        snr = np.random.randint(noise_snr[0], noise_snr[1] + 1)
    clean_rms = np.sqrt(np.mean(np.square(clean_wav), axis=-1))
    if len(clean_wav) > len(noise_wav):
        ratio = int(np.ceil(len(clean_wav) / len(noise_wav)))
        noise_wav = np.concatenate([noise_wav for _ in range(ratio)])
    if len(clean_wav) < len(noise_wav):
        start = 0
        noise_wav = noise_wav[start: start + len(clean_wav)]
    noise_rms = np.sqrt(np.mean(np.square(noise_wav), axis=-1))
    adjusted_noise_rms = clean_rms / (10 ** (snr / 20))
    adjusted_noise_wav = noise_wav * (adjusted_noise_rms / noise_rms)
    mixed = clean_wav + adjusted_noise_wav
    print(666, "clean_wav", clean_wav.shape, clean_wav)
    print(666, "adjusted_noise_wav", adjusted_noise_wav.shape, adjusted_noise_wav)
    print(666, "mixed", mixed.shape, mixed)
    # Avoid clipping noise
    max_int16 = np.iinfo(np.int16).max
    min_int16 = np.iinfo(np.int16).min
    # if mixed.max(axis=0) > max_int16 or mixed.min(axis=0) < min_int16:
    if (mixed.max(axis=0) > max_int16).any() or (mixed.min(axis=0) < min_int16).any():
        # if mixed.max(axis=0) >= abs(mixed.min(axis=0)): 
        if (mixed.max(axis=0) >= abs(mixed.min(axis=0))).any(): 
            reduction_rate = max_int16 / mixed.max(axis=0)
        else:
            reduction_rate = min_int16 / mixed.min(axis=0)
        mixed = mixed * (reduction_rate)
    # mixed = mixed.astype(np.int16)
    print(666, "mixed", mixed.shape, mixed)
    return mixed


def compute_amplitude(waveforms, lengths=None, amp_type="avg", scale="linear"):
    """Compute amplitude of a batch of waveforms.

    Arguments
    ---------
    waveform : tensor
        The waveforms used for computing amplitude.
        Shape should be `[time]` or `[batch, time]` or
        `[batch, time, channels]`.
    lengths : tensor
        The lengths of the waveforms excluding the padding.
        Shape should be a single dimension, `[batch]`.
    amp_type : str
        Whether to compute "avg" average or "peak" amplitude.
        Choose between ["avg", "peak"].
    scale : str
        Whether to compute amplitude in "dB" or "linear" scale.
        Choose between ["linear", "dB"].

    Returns
    -------
    The average amplitude of the waveforms.

    Example
    -------
    >>> signal = torch.sin(torch.arange(16000.0)).unsqueeze(0)
    >>> compute_amplitude(signal, signal.size(1))
    tensor([[0.6366]])
    """
    if len(waveforms.shape) == 1:
        waveforms = waveforms.unsqueeze(0)

    assert amp_type in ["avg", "peak"]
    assert scale in ["linear", "dB"]

    if amp_type == "avg":
        if lengths is None:
            out = torch.mean(torch.abs(waveforms), dim=1, keepdim=True)
        else:
            wav_sum = torch.sum(input=torch.abs(waveforms), dim=1, keepdim=True)
            out = wav_sum / lengths
    elif amp_type == "peak":
        out = torch.max(torch.abs(waveforms), dim=1, keepdim=True)[0]
    else:
        raise NotImplementedError

    if scale == "linear":
        return out
    elif scale == "dB":
        return torch.clamp(20 * torch.log10(out), min=-80)  # clamp zeros
    else:
        raise NotImplementedError
    

def dB_to_amplitude(SNR):
    """Returns the amplitude ratio, converted from decibels.

    Arguments
    ---------
    SNR : float
        The ratio in decibels to convert.

    Example
    -------
    >>> round(dB_to_amplitude(SNR=10), 3)
    3.162
    >>> dB_to_amplitude(SNR=0)
    1.0
    """
    return 10 ** (SNR / 20)


def add_noise_v2(
    waveforms: torch.Tensor, 
    noise_waveform: torch.Tensor, 
    snr_low: Optional[int], 
    snr_high: Optional[int], 
    noise_waveform_start: Optional[int] = 0, 
    add_white_noise: Optional[bool] = False, 
    normalize: Optional[bool] = True, 
):
    """
    Arguments
    ---------
    waveforms : tensor
        Shape should be `[batch, time]` or `[batch, time, channels]`.
    lengths : tensor
        Shape should be a single dimension, `[batch]`.

    Returns
    -------
    Tensor of shape `[batch, time]` or `[batch, time, channels]`.
    """

    # Copy clean waveform to initialize noisy waveform
    noisy_waveform = waveforms.clone()
    lengths = torch.tensor([waveforms.shape[0]], device=waveforms.device)
    lengths = (lengths * waveforms.shape[1]).unsqueeze(1)

    # Compute the average amplitude of the clean waveforms
    clean_amplitude = compute_amplitude(waveforms, lengths)

    # Pick an SNR and use it to compute the mixture amplitude factors
    # SNR = torch.rand(len(waveforms), 1, device=waveforms.device)
    SNR = torch.rand(waveforms.shape[0], 1, device=waveforms.device)
    SNR = SNR * (snr_high - snr_low) + snr_low
    noise_amplitude_factor = 1 / (dB_to_amplitude(SNR) + 1)
    new_noise_amplitude = noise_amplitude_factor * clean_amplitude

    # Scale clean signal appropriately
    noisy_waveform *= 1 - noise_amplitude_factor

    # Loop through clean samples and create mixture
    if add_white_noise:
        white_noise = torch.randn_like(waveforms)
        noisy_waveform += new_noise_amplitude * white_noise
    else:
        # tensor_length = waveforms.shape[1]
        # noise_waveform, noise_length = self._load_noise(
        #     lengths, tensor_length,
        # )
        if waveforms.shape[1] > noise_waveform.shape[1]:
            ratio = int(np.ceil(waveforms.shape[1] / noise_waveform.shape[1]))
            noise_waveform = torch.cat([noise_waveform for _ in range(ratio)], dim=-1)
        if waveforms.shape[1] < noise_waveform.shape[1]:
            start = noise_waveform_start
            if noise_waveform_start < 0:
                start = torch.randint(0, noise_waveform.shape[1] - waveforms.shape[1], size=(1,)).cpu().item()
            noise_waveform = noise_waveform[:, start: start + waveforms.shape[1]]

        # Rescale and add
        noise_length = torch.tensor([noise_waveform.shape[0]], device=waveforms.device)
        noise_length = (noise_length * noise_waveform.shape[1]).unsqueeze(1)
        noise_amplitude = compute_amplitude(noise_waveform, noise_length)
        noise_waveform *= new_noise_amplitude / (noise_amplitude + 1e-14)
        noisy_waveform += noise_waveform

    # Normalizing to prevent clipping
    if normalize:
        abs_max, _ = torch.max(
            torch.abs(noisy_waveform), dim=1, keepdim=True
        )
        noisy_waveform = noisy_waveform / abs_max.clamp(min=1.0)

    return noisy_waveform


def get_waveform(
    waveform=None, sample_rate=None, 
    path_or_fp: Union[str, BinaryIO] = None,
    normalization: bool = True,
    mono: bool = True,
    frames: int = -1,
    start: int = 0,
    always_2d: bool = True,
    output_sample_rate: Optional[int] = None,
    normalize_volume: bool = False,
    waveform_transforms: Optional[CompositeAudioWaveformTransform] = None,
) -> Tuple[np.ndarray, int]:
    """Get the waveform and sample rate of a 16-bit WAV/FLAC/OGG Vorbis audio.

    Args:
        path_or_fp (str or BinaryIO): the path or file-like object
        normalization (bool): normalize values to [-1, 1] (Default: True)
        mono (bool): convert multi-channel audio to mono-channel one
        frames (int): the number of frames to read. (-1 for reading all)
        start (int): Where to start reading. A negative value counts from the end.
        always_2d (bool): always return 2D array even for mono-channel audios
        output_sample_rate (Optional[int]): output sample rate
        normalize_volume (bool): normalize volume
    Returns:
        waveform (numpy.ndarray): 1D or 2D waveform (channels x length)
        sample_rate (float): sample rate
    """
    assert (waveform is not None and sample_rate is not None) ^ (path_or_fp is not None), \
        "one and only one of (waveform, path_or_fp) is true"
    if waveform is not None and sample_rate is not None:
        pass
    elif path_or_fp is not None:
        if isinstance(path_or_fp, str):
            ext = Path(path_or_fp).suffix
            if ext not in SF_AUDIO_FILE_EXTENSIONS:
                raise ValueError(f"Unsupported audio format: {ext}")
        try:
            import soundfile as sf
        except ImportError:
            raise ImportError("Please install soundfile: pip install soundfile")
        waveform, sample_rate = sf.read(
            path_or_fp, dtype="float32", always_2d=True, frames=frames, start=start
        )
    else:
        raise NotImplementedError("one and only one of (waveform, path_or_fp) is true")
    waveform = waveform.T  # T x C -> C x T
    waveform, sample_rate = convert_waveform(
        waveform,
        sample_rate,
        normalize_volume=normalize_volume,
        to_mono=mono,
        to_sample_rate=output_sample_rate,
    )
    if not normalization:
        waveform *= 2**15  # denormalized to 16-bit signed integers
    if waveform_transforms is not None:
        waveform, sample_rate = waveform_transforms(waveform, sample_rate)
    if not always_2d:
        waveform = waveform.squeeze(axis=0)
    return waveform, sample_rate


def get_features_from_npy_or_audio(
    waveform=None, sample_rate=None, 
    path: Union[str, BinaryIO] = None,
    waveform_transforms=None
):
    assert (waveform is not None and sample_rate is not None) ^ (path is not None), \
        "one and only one of (waveform, path) is true"
    if waveform is not None and sample_rate is not None:
        return get_fbank(
            waveform=waveform, sample_rate=sample_rate,
            path_or_fp=None, waveform_transforms=waveform_transforms
        )
    elif path is not None:
        ext = Path(path).suffix
        if ext not in FEATURE_OR_SF_AUDIO_FILE_EXTENSIONS:
            raise ValueError(f'Unsupported file format for "{path}"')
        return (
            np.load(path)
            if ext == ".npy"
            else get_fbank(
                    waveform=None, sample_rate=None,
                    path_or_fp=path, waveform_transforms=waveform_transforms
                )
        )
    else:
        raise NotImplementedError("one and only one of (waveform, path) is true")


def get_fbank(
    waveform=None, sample_rate=None, 
    path_or_fp: Union[str, BinaryIO] = None, 
    n_bins=80, waveform_transforms=None
) -> np.ndarray:
    """Get mel-filter bank features via PyKaldi or TorchAudio. Prefer PyKaldi
    (faster CPP implementation) to TorchAudio (Python implementation). Note that
    Kaldi/TorchAudio requires 16-bit signed integers as inputs and hence the
    waveform should not be normalized."""
    assert (waveform is not None and sample_rate is not None) ^ (path_or_fp is not None), \
        "one and only one of (waveform, path_or_fp) is true"
    waveform, sample_rate = get_waveform(
        waveform=waveform, sample_rate=sample_rate, 
        path_or_fp=path_or_fp, normalization=False, waveform_transforms=waveform_transforms
    )
    features = _get_kaldi_fbank(waveform, sample_rate, n_bins)
    if features is None:
        features = _get_torchaudio_fbank(waveform, sample_rate, n_bins)
    if features is None:
        raise ImportError(
            "Please install pyKaldi or torchaudio to enable "
            "online filterbank feature extraction"
        )
    return features


def get_features_or_waveform(
    waveform=None, sample_rate=None, 
    path: Optional[str] = None, need_waveform=False, 
    use_sample_rate=None, waveform_transforms=None, 
):
    """Get speech features from .npy file or waveform from .wav/.flac file.
    The file may be inside an uncompressed ZIP file and is accessed via byte
    offset and length.

    Args:
        path (str): File path in the format of "<.npy/.wav/.flac path>" or
        "<zip path>:<byte offset>:<byte length>".
        need_waveform (bool): return waveform instead of features.
        use_sample_rate (int): change sample rate for the input wave file

    Returns:
        features_or_waveform (numpy.ndarray): speech features or waveform.
    """
    assert (waveform is not None and sample_rate is not None) ^ (path is not None), \
        "one and only one of (waveform, path) is true"
    if waveform is not None and sample_rate is not None:
        if need_waveform:
            return get_waveform(
                waveform=waveform, sample_rate=sample_rate, 
                path_or_fp=None,
                always_2d=False,
                output_sample_rate=use_sample_rate,
                waveform_transforms=waveform_transforms,
            )[0]
        return get_features_from_npy_or_audio(
            waveform=waveform, sample_rate=sample_rate, 
            path=None, waveform_transforms=waveform_transforms
        )
    elif path is not None:
        _path, slice_ptr = parse_path(path)
        if len(slice_ptr) == 0:
            if need_waveform:
                return get_waveform(
                    waveform=None, sample_rate=None, 
                    path_or_fp=_path,
                    always_2d=False,
                    output_sample_rate=use_sample_rate,
                    waveform_transforms=waveform_transforms,
                )[0]
            return get_features_from_npy_or_audio(
                waveform=None, sample_rate=None, 
                path=_path, waveform_transforms=waveform_transforms
            )
        elif len(slice_ptr) == 2:
            features_or_waveform = get_features_or_waveform_from_stored_zip(
                _path,
                slice_ptr[0],
                slice_ptr[1],
                need_waveform=need_waveform,
                use_sample_rate=use_sample_rate,
                waveform_transforms=waveform_transforms,
            )
        else:
            raise ValueError(f"Invalid path: {path}")
    else:
        raise NotImplementedError("one and only one of (waveform, path_or_fp) is true")
    return features_or_waveform
