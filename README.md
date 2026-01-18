Multimodal Speech-to-Unit Translation (MM-S2UT)

<div align="center">





An End-to-End Speech-to-Speech Translation System Based on Multimodal Fusion

Features • Architecture • Installation & Setup • Usage Examples

</div>

---

📖 Overview

MM-S2UT (Multimodal Speech-to-Unit Translation) is a speech-to-speech translation system that fuses speech and visual information for more accurate and context-aware translation. Built on Fairseq framework with Transformer architecture, supporting end-to-end speech unit generation and translation.

Tech Stack: PyTorch, Fairseq, Hugging Face Transformers, timm, Wav2Vec2, HuBERT, Vision Transformer (ViT), DETR

---

✨ Features

- Multimodal speech translation with audio + image joint input
- Selective attention mechanism for dynamic multimodal fusion
- Multiple visual encoders: ViT, DETR
- Complete pipeline: data preprocessing, training, inference, evaluation
- Modality dropout for enhanced robustness
- Distributed training support
- Multi-metric evaluation: BLEU, ASR-BLEU, WER

---

🏗️ Project Architecture

System Architecture Overview

		The MM-S2UT system adopts an encoder-decoder architecture, fusing audio and visual features through selective attention mechanisms:

    graph TB
        A[Audio Input] --> B[Speech Encoder<br/>Wav2Vec2/HuBERT]
        C[Image Input] --> D[Visual Encoder<br/>ViT/DETR/ResNet]
        
        B --> E[Multimodal Fusion Layer<br/>Selective Attention]
        D --> E
        
        E --> F[Transformer Decoder]
        F --> G[Speech Unit Sequence]
        G --> H[Vocoder<br/>Unit-to-Waveform]
        H --> I[Output Speech]
        
        style E fill:#ff9900,stroke:#333,stroke-width:3px
        style B fill:#4a90e2,stroke:#333,stroke-width:2px
        style D fill:#4a90e2,stroke:#333,stroke-width:2px
        style F fill:#50c878,stroke:#333,stroke-width:2px

Core Modules

1. Speech Encoder

- Pre-trained models: Wav2Vec2, HuBERT, mHuBERT
- Extracts high-level semantic features from speech

2. Visual Encoder

- ViT (Vision Transformer): Patch-based visual representation
- DETR: Object detection features
- ResNet + Transformer Encoder: Convolution + attention

3. Multimodal Fusion Layer

- Selective Attention: Dynamically computes audio/visual feature weights
- Multimodal Attention: Cross-modal cross-attention

4. Decoder & Vocoder

- Transformer decoder generates target language speech units
- Neural vocoder converts units to waveforms

Data Flow Diagram

    flowchart LR
        A[Raw Data] --> B[Preprocessing]
        B --> C[Feature Extraction]
        C --> D[Training Data]
        
        D --> E[Model Training]
        E --> F[Checkpoint]
        
        F --> G[Inference]
        G --> H[Speech Units]
        H --> I[Waveform Generation]
        I --> J[Evaluation]
        
        style E fill:#ff6b6b,stroke:#333,stroke-width:2px
        style G fill:#4ecdc4,stroke:#333,stroke-width:2px

---

🚀 Installation & Setup

Requirements

- Operating System: Linux (Ubuntu 20.04+ recommended)
- Python Version: 3.10+
- CUDA Version: 12.0+ (Required for GPU training)

All other Python-specific dependencies are listed in requirements.txt.

Setup

1. Clone the Repository

Download the source code to your local machine or server.

2. Environment Setup

You can set up your environment using pip or Conda. Then create a environment either  install the requirements listed above manually or   using the requirements.txt file:

    pip install -r requirements.txt

Configuration

Main Configuration Files

The project provides two main configuration files:

1. mm_s2ut/config/multimodal_s2ut_transformer.yaml
   - Complete configuration for multimodal speech translation
   - Includes visual encoder, fusion strategy, and other parameters
2. mm_s2ut/config/xm_transformer.yaml
   - Cross-lingual model configuration

Key Configuration Parameters

    # Selective attention parameters
    SA_image_dropout: 0.1          # Image feature dropout rate
    SA_attention_dropout: 0.1      # Attention dropout rate
    use_selective_gate: True       # Whether to use selective gating
    
    image_feat_dim: [768]           # Image feature dimension

Data Preparation

    cd mm_s2ut/scripts/preprocess
    
    # Preprocess
    jupyter notebook 1_preprocess.ipynb
    
    # Generate manifest
    bash 2_manifest.sh
    
    # Cluster speech units
    bash 3_cluster.sh
    
    # Prepare S2UT data
    bash 5_prep_s2ut_data.sh
    
    # Extract image features
    cd ../extract_feature
    python get_img_feat_vit.py \
        --model vit_base_patch16_384 \
        --image_dir /path/to/images \
        --output_dir /path/to/output

Training

Enhanced Version (Recommended)

    cd mm_s2ut/scripts/enhanced
    bash 1_train.sh

Example training script:

    #!/bin/bash
    
    # Set environment variables
    export CUDA_VISIBLE_DEVICES=0,1,2,3
    
    fairseq-train /path/to/data \
        --config-yaml multimodal_s2ut_transformer.yaml \
        --task multimodal_speech_to_speech \
        --arch mm_s2ut_transformer \
        --criterion speech_to_speech \
        --optimizer adam \
        --lr 0.0001 \
        --max-tokens 40000 \
        --update-freq 2 \
        --max-epoch 100 \
        --save-dir checkpoints/mm_s2ut \
        --log-interval 100 \
        --fp16

Textless Version

    cd mm_s2ut/scripts/textless
    bash 1_train.sh

Inference and Evaluation

    cd mm_s2ut/scripts/enhanced
    
    # Single inference
    bash 2_inference.sh
    
    # Batch inference
    bash inference_all.sh
    
    # Generate waveform
    cd ..
    bash 3_generate_waveform.sh
    
    # ASR transcription
    python 4_transcript.py \
        --input_dir /path/to/generated/audio \
        --output_file transcripts.txt
    
    # Calculate BLEU
    python 5_bleu_asr.py \
        --reference ref.txt \
        --hypothesis hyp.txt

---

💡 Usage Examples

Command-Line Inference Example

    # Single sample inference
    python -m mm_s2ut.inference \
        --model_path checkpoints/mm_s2ut/checkpoint_best.pt \
        --audio_path examples/audio.wav \
        --image_path examples/image.jpg \
        --output_dir outputs/
    
    # Batch inference
    python -m mm_s2ut.inference \
        --model_path checkpoints/mm_s2ut/checkpoint_best.pt \
        --manifest_file data/test.tsv \
        --output_dir outputs/batch/

Python API Example

    import torch
    from fairseq import checkpoint_utils, tasks
    from mm_s2ut.models import MM_S2UTTransformerModel
    
    # Load model
    models, cfg, task = checkpoint_utils.load_model_ensemble_and_task(
        ['checkpoints/mm_s2ut/checkpoint_best.pt']
    )
    model = models[0].cuda()
    model.eval()
    
    # Prepare input
    audio_path = 'examples/audio.wav'
    image_path = 'examples/image.jpg'
    
    # Load data
    sample = task.load_sample(audio_path, image_path)
    
    # Inference
    with torch.no_grad():
        units = model.generate(sample)
    
    # Generate waveform
    waveform = vocoder.generate(units)
    
    # Save output
    import soundfile as sf
    sf.write('output.wav', waveform, 16000)

Evaluation Example

    # Calculate ASR-BLEU
    python mm_s2ut/scripts/bleu_asr.py \
        --reference_text data/test.en \
        --generated_audio outputs/batch/*.wav \
        --asr_model facebook/wav2vec2-large-960h
    
    # Example output:
    # BLEU Score: 28.4
    # Detailed Scores:
    #   BLEU-1: 52.3
    #   BLEU-2: 35.6
    #   BLEU-3: 25.8
    #   BLEU-4: 28.4
    
    # Calculate WER (Word Error Rate)
    python mm_s2ut/scripts/wer.py \
        --reference data/test.txt \
        --hypothesis outputs/transcripts.txt
    
    # Example output:
    # WER: 15.2%
    # Insertions: 23
    # Deletions: 45
    # Substitutions: 89

---

🙏 Acknowledgments

This project is built on the following excellent open-source projects:

- Fairseq - Sequence-to-sequence toolkit by Meta AI Research
- Hugging Face Transformers - Pre-trained model library
- timm - PyTorch image models library
- Wav2Vec 2.0 - Self-supervised speech representation learning
- HuBERT - Hidden-Unit BERT

Thanks to all contributors for their hard work!

---

<div align="center">

⬆ Back to Top

</div>
