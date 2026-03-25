#!/bin/bash

OUTPUT_DIR="data/raw"
mkdir -p "$OUTPUT_DIR"

SUCCESS=0
SKIP=0
FAIL=0

download() {
  local ID="$1"
  local NAME="$2"
  local OUTFILE="$OUTPUT_DIR/${NAME}.pdf"

  if [ -f "$OUTFILE" ]; then
    echo "SKIP (already exists): $NAME"
    ((SKIP++))
    return
  fi

  echo "Downloading: $NAME..."
  HTTP_CODE=$(curl -L --fail --silent --show-error \
    -w "%{http_code}" \
    "https://arxiv.org/pdf/${ID}" \
    -o "$OUTFILE")

  if [ "$HTTP_CODE" -eq 200 ]; then
    echo "  OK: $NAME"
    ((SUCCESS++))
  else
    echo "  FAIL: $NAME (HTTP $HTTP_CODE)"
    rm -f "$OUTFILE"
    ((FAIL++))
  fi

  sleep 1
}

# Foundation Architecture
download 1706.03762 "Attention_Is_All_You_Need"
download 1810.04805 "BERT"
download 2005.14165 "GPT-3"
download 2307.09288 "LLaMA_2"
download 2310.06825 "Mistral_7B"
download 2404.14219 "Phi-3_Technical_Report"
download 2407.10671 "Qwen2_Technical_Report"
download 2401.02385 "TinyLlama"

# RAG & Retrieval
download 2005.11401 "RAG_Lewis_et_al"
download 2310.11511 "Self-RAG"
download 2401.18059 "RAPTOR"
download 2212.10496 "HyDE"
download 2004.12832 "ColBERT"
download 2307.03172 "Lost_in_the_Middle"
download 1908.10084 "Sentence-BERT"
download 2309.07597 "BGE_Embedding"
download 2401.15884 "Corrective_RAG"

# Efficient Inference & Quantization
download 2106.09685 "LoRA"
download 2305.14314 "QLoRA"
download 2210.17323 "GPTQ"
download 2306.00978 "AWQ"
download 2208.07339 "LLM_int8"
download 2205.14135 "FlashAttention"
download 2307.08691 "FlashAttention_2"
download 2211.17192 "Speculative_Decoding"

# Agents & Reasoning
download 2201.11903 "Chain-of-Thought_Prompting"
download 2210.03629 "ReAct"
download 2305.10601 "Tree_of_Thoughts"
download 2302.04761 "Toolformer"
download 2205.00445 "MRKL_LangChain_Agent"
download 2203.11171 "Self-Consistency"
download 2303.11366 "Reflexion"

# Alignment & Safety
download 2203.02155 "InstructGPT_RLHF"
download 2212.08073 "Constitutional_AI"
download 2305.18290 "DPO"
download 2202.03629 "Hallucination_Survey"
download 2109.07958 "TruthfulQA"

# Vision & Multimodal
download 2010.11929 "ViT_Vision_Transformer"
download 2103.00020 "CLIP"
download 2304.07193 "DINOv2"
download 2304.02643 "SAM_Segment_Anything"
download 2304.08485 "LLaVA"
download 2303.08774 "GPT-4V_Technical_Report"
download 2204.14198 "Flamingo"

# Architecture Details
download 1502.03167 "Batch_Normalization"
download 1607.06450 "Layer_Normalization"
download 2104.09864 "RoPE_Rotary_Position_Embedding"
download 2305.13245 "GQA_Grouped_Query_Attention"
download 2101.03961 "Mixture_of_Experts"
download 1512.03385 "ResNet"

# Foundation LLMs (additional)
download 2204.02311 "PaLM"
download 2305.10403 "PaLM_2"
download 2403.05530 "Gemini_1.5"
download 2407.21783 "LLaMA_3"
download 2405.04434 "DeepSeek_V2"
download 2403.08295 "Gemma"
download 2205.01068 "OPT"
download 2211.05100 "BLOOM"
download 1910.10683 "T5"
download 1906.08237 "XLNet"
download 1907.11692 "RoBERTa"
download 2109.01652 "FLAN"
download 2203.15556 "Chinchilla_Scaling"
download 2001.08361 "Scaling_Laws_for_LLMs"
download 2206.07682 "Emergent_Abilities_of_LLMs"
download 2305.13048 "RWKV"
download 2312.00752 "Mamba"
download 1909.11942 "ALBERT"
download 2003.10555 "ELECTRA"
download 2501.12948 "DeepSeek_R1"

# Efficient Inference & Training
download 2004.05150 "Longformer"
download 2007.14062 "BigBird"
download 2001.04451 "Reformer"
download 2307.08621 "RetNet"
download 2309.06180 "vLLM_PagedAttention"
download 2211.10438 "SmoothQuant"
download 1711.05101 "AdamW"
download 1710.03740 "Mixed_Precision_Training"
download 1910.02054 "ZeRO_DeepSpeed"
download 1503.02531 "Knowledge_Distillation"

# Diffusion & Generative Models
download 2006.11239 "DDPM"
download 2010.02502 "DDIM"
download 2112.10752 "Latent_Diffusion_Stable_Diffusion"
download 2102.12092 "DALL-E"
download 2205.11487 "Imagen"
download 2011.13456 "Score_Based_Generative_Models"
download 2210.02747 "Flow_Matching"
download 2212.09748 "DiT_Diffusion_Transformer"
download 2302.05543 "ControlNet"
download 1406.2661  "GAN"
download 1912.04958 "StyleGAN2"
download 1312.6114  "VAE"
download 1906.00446 "VQ-VAE-2"
download 2204.06125 "DALL-E_2"
download 2304.08818 "VideoLDM"

# Computer Vision
download 2201.03545 "ConvNeXt"
download 2103.14030 "Swin_Transformer"
download 2012.12877 "DeiT"
download 2111.06377 "MAE_Masked_Autoencoder"
download 2106.08254 "BEiT"
download 2104.14294 "DINO_Self_Supervised_Vision"
download 2002.05709 "SimCLR"
download 2006.07733 "BYOL"
download 1905.11946 "EfficientNet"
download 2005.12872 "DETR"
download 1703.06870 "Mask_R-CNN"
download 1506.01497 "Faster_R-CNN"
download 2004.10934 "YOLOv4"
download 2003.13678 "RegNet"
download 2201.09792 "ConvMixer"

# Multimodal
download 2201.12086 "BLIP"
download 2301.12597 "BLIP-2"
download 2305.06500 "InstructBLIP"
download 2304.10592 "MiniGPT-4"
download 2305.05665 "ImageBind"
download 2308.12966 "Qwen-VL"
download 2311.03079 "CogVLM"
download 2312.14238 "InternVL"
download 2306.02858 "Video-LLaMA"
download 2212.04356 "Whisper"

# RAG & Retrieval (additional)
download 2004.04906 "DPR_Dense_Passage_Retrieval"
download 2002.08909 "REALM"
download 2007.01282 "FiD_Fusion_in_Decoder"
download 2104.08663 "BEIR_Benchmark"
download 2107.05720 "SPLADE"
download 2112.09118 "Contriever"
download 2212.03533 "E5_Embedding"
download 2210.07316 "MTEB_Benchmark"
download 2112.04426 "RETRO"
download 2203.05115 "Internet_Augmented_LLM"

# Agents & Tools
download 2303.17580 "HuggingGPT"
download 2308.08155 "AutoGen"
download 2308.00352 "MetaGPT"
download 2307.16789 "ToolBench"
download 2305.15334 "Gorilla_API_Calling"
download 2308.03688 "AgentBench"
download 2402.01030 "CodeAct"
download 2310.10634 "OpenAgents"
download 2303.16434 "TaskMatrix"
download 2405.15793 "SWE-agent"

# Alignment & Safety (additional)
download 1707.06347 "PPO"
download 2402.03300 "GRPO"
download 2305.11206 "LIMA"
download 2212.10560 "Self-Instruct"
download 2304.07327 "OpenAssistant"
download 2202.03286 "Red_Teaming_LLMs"
download 2403.13787 "RewardBench"
download 2307.01232 "RLHF_Survey"
download 2402.01306 "KTO"
download 2310.12773 "Safe_RLHF"

# Benchmarks & Evaluation
download 2009.03300 "MMLU"
download 2107.03374 "HumanEval_Codex"
download 2110.14168 "GSM8K"
download 2206.04615 "BIG-Bench"
download 2211.09110 "HELM"
download 2306.05685 "MT-Bench"
download 2310.06770 "SWE-bench"
download 2103.03874 "MATH_Benchmark"
download 2304.06364 "AGIEval"
download 2403.04132 "Chatbot_Arena"

# Code & Programming
download 2305.06161 "StarCoder"
download 2308.12950 "CodeLlama"
download 2401.14196 "DeepSeek_Coder"
download 2203.07814 "AlphaCode"
download 2306.08568 "WizardCoder"

# Speech & Audio
download 2209.03143 "AudioLM"
download 2301.11325 "MusicLM"
download 2210.13438 "EnCodec"
download 2006.11477 "Wav2Vec_2"
download 2306.15687 "Voicebox"

# Graph Neural Networks
download 1609.02907 "GCN"
download 1710.10903 "GAT"
download 1706.02216 "GraphSAGE"
download 1902.10197 "RotatE"
download 2106.05234 "Graphormer"

# Domain Applications
download 2212.13138 "Med-PaLM"
download 2210.10341 "BioGPT"
download 1903.10676 "SciBERT"
download 2202.07622 "ESMFold"
download 2211.09085 "Galactica"
download 2303.14070 "ChatDoctor"
download 2204.11817 "MolT5"
download 2306.06031 "FinGPT"
download 2308.11462 "LegalBench"
download 2303.17564 "BloombergGPT"

# Reasoning & Math
download 2211.12588 "Program_of_Thoughts"
download 2211.10435 "PAL"
download 2205.10625 "Least_to_Most_Prompting"
download 2310.06117 "Step_Back_Prompting"
download 2112.00114 "Scratchpad_Show_Your_Work"
download 2310.03731 "MathCoder"
download 2308.09583 "WizardMath"
download 2305.20050 "Process_Reward_Model"
download 2403.09629 "Quiet-STaR"
download 2203.14465 "STaR_Self_Taught_Reasoner"

# Misc Foundational
download 1301.3666  "Word2Vec"
download 1607.04606 "FastText"
download 1808.05377 "NAS_Survey"
download 2110.04544 "CLIP_Adapter"
download 2104.08691 "Prompt_Tuning"

echo ""
echo "Done.  Success: $SUCCESS / Skip: $SKIP / Fail: $FAIL  (Total target: 200 papers)"
