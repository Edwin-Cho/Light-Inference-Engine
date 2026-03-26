# download_papers.ps1 — Windows PowerShell version of download_papers.sh
# Usage: powershell -ExecutionPolicy Bypass -File download_papers.ps1

$OutputDir = "data\raw"
New-Item -ItemType Directory -Force -Path $OutputDir | Out-Null

$Success = 0
$Skip = 0
$Fail = 0

function Download-Paper {
    param([string]$Id, [string]$Name)
    $OutFile = "$OutputDir\$Name.pdf"

    if (Test-Path $OutFile) {
        Write-Host "SKIP (already exists): $Name"
        $script:Skip++
        return
    }

    Write-Host "Downloading: $Name..."
    try {
        $url = "https://arxiv.org/pdf/$Id"
        $response = Invoke-WebRequest -Uri $url -OutFile $OutFile -PassThru -UseBasicParsing
        if ($response.StatusCode -eq 200) {
            Write-Host "  OK: $Name"
            $script:Success++
        } else {
            Write-Host "  FAIL: $Name (HTTP $($response.StatusCode))"
            Remove-Item -Force $OutFile -ErrorAction SilentlyContinue
            $script:Fail++
        }
    } catch {
        Write-Host "  FAIL: $Name ($_)"
        Remove-Item -Force $OutFile -ErrorAction SilentlyContinue
        $script:Fail++
    }
    Start-Sleep -Seconds 1
}

# Foundation Architecture
Download-Paper "1706.03762" "Attention_Is_All_You_Need"
Download-Paper "1810.04805" "BERT"
Download-Paper "2005.14165" "GPT-3"
Download-Paper "2307.09288" "LLaMA_2"
Download-Paper "2310.06825" "Mistral_7B"
Download-Paper "2404.14219" "Phi-3_Technical_Report"
Download-Paper "2407.10671" "Qwen2_Technical_Report"
Download-Paper "2401.02385" "TinyLlama"

# RAG & Retrieval
Download-Paper "2005.11401" "RAG_Lewis_et_al"
Download-Paper "2310.11511" "Self-RAG"
Download-Paper "2401.18059" "RAPTOR"
Download-Paper "2212.10496" "HyDE"
Download-Paper "2004.12832" "ColBERT"
Download-Paper "2307.03172" "Lost_in_the_Middle"
Download-Paper "1908.10084" "Sentence-BERT"
Download-Paper "2309.07597" "BGE_Embedding"
Download-Paper "2401.15884" "Corrective_RAG"

# Efficient Inference & Quantization
Download-Paper "2106.09685" "LoRA"
Download-Paper "2305.14314" "QLoRA"
Download-Paper "2210.17323" "GPTQ"
Download-Paper "2306.00978" "AWQ"
Download-Paper "2208.07339" "LLM_int8"
Download-Paper "2205.14135" "FlashAttention"
Download-Paper "2307.08691" "FlashAttention_2"
Download-Paper "2211.17192" "Speculative_Decoding"

# Agents & Reasoning
Download-Paper "2201.11903" "Chain-of-Thought_Prompting"
Download-Paper "2210.03629" "ReAct"
Download-Paper "2305.10601" "Tree_of_Thoughts"
Download-Paper "2302.04761" "Toolformer"
Download-Paper "2205.00445" "MRKL_LangChain_Agent"
Download-Paper "2203.11171" "Self-Consistency"
Download-Paper "2303.11366" "Reflexion"

# Alignment & Safety
Download-Paper "2203.02155" "InstructGPT_RLHF"
Download-Paper "2212.08073" "Constitutional_AI"
Download-Paper "2305.18290" "DPO"
Download-Paper "2202.03629" "Hallucination_Survey"
Download-Paper "2109.07958" "TruthfulQA"

# Vision & Multimodal
Download-Paper "2010.11929" "ViT_Vision_Transformer"
Download-Paper "2103.00020" "CLIP"
Download-Paper "2304.07193" "DINOv2"
Download-Paper "2304.02643" "SAM_Segment_Anything"
Download-Paper "2304.08485" "LLaVA"
Download-Paper "2303.08774" "GPT-4V_Technical_Report"
Download-Paper "2204.14198" "Flamingo"

# Architecture Details
Download-Paper "1502.03167" "Batch_Normalization"
Download-Paper "1607.06450" "Layer_Normalization"
Download-Paper "2104.09864" "RoPE_Rotary_Position_Embedding"
Download-Paper "2305.13245" "GQA_Grouped_Query_Attention"
Download-Paper "2101.03961" "Mixture_of_Experts"
Download-Paper "1512.03385" "ResNet"

# Foundation LLMs
Download-Paper "2204.02311" "PaLM"
Download-Paper "2305.10403" "PaLM_2"
Download-Paper "2403.05530" "Gemini_1.5"
Download-Paper "2407.21783" "LLaMA_3"
Download-Paper "2405.04434" "DeepSeek_V2"
Download-Paper "2403.08295" "Gemma"
Download-Paper "2205.01068" "OPT"
Download-Paper "2211.05100" "BLOOM"
Download-Paper "1910.10683" "T5"
Download-Paper "1906.08237" "XLNet"
Download-Paper "1907.11692" "RoBERTa"
Download-Paper "2109.01652" "FLAN"
Download-Paper "2203.15556" "Chinchilla_Scaling"
Download-Paper "2001.08361" "Scaling_Laws_for_LLMs"
Download-Paper "2206.07682" "Emergent_Abilities_of_LLMs"
Download-Paper "2305.13048" "RWKV"
Download-Paper "2312.00752" "Mamba"
Download-Paper "1909.11942" "ALBERT"
Download-Paper "2003.10555" "ELECTRA"
Download-Paper "2501.12948" "DeepSeek_R1"

# Efficient Inference & Training
Download-Paper "2004.05150" "Longformer"
Download-Paper "2007.14062" "BigBird"
Download-Paper "2001.04451" "Reformer"
Download-Paper "2307.08621" "RetNet"
Download-Paper "2309.06180" "vLLM_PagedAttention"
Download-Paper "2211.10438" "SmoothQuant"
Download-Paper "1711.05101" "AdamW"
Download-Paper "1710.03740" "Mixed_Precision_Training"
Download-Paper "1910.02054" "ZeRO_DeepSpeed"
Download-Paper "1503.02531" "Knowledge_Distillation"

# Diffusion & Generative Models
Download-Paper "2006.11239" "DDPM"
Download-Paper "2010.02502" "DDIM"
Download-Paper "2112.10752" "Latent_Diffusion_Stable_Diffusion"
Download-Paper "2102.12092" "DALL-E"
Download-Paper "2205.11487" "Imagen"
Download-Paper "2011.13456" "Score_Based_Generative_Models"
Download-Paper "2210.02747" "Flow_Matching"
Download-Paper "2212.09748" "DiT_Diffusion_Transformer"
Download-Paper "2302.05543" "ControlNet"
Download-Paper "1406.2661"  "GAN"
Download-Paper "1912.04958" "StyleGAN2"
Download-Paper "1312.6114"  "VAE"
Download-Paper "1906.00446" "VQ-VAE-2"
Download-Paper "2204.06125" "DALL-E_2"
Download-Paper "2304.08818" "VideoLDM"

# Computer Vision
Download-Paper "2201.03545" "ConvNeXt"
Download-Paper "2103.14030" "Swin_Transformer"
Download-Paper "2012.12877" "DeiT"
Download-Paper "2111.06377" "MAE_Masked_Autoencoder"
Download-Paper "2106.08254" "BEiT"
Download-Paper "2104.14294" "DINO_Self_Supervised_Vision"
Download-Paper "2002.05709" "SimCLR"
Download-Paper "2006.07733" "BYOL"
Download-Paper "1905.11946" "EfficientNet"
Download-Paper "2005.12872" "DETR"
Download-Paper "1703.06870" "Mask_R-CNN"
Download-Paper "1506.01497" "Faster_R-CNN"
Download-Paper "2004.10934" "YOLOv4"
Download-Paper "2003.13678" "RegNet"
Download-Paper "2201.09792" "ConvMixer"

# Multimodal
Download-Paper "2201.12086" "BLIP"
Download-Paper "2301.12597" "BLIP-2"
Download-Paper "2305.06500" "InstructBLIP"
Download-Paper "2304.10592" "MiniGPT-4"
Download-Paper "2305.05665" "ImageBind"
Download-Paper "2308.12966" "Qwen-VL"
Download-Paper "2311.03079" "CogVLM"
Download-Paper "2312.14238" "InternVL"
Download-Paper "2306.02858" "Video-LLaMA"
Download-Paper "2212.04356" "Whisper"

# RAG & Retrieval (additional)
Download-Paper "2004.04906" "DPR_Dense_Passage_Retrieval"
Download-Paper "2002.08909" "REALM"
Download-Paper "2007.01282" "FiD_Fusion_in_Decoder"
Download-Paper "2104.08663" "BEIR_Benchmark"
Download-Paper "2107.05720" "SPLADE"
Download-Paper "2112.09118" "Contriever"
Download-Paper "2212.03533" "E5_Embedding"
Download-Paper "2210.07316" "MTEB_Benchmark"
Download-Paper "2112.04426" "RETRO"
Download-Paper "2203.05115" "Internet_Augmented_LLM"

# Agents & Tools
Download-Paper "2303.17580" "HuggingGPT"
Download-Paper "2308.08155" "AutoGen"
Download-Paper "2308.00352" "MetaGPT"
Download-Paper "2307.16789" "ToolBench"
Download-Paper "2305.15334" "Gorilla_API_Calling"
Download-Paper "2308.03688" "AgentBench"
Download-Paper "2402.01030" "CodeAct"
Download-Paper "2310.10634" "OpenAgents"
Download-Paper "2303.16434" "TaskMatrix"
Download-Paper "2405.15793" "SWE-agent"

# Alignment & Safety (additional)
Download-Paper "1707.06347" "PPO"
Download-Paper "2402.03300" "GRPO"
Download-Paper "2305.11206" "LIMA"
Download-Paper "2212.10560" "Self-Instruct"
Download-Paper "2304.07327" "OpenAssistant"
Download-Paper "2202.03286" "Red_Teaming_LLMs"
Download-Paper "2403.13787" "RewardBench"
Download-Paper "2307.01232" "RLHF_Survey"
Download-Paper "2402.01306" "KTO"
Download-Paper "2310.12773" "Safe_RLHF"

# Benchmarks & Evaluation
Download-Paper "2009.03300" "MMLU"
Download-Paper "2107.03374" "HumanEval_Codex"
Download-Paper "2110.14168" "GSM8K"
Download-Paper "2206.04615" "BIG-Bench"
Download-Paper "2211.09110" "HELM"
Download-Paper "2306.05685" "MT-Bench"
Download-Paper "2310.06770" "SWE-bench"
Download-Paper "2103.03874" "MATH_Benchmark"
Download-Paper "2304.06364" "AGIEval"
Download-Paper "2403.04132" "Chatbot_Arena"

# Code & Programming
Download-Paper "2305.06161" "StarCoder"
Download-Paper "2308.12950" "CodeLlama"
Download-Paper "2401.14196" "DeepSeek_Coder"
Download-Paper "2203.07814" "AlphaCode"
Download-Paper "2306.08568" "WizardCoder"

# Speech & Audio
Download-Paper "2209.03143" "AudioLM"
Download-Paper "2301.11325" "MusicLM"
Download-Paper "2210.13438" "EnCodec"
Download-Paper "2006.11477" "Wav2Vec_2"
Download-Paper "2306.15687" "Voicebox"

# Graph Neural Networks
Download-Paper "1609.02907" "GCN"
Download-Paper "1710.10903" "GAT"
Download-Paper "1706.02216" "GraphSAGE"
Download-Paper "1902.10197" "RotatE"
Download-Paper "2106.05234" "Graphormer"

# Domain Applications
Download-Paper "2212.13138" "Med-PaLM"
Download-Paper "2210.10341" "BioGPT"
Download-Paper "1903.10676" "SciBERT"
Download-Paper "2202.07622" "ESMFold"
Download-Paper "2211.09085" "Galactica"
Download-Paper "2303.14070" "ChatDoctor"
Download-Paper "2204.11817" "MolT5"
Download-Paper "2306.06031" "FinGPT"
Download-Paper "2308.11462" "LegalBench"
Download-Paper "2303.17564" "BloombergGPT"

# Reasoning & Math
Download-Paper "2211.12588" "Program_of_Thoughts"
Download-Paper "2211.10435" "PAL"
Download-Paper "2205.10625" "Least_to_Most_Prompting"
Download-Paper "2310.06117" "Step_Back_Prompting"
Download-Paper "2112.00114" "Scratchpad_Show_Your_Work"
Download-Paper "2310.03731" "MathCoder"
Download-Paper "2308.09583" "WizardMath"
Download-Paper "2305.20050" "Process_Reward_Model"
Download-Paper "2403.09629" "Quiet-STaR"
Download-Paper "2203.14465" "STaR_Self_Taught_Reasoner"

# Misc Foundational
Download-Paper "1301.3666"  "Word2Vec"
Download-Paper "1607.04606" "FastText"
Download-Paper "1808.05377" "NAS_Survey"
Download-Paper "2110.04544" "CLIP_Adapter"
Download-Paper "2104.08691" "Prompt_Tuning"

Write-Host ""
Write-Host "Done.  Success: $Success / Skip: $Skip / Fail: $Fail  (Total target: 200 papers)"
