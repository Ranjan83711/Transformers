# Transformers

# Detailed Note on Transformer Model Types

## Table of Contents
1. [Introduction to Transformers](#introduction)
2. [Core Architecture](#core-architecture)
3. [Encoder-Only Models](#encoder-only)
4. [Decoder-Only Models (Generative AI)](#decoder-only)
5. [Encoder-Decoder Models](#encoder-decoder)
6. [Variational Autoencoders (VAE) with Transformers](#vae)
7. [Vision Transformers (ViT)](#vision-transformers)
8. [Multimodal Transformers](#multimodal)
9. [Diffusion Transformers (DiT)](#diffusion-transformers)
10. [Graph Transformers](#graph-transformers)
11. [State Space Models & Hybrid Architectures](#ssm)
12. [Mixture of Experts (MoE) Transformers](#moe)
13. [Retrieval-Augmented Transformers](#rag)

---

## 1. Introduction to Transformers <a name="introduction"></a>

The Transformer architecture, introduced in the landmark paper **"Attention Is All You Need"** (Vaswani et al., 2017), revolutionized deep learning by replacing recurrent and convolutional mechanisms with **self-attention**. This allowed for:

- **Parallelized computation** (unlike sequential RNNs)
- **Long-range dependency modeling**
- **Scalability** to billions of parameters
- **Transfer learning** across tasks and domains

Since its introduction, transformers have branched into numerous architectural variants, each optimized for specific tasks and modalities.

---

## 2. Core Architecture <a name="core-architecture"></a>

The original transformer consists of:

### Self-Attention Mechanism
```
Attention(Q, K, V) = softmax(QK^T / √d_k) V
```

### Key Components:
| Component | Function |
|-----------|----------|
| Multi-Head Attention | Attends to different representation subspaces |
| Feed-Forward Network (FFN) | Non-linear transformation per position |
| Layer Normalization | Stabilizes training |
| Positional Encoding | Injects sequence order information |
| Residual Connections | Enables gradient flow in deep networks |

### Three Architectural Paradigms:
1. **Encoder-only** → Bidirectional understanding
2. **Decoder-only** → Autoregressive generation
3. **Encoder-Decoder** → Sequence-to-sequence mapping

---

## 3. Encoder-Only Models <a name="encoder-only"></a>

### Philosophy
These models process the **entire input simultaneously** with bidirectional attention, making them excellent for understanding and classification tasks.

### How They Work
- Input tokens attend to **all other tokens** in both directions
- Produce rich contextual embeddings
- Pre-trained using masked language modeling (MLM) or similar objectives

### Key Models

#### **BERT (Bidirectional Encoder Representations from Transformers)**
- **Developer:** Google (2018)
- **Pre-training objectives:** Masked Language Modeling (MLM) + Next Sentence Prediction (NSP)
- **Architecture:** 12/24 layers, 110M/340M parameters
- **Key innovation:** Bidirectional context for all tokens
- **Use cases:** Text classification, NER, question answering, semantic similarity

#### **RoBERTa (Robustly Optimized BERT)**
- **Developer:** Meta/Facebook (2019)
- Removed NSP, trained longer with more data, dynamic masking
- Consistently outperforms BERT

#### **ALBERT (A Lite BERT)**
- Parameter sharing across layers
- Factorized embedding parameterization
- Sentence Order Prediction instead of NSP

#### **ELECTRA**
- **Training:** Replaced Token Detection (more sample-efficient than MLM)
- Uses a generator-discriminator setup
- Discriminator learns to identify which tokens were replaced

#### **DeBERTa (Decoding-enhanced BERT with Disentangled Attention)**
- **Developer:** Microsoft
- Disentangled attention: separate content and position embeddings
- Enhanced mask decoder for pre-training

### Applications:
- Sentiment analysis
- Named Entity Recognition (NER)
- Semantic textual similarity
- Document classification
- Feature extraction for downstream tasks

---

## 4. Decoder-Only Models (Generative AI) <a name="decoder-only"></a>

### Philosophy
These models generate text **autoregressively** — predicting one token at a time, conditioned on all previous tokens. They use **causal (masked) self-attention** to prevent looking at future tokens.

### How They Work
- Unidirectional (left-to-right) attention
- Training objective: Next-token prediction (language modeling)
- Generation: Sample or greedily decode tokens sequentially
- **P(x₁, x₂, ..., xₙ) = ∏ P(xₜ | x₁, ..., xₜ₋₁)**

### Key Models

#### **GPT Series (Generative Pre-trained Transformer)**
| Model | Developer | Parameters | Key Features |
|-------|-----------|-----------|--------------|
| GPT-1 | OpenAI (2018) | 117M | Demonstrated generative pre-training + fine-tuning |
| GPT-2 | OpenAI (2019) | 1.5B | Zero-shot capabilities, "too dangerous to release" |
| GPT-3 | OpenAI (2020) | 175B | In-context learning, few-shot prompting |
| GPT-4 | OpenAI (2023) | ~1.7T (rumored MoE) | Multimodal, RLHF-aligned, reasoning |
| GPT-4o | OpenAI (2024) | Undisclosed | Omni-modal: text, vision, audio |

#### **LLaMA Series (Meta)**
- **LLaMA 1** (2023): 7B–65B, open-weights, efficient training
- **LLaMA 2** (2023): Improved training, RLHF chat variants, 4K context
- **LLaMA 3** (2024): 8B–405B, 128K context, multilingual

#### **Claude (Anthropic)**
- Constitutional AI (CAI) alignment approach
- Focus on safety, helpfulness, and harmlessness
- Long context windows (200K tokens)

#### **PaLM / Gemini (Google)**
- **PaLM** (2022): 540B params, Pathways system
- **Gemini** (2023): Natively multimodal, reasoning-focused

#### **Mistral / Mixtral**
- Efficient open models with sliding window attention
- Mixtral: Sparse MoE architecture

#### **Other Notable Models:**
- **Falcon** (Technology Innovation Institute)
- **Phi Series** (Microsoft) — small but capable
- **Qwen** (Alibaba)
- **Yi** (01.AI)
- **Cohere Command R**

### Decoding Strategies:
- **Greedy decoding:** Always pick highest probability token
- **Beam search:** Maintain top-k hypotheses
- **Top-k sampling:** Sample from top k tokens
- **Top-p (nucleus) sampling:** Sample from smallest set with cumulative probability ≥ p
- **Temperature scaling:** Control randomness
- **Repetition penalty:** Discourage repeated phrases

### Training Pipeline:
1. **Pre-training:** Massive unsupervised corpus, next-token prediction
2. **Supervised Fine-Tuning (SFT):** Instruction-following examples
3. **RLHF/RLAIF:** Reinforcement Learning from Human/AI Feedback
4. **DPO:** Direct Preference Optimization (simpler alternative to RLHF)

### Applications:
- Text generation and completion
- Code generation (Codex, GitHub Copilot)
- Chatbots and conversational AI
- Summarization, translation
- Reasoning and problem-solving
- Creative writing

---

## 5. Encoder-Decoder Models <a name="encoder-decoder"></a>

### Philosophy
These follow the original transformer design: an **encoder** processes the input, and a **decoder** generates the output while attending to the encoder's representations via **cross-attention**.

### How They Work
- Encoder: Bidirectional attention over input
- Decoder: Causal attention over generated tokens + cross-attention to encoder
- Naturally suited for tasks with distinct input-output pairs

### Key Models

#### **T5 (Text-to-Text Transfer Transformer)**
- **Developer:** Google (2019)
- **Key idea:** Frame ALL NLP tasks as text-to-text
  - Classification: "sentiment: I love this movie" → "positive"
  - Translation: "translate English to French: Hello" → "Bonjour"
- Variants: T5-Small (60M) to T5-11B, Flan-T5 (instruction-tuned)

#### **BART (Bidirectional and Auto-Regressive Transformers)**
- **Developer:** Meta (2019)
- Pre-training: Corrupt text (masking, deletion, permutation, rotation) → reconstruct
- Excellent for summarization, text generation

#### **mBART**
- Multilingual BART for machine translation
- Pre-trained on 25 languages simultaneously

#### **Pegasus**
- Specifically designed for abstractive summarization
- Gap Sentences Generation (GSG) pre-training objective

#### **MarianMT**
- Optimized for machine translation
- Thousands of language pair models available

### Applications:
- Machine translation
- Text summarization
- Question answering
- Data-to-text generation
- Grammar correction

---

## 6. Variational Autoencoders (VAE) with Transformers <a name="vae"></a>

### Background on VAEs

A **Variational Autoencoder** is a generative model that learns a **latent space** representation of data. Unlike standard autoencoders, VAEs impose a probabilistic structure on the latent space.

### Mathematical Foundation:
```
ELBO = E_q(z|x)[log p(x|z)] - KL(q(z|x) || p(z))
         ↑ Reconstruction       ↑ Regularization
```

Where:
- **q(z|x):** Encoder (approximate posterior)
- **p(x|z):** Decoder (likelihood)
- **p(z):** Prior (typically N(0, I))
- **KL divergence:** Forces latent space to be smooth and continuous

### Transformer-Based VAE Architectures

#### **Optimus (Organizing Sentences via Pre-trained Modeling of a Latent Space)**
- Combines BERT (encoder) + GPT-2 (decoder) with a VAE latent space
- Learns smooth sentence-level representations
- Enables interpolation between sentences in latent space
- Applications: Controlled text generation, style transfer

#### **DELLA (Decoding-Enhanced Latent Language Autoencoder)**
- Addresses the "posterior collapse" problem in text VAEs
- Uses decoder-side auxiliary connections

#### **VQ-VAE (Vector Quantized VAE)**
- Discrete latent spaces instead of continuous
- Combined with transformers for autoregressive prior modeling
- **VQ-VAE-2:** Hierarchical, high-quality image generation

#### **DALL·E 1 (Discrete VAE + Transformer)**
- Uses a **discrete VAE** to encode images into token sequences
- Autoregressive transformer generates image tokens conditioned on text
- Architecture: dVAE encoder → 256 image tokens → GPT-like transformer

#### **Latent Diffusion Models (Stable Diffusion)**
- VAE encodes images to compact latent space
- Diffusion process operates in latent space (not pixel space)
- Transformer-based U-Net (or DiT) for denoising

### The Posterior Collapse Problem
In Transformer VAEs for text, the powerful autoregressive decoder can ignore the latent variable:
- **Solutions:** KL annealing, free bits, aggressive training, δ-VAE, skip connections to latent

### Transformer-VAE Variants:

| Model | Latent Type | Domain | Key Feature |
|-------|------------|--------|-------------|
| Optimus | Continuous | Text | BERT+GPT2 |
| VQ-VAE-2 | Discrete | Images | Hierarchical codebook |
| DALL·E 1 | Discrete | Text→Image | dVAE + Autoregressive |
| Stable Diffusion | Continuous | Images | Latent diffusion |
| MusicVAE | Continuous | Music | Long-term structure |
| ProtTrans-VAE | Continuous | Proteins | Protein design |

### Applications:
- Controlled text generation (style, sentiment, topic)
- Image synthesis and manipulation
- Drug/molecule discovery
- Data augmentation
- Anomaly detection
- Disentangled representation learning
- Latent space arithmetic (e.g., "king" - "man" + "woman" = "queen")

---

## 7. Vision Transformers (ViT) <a name="vision-transformers"></a>

### Philosophy
Apply transformers directly to images by treating image patches as tokens.

### How They Work
1. Split image into fixed-size patches (e.g., 16×16)
2. Linearly embed each patch
3. Add positional embeddings
4. Process through standard transformer encoder
5. Use [CLS] token for classification (or average pooling)

### Key Models

#### **ViT (Vision Transformer)**
- **Developer:** Google (2020)
- Surprisingly effective with enough data (JFT-300M)
- Variants: ViT-B/16, ViT-L/16, ViT-H/14

#### **DeiT (Data-efficient Image Transformer)**
- **Developer:** Meta
- Knowledge distillation from CNN teachers
- Trains effectively on ImageNet alone (no JFT)

#### **Swin Transformer (Shifted Windows)**
- Hierarchical feature maps (like CNNs)
- Shifted window attention for efficiency: O(n) instead of O(n²)
- Excellent for detection and segmentation

#### **BEiT (BERT for Images)**
- Masked image modeling pre-training
- Predicts visual tokens of masked patches

#### **MAE (Masked Autoencoders)**
- Mask 75% of patches, reconstruct in pixel space
- Asymmetric encoder-decoder design
- Very efficient pre-training

#### **DINO / DINOv2**
- Self-supervised ViTs using self-distillation
- Emergent segmentation in attention maps
- Strong general visual features without labels

#### **EVA**
- Billion-scale ViT models
- EVA-02: State-of-the-art visual representations

### Applications:
- Image classification
- Object detection (DETR, DINO detector)
- Semantic/instance segmentation
- Medical imaging
- Satellite/aerial image analysis
- Video understanding (TimeSformer, ViViT)

---

## 8. Multimodal Transformers <a name="multimodal"></a>

### Philosophy
Process and align multiple modalities (text, image, audio, video) within unified transformer architectures.

### Contrastive Learning Approaches

#### **CLIP (Contrastive Language-Image Pre-training)**
- **Developer:** OpenAI (2021)
- Trains image encoder + text encoder jointly
- Contrastive loss: Match correct image-text pairs
- Zero-shot image classification via text prompts
- Foundation for many downstream models

#### **ALIGN (Google)**
- Similar to CLIP but with noisy web-scale data (1.8B pairs)
- Demonstrates that scale compensates for noise

#### **SigLIP**
- Sigmoid loss instead of softmax (more efficient for large batches)
- Used in many modern vision-language models

### Generative Multimodal Models

#### **DALL·E 2 / DALL·E 3**
- Text-to-image generation
- DALL·E 2: CLIP text → CLIP image embedding → Diffusion decoder
- DALL·E 3: Improved prompt following with better captioning

#### **Flamingo (DeepMind)**
- Few-shot visual learning
- Interleaves visual tokens with language model
- Perceiver Resampler for flexible visual token count

#### **GPT-4V / GPT-4o**
- Native multimodal understanding
- Processes interleaved text and images
- Omni-modal: Text, vision, audio in unified model

#### **LLaVA (Large Language and Vision Assistant)**
- Simple architecture: CLIP ViT + LLM with linear projection
- Surprisingly effective visual instruction following
- Open-source ecosystem: LLaVA-1.5, LLaVA-NeXT

#### **Gemini (Google)**
- Natively multimodal from pre-training
- Handles text, images, video, audio, code
- Interleaved multimodal generation

#### **CogVLM / Qwen-VL / InternVL**
- Open-source multimodal models
- Various architectural choices for vision-language fusion

### Audio/Speech Transformers

#### **Whisper (OpenAI)**
- Robust speech recognition via large-scale weak supervision
- Encoder-decoder: Audio spectrogram → Text

#### **AudioLM / MusicLM (Google)**
- Audio generation using discrete audio tokens
- Semantic + acoustic tokenization

#### **Bark / VALL-E**
- Neural codec language models for speech
- Zero-shot voice cloning from short audio prompts

### Fusion Strategies:
| Strategy | Description | Examples |
|----------|-------------|----------|
| Early Fusion | Concatenate modality tokens before transformer | Unified-IO |
| Late Fusion | Separate encoders, combine at final layers | CLIP |
| Cross-Attention | One modality attends to another | Flamingo |
| Interleaved | Mix modality tokens in sequence | Gemini, GPT-4o |

---

## 9. Diffusion Transformers (DiT) <a name="diffusion-transformers"></a>

### Philosophy
Replace the U-Net backbone in diffusion models with transformer architecture for better scalability and performance.

### How They Work
1. **Forward process:** Gradually add noise to data
2. **Reverse process:** Transformer predicts and removes noise
3. Input: Noisy latent + timestep + conditioning
4. Output: Predicted noise (ε-prediction) or clean data (x₀-prediction)

### Key Models

#### **DiT (Diffusion Transformer)**
- **Developer:** Meta/UC Berkeley (2022)
- Standard transformer blocks with adaptive layer norm (adaLN-Zero)
- Scaled effectively: DiT-XL/2 matches/beats U-Net diffusion

#### **U-ViT**
- Combines U-Net skip connections with ViT backbone
- Long skip connections between transformer blocks

#### **Sora (OpenAI)**
- Video generation using "spacetime patches"
- Operates in compressed latent video space
- Demonstrates temporal consistency and world modeling

#### **Stable Diffusion 3 / SD3.5**
- **MMDiT (Multi-Modal DiT):** Joint attention between text and image latents
- Two separate streams with cross-attention
- Rectified Flow matching

#### **FLUX (Black Forest Labs)**
- Next-generation architecture from Stable Diffusion creators
- Improved flow matching and attention mechanisms

#### **Imagen / Imagen Video (Google)**
- Text-to-image/video diffusion
- Cascaded diffusion models at multiple resolutions

#### **PixArt-α / PixArt-Σ**
- Efficient DiT training strategies
- High-quality generation with less compute

### Key Techniques:
- **Classifier-Free Guidance (CFG):** Trade diversity for quality
- **Rectified Flow:** Straight-line probability paths
- **Latent Consistency Models:** Few-step generation
- **IP-Adapter / ControlNet:** Additional conditioning

### Applications:
- Text-to-image generation
- Text-to-video generation
- Image editing and inpainting
- 3D asset generation
- Audio synthesis
- Scientific simulations

---

## 10. Graph Transformers <a name="graph-transformers"></a>

### Philosophy
Apply attention mechanisms to graph-structured data, handling nodes, edges, and global graph properties.

### Challenges:
- No natural sequence order → need graph-aware positional encodings
- Varying graph sizes and structures
- Computational cost on large graphs

### Key Approaches

#### **Graphormer (Microsoft)**
- Centrality encoding, spatial encoding, edge encoding
- Won OGB Large-Scale Challenge
- Degree-based node importance

#### **Graph Attention Networks (GAT)**
- Attention coefficients between connected nodes
- Message passing with learned importance weights

#### **GPS (General, Powerful, Scalable)**
- Combines local message passing + global attention
- Flexible positional/structural encodings (random walk, Laplacian eigenvectors)

#### **TokenGT**
- Treats nodes and edges as tokens
- Standard transformer with graph-specific tokenization

### Positional Encodings for Graphs:
- Laplacian eigenvectors
- Random walk structural encoding
- Shortest path distances
- Node degree encoding

### Applications:
- Molecular property prediction
- Drug discovery
- Social network analysis
- Knowledge graph reasoning
- Traffic prediction
- Chip design

---

## 11. State Space Models & Hybrid Architectures <a name="ssm"></a>

### Motivation
Address transformer's O(n²) attention complexity for very long sequences.

### Key Models

#### **Mamba (S6)**
- **Developer:** Albert Gu, Tri Dao (2023)
- Selective State Space Model
- Linear-time sequence modeling
- Input-dependent selection mechanism
- Competitive with transformers at similar scale

#### **Mamba-2**
- Structured State Space Duality (SSD)
- Connects SSMs to attention theoretically
- 2-8x faster than Mamba-1

#### **RWKV (Receptance Weighted Key Value)**
- Combines transformer-like training with RNN-like inference
- Linear attention approximation
- Constant memory during inference

#### **RetNet (Retentive Network)**
- **Developer:** Microsoft
- Three computation paradigms: parallel, recurrent, chunkwise
- Retention mechanism replaces attention

#### **Jamba (AI21)**
- Hybrid Mamba + Transformer + MoE
- Selective attention layers for critical positions
- Efficient long-context processing

#### **Zamba / Hybrid approaches**
- Strategic placement of attention layers within SSM backbone
- Best of both worlds: efficiency + capability

### Comparison:
| Feature | Transformer | Mamba/SSM | Hybrid |
|---------|------------|-----------|--------|
| Training | Parallel | Parallel | Parallel |
| Inference | O(n²) memory | O(1) memory | Mixed |
| Long sequences | Expensive | Efficient | Efficient |
| In-context learning | Excellent | Good | Excellent |

---

## 12. Mixture of Experts (MoE) Transformers <a name="moe"></a>

### Philosophy
Use **conditional computation** — only activate a subset of parameters for each input token, enabling larger model capacity without proportional compute increase.

### How They Work
1. Router/gating network assigns each token to top-k experts
2. Only selected experts process the token
3. Outputs are combined (usually weighted sum)
4. Typically replaces dense FFN layers with MoE FFN

### Key Mechanism:
```
y = Σ G(x)ᵢ · Expertᵢ(x)  for top-k experts
```

### Key Models

#### **Switch Transformer (Google)**
- Routes to single expert (top-1)
- Simplified training, reduced communication
- 1.6T parameters, up to 4x pre-training speedup

#### **Mixtral 8x7B (Mistral AI)**
- 8 experts, 2 active per token
- ~46.7B total parameters, ~12.9B active
- Outperforms LLaMA-2 70B with much less compute

#### **Mixtral 8x22B**
- Larger variant: 176B total, ~39B active
- Competitive with much larger dense models

#### **GPT-4 (Rumored)**
- Reportedly 8 experts, ~220B each
- Would explain capability-to-latency ratio

#### **DeepSeek-MoE / DeepSeek-V2**
- Fine-grained expert segmentation
- Shared experts + routed experts
- Efficient expert-parallel training

#### **Grok (xAI)**
- MoE architecture for large-scale models

### Challenges:
- **Load balancing:** Ensuring even expert utilization
- **Routing collapse:** All tokens going to same expert
- **Training instability:** Auxiliary losses needed
- **Communication overhead:** In distributed settings
- **Fine-tuning complexity:** Expert specialization

### Routing Strategies:
- Token Choice: Each token picks experts
- Expert Choice: Each expert picks tokens
- Hash routing: Deterministic assignment
- Soft MoE: Differentiable weighted combination

---

## 13. Retrieval-Augmented Transformers <a name="rag"></a>

### Philosophy
Augment transformer generation with external knowledge retrieval, reducing hallucination and enabling updatable knowledge.

### Architecture:
```
Query → Retriever → Relevant Documents → Generator (LLM) → Answer
```

### Key Approaches

#### **RAG (Retrieval-Augmented Generation)**
- **Developer:** Meta (2020)
- Dense Passage Retrieval (DPR) + BART/seq2seq generator
- RAG-Sequence vs. RAG-Token (marginalization over documents)

#### **RETRO (Retrieval-Enhanced Transformer)**
- **Developer:** DeepMind
- Retrieval integrated into transformer architecture
- Chunked cross-attention to retrieved neighbors
- 25x fewer parameters with comparable performance to GPT-3

#### **Atlas**
- Few-shot learner with retrieval augmentation
- Joint training of retriever and reader

#### **REALM (Retrieval-Augmented Language Model)**
- Pre-trains with retrieval as latent variable
- End-to-end differentiable retrieval

### Modern RAG Systems:
- **Embedding models:** E5, BGE, Cohere Embed, Voyage
- **Vector databases:** Pinecone, Weaviate, Qdrant, Chroma, FAISS
- **Advanced techniques:**
  - Hybrid search (dense + sparse/BM25)
  - Re-ranking (cross-encoders)
  - Query decomposition
  - HyDE (Hypothetical Document Embeddings)
  - Corrective RAG, Self-RAG
  - Graph RAG (Microsoft)

### Applications:
- Question answering over documents
- Enterprise knowledge bases
- Up-to-date information access
- Reducing hallucinations
- Domain-specific assistants

---

## Additional Specialized Architectures

### Sparse Attention Transformers
| Model | Attention Pattern |
|-------|-------------------|
| Longformer | Local + Global attention |
| BigBird | Random + Window + Global |
| Sparse Transformer | Fixed stride patterns |
| LongT5 | Transient Global attention |

### Efficient / Linear Transformers
- **Linformer:** Low-rank approximation of attention
- **Performer:** Random feature approximation (FAVOR+)
- **Flash Attention:** IO-aware exact attention (hardware optimization)
- **Ring Attention:** Distributed long-context via ring communication
- **Paged Attention (vLLM):** Memory-efficient inference

### Protein/Science Transformers
- **AlphaFold 2:** Evoformer for protein structure prediction
- **ESM (Evolutionary Scale Modeling):** Protein language models
- **RoseTTAFold:** Three-track architecture for structure prediction

### Reinforcement Learning Transformers
- **Decision Transformer:** RL as sequence modeling
- **Gato (DeepMind):** Generalist agent across diverse tasks
- **RT-2 (Google):** Robotic Transformer for action generation

### Code Transformers
- **Codex / GitHub Copilot**
- **StarCoder / StarCoder2**
- **CodeLlama**
- **DeepSeek-Coder**
- **AlphaCode / AlphaCode 2**

---

## Comparative Summary

| Architecture | Attention Type | Primary Use | Key Strength |
|-------------|---------------|-------------|--------------|
| Encoder-only | Bidirectional | Understanding | Rich representations |
| Decoder-only | Causal | Generation | Scalable, general-purpose |
| Encoder-Decoder | Both | Seq2Seq | Distinct I/O mapping |
| Transformer-VAE | Both + Latent | Controlled generation | Smooth latent space |
| ViT | Bidirectional | Vision | Patch-based attention |
| Multimodal | Cross-modal | Cross-domain | Unified representations |
| DiT | Bidirectional/Causal | Synthesis | High-quality generation |
| MoE | Conditional sparse | Efficient scaling | Capacity without compute |
| SSM/Hybrid | Linear recurrence | Long sequences | Sub-quadratic complexity |
| Graph Transformer | Node-to-node | Structured data | Relational reasoning |
| RAG | Augmented | Knowledge-grounded | Reduces hallucination |

---

## Evolution Timeline

```
2017: Original Transformer (Vaswani et al.)
2018: GPT-1, BERT
2019: GPT-2, T5, BART, RoBERTa, ALBERT, XLNet
2020: GPT-3, ViT, DALL·E, RAG, CLIP
2021: Codex, DALL·E 2 concept, Switch Transformer
2022: PaLM, Chinchilla, Stable Diffusion, ChatGPT, AlphaFold2
2023: GPT-4, LLaMA 1&2, Mistral, Mixtral, Mamba, Gemini
2024: GPT-4o, LLaMA 3, Claude 3.5, Sora, Flux, Mamba-2, Gemini 1.5
2025: Reasoning models (o1, o3), DeepSeek-V3/R1, Claude 4, Gemini 2
```

---

## Key Trends & Future Directions

1. **Scaling Laws:** Larger models continue to show emergent capabilities
2. **Efficiency:** MoE, SSMs, quantization, distillation for deployment
3. **Multimodality:** Converging toward unified omni-modal models
4. **Reasoning:** Chain-of-thought, tree-of-thought, search-augmented thinking
5. **Agents:** Tool use, planning, multi-step execution
6. **Long Context:** 1M+ token windows becoming standard
7. **Alignment:** RLHF → DPO → Constitutional AI → Process Reward Models
8. **Open Source:** Increasingly competitive open models
9. **Architecture Search:** Beyond pure attention (hybrid SSM-attention)
10. **World Models:** From language models to simulators of reality

---

## Conclusion

The transformer architecture has proven to be remarkably versatile, spawning an entire ecosystem of models across modalities, scales, and applications. From understanding (BERT) to generation (GPT), from language (LLMs) to vision (ViT) to multimodal (Gemini), and from deterministic (standard transformers) to probabilistic (VAEs, diffusion), the core attention mechanism has been adapted, optimized, and combined in countless ways. The field continues to evolve rapidly, with hybrid architectures, efficient attention mechanisms, and new training paradigms pushing the boundaries of what's possible.
