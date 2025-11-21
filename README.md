# AI Text to Sign Language Generation

**Project Title:** AI Text to Sign Language

## 👥 Group Members

- **Ganesh Atre** (202201040068)
- **Sahil Chalotra** (202201040072)
- **Siddhesh Kotwal** (202201040095)
- **Dhiraj Mundane** (202201040211)

---

## 📖 Project Overview

This project implements a sophisticated deep learning pipeline to translate natural language text (English/German) into continuous, realistic sign language video sequences.

Unlike standard end-to-end approaches which often struggle with the high dimensionality of video data, this project utilizes a **two-stage generative approach** combined with advanced **post-processing refinement**. The system first learns a discrete "vocabulary" of sign language motions and then learns to translate spoken language into these motion tokens. Finally, a refinement stage (using GANs or Denoising Autoencoders) smooths the output to create more human-like signing.

## 🏗️ Methodology & Architecture

The project effectively integrates four distinct classes of generative models: **Variational Autoencoders (VAE)**, **Transformers**, **Generative Adversarial Networks (GAN)**, and **Diffusion/Denoising Models**.

### Phase 1: The Base Translation Pipeline (Text-to-Robotic-Pose)

This phase builds the core translation engine using a "Divide and Conquer" strategy.

#### 1. Motion Tokenization (VQ-VAE)

- **Model:** VQ-Transformer (Vector Quantized Variational Autoencoder)
- **Role:** Learns a discrete codebook (dictionary) of fundamental sign language motion segments (e.g., hand shapes, small movements). It compresses high-dimensional pose data into compact integer tokens.
- **Input:** Windowed segments of 3D skeletal pose data
- **Output:** Reconstructed pose segments

#### 2. Sequence-to-Sequence Translation (Transformer)

- **Model:** Autoregressive Transformer (Encoder-Decoder)
- **Role:** Acts as the "translator." It learns the mapping between written text and the sequence of motion tokens learned by the VQ-VAE.
- **Input:** Tokenized Text (German/English)
- **Output:** A sequence of VQ codebook indices

### Phase 2: The Refinement Pipeline (Robotic-to-Human-like)

The raw output from Phase 1 can appear "robotic" due to the concatenation of discrete segments. Phase 2 employs refinement models to smooth transitions and correct artifacts.

#### Option A: Adversarial Refinement (GAN)

- **Generator:** A 1D-Convolutional network that predicts a "residual" correction to apply to the robotic pose
- **Discriminator:** A PatchGAN discriminator that learns to distinguish between "real" human motion and "refined" model output

#### Option B: Denoising Refinement (DAE/Diffusion)

- **Model:** A Denoising Autoencoder (DAE)
- **Role:** Trained to map "noisy" (VQ-reconstructed) poses back to their "clean" (ground truth) counterparts, effectively removing quantization noise and smoothing the sequence

## 🛠️ Technical Stack

- **Frameworks:** PyTorch, PyTorch Lightning
- **Models:** Transformers, VQ-VAE, PatchGAN, Denoising Autoencoder (1D-ResNet)
- **Data Processing:** MediaPipe (for skeleton visualization), NumPy, SciPy (Signal processing)
- **Metrics:** BLEU-4, ROUGE, chrf, WER (Word Error Rate) via Back-Translation
- **Tracking:** Weights & Biases (W&B)

## 🚀 Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/your-username/your-repo-name.git
   cd your-repo-name
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   # Additional requirements for video processing and metrics
   pip install "numpy<2" diffusers accelerate deep-translator moviepy
   ```

## 💻 Usage

### 1. Training the VQ-VAE (Motion Codebook)

Train the model to learn the dictionary of sign motions.

```bash
python __main__.py train vq config/codebook/codebook_config.yaml
```

### 2. Training the Translation Model

Train the transformer to translate text to motion codes.

```bash
python __main__.py train translate config/translation/translation_config.yaml
```

### 3. Inference (Text-to-Video)

Use the provided notebooks or scripts to run the full pipeline:

1. Input English text
2. Translate to German (if using Phoenix-2014T dataset)
3. Generate VQ tokens via the **Translation Model**
4. Decode tokens to poses via the **VQ-VAE**
5. Refine poses via the **GAN** or **DAE**
6. Render final `.mp4` video

## ⚖️ Ethical Considerations

- **Representation:** This model is trained on specific datasets (e.g., Phoenix-2014T) and may not represent the full diversity of sign language dialects or individual signing styles.
- **Accuracy:** As a generative AI, the system may produce grammatically incorrect or nonsensical signs. It should be viewed as an experimental assistive tool, not a replacement for qualified human interpreters.
- **Cultural Nuance:** Sign language relies heavily on facial expressions and non-manual markers which are difficult to capture perfectly; users should be aware of these limitations regarding emotional and tonal accuracy.

## 👏 Acknowledgements

This project builds upon the research and code structures provided by:

- **Sign-VQ-Transformer:** Walsh, Harry, et al. "A data-driven representation for sign language production."
- **RWTH-PHOENIX-Weather 2014T Dataset**