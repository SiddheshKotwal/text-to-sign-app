# --- AGGRESSIVE PosixPath FIX ---
# This MUST be the very first block of code
import pathlib
import os
import torch
if os.name == 'nt':  # Check if running on Windows
    # This monkey-patches the pathlib module
    pathlib.PosixPath = pathlib.WindowsPath
# This tells torch to trust this object type
torch.serialization.add_safe_globals([pathlib.PosixPath])
# --- END FIX ---

import torch.nn as nn
import lightning as L
import streamlit as st
from pathlib import Path
import torch.nn.functional as F
from torch import Tensor
import sys

# --- Imports from your 'src' folder ---
# We REMOVE find_best_model, it's not needed
from helpers import load_config
from model_vq import VQ_Transformer
from model_translation import Transformer

# ---
# MODEL DEFINITIONS (Copied from your notebooks)
# ---

# --- GAN Generator Definition ---
SEQUENCE_LENGTH_GAN = 64
POSE_FEATURES = 534 

class Generator(nn.Module):
    def __init__(self, in_features=POSE_FEATURES, seq_len=SEQUENCE_LENGTH_GAN):
        super(Generator, self).__init__()
        channels = 256
        def conv_block(in_channels, out_channels, kernel_size=3, padding=1):
            return nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size, padding=padding, bias=False),
                nn.BatchNorm1d(out_channels),
                nn.ReLU(inplace=True)
            )
        self.model = nn.Sequential(
            conv_block(in_features, channels),
            conv_block(channels, channels),
            conv_block(channels, channels),
            nn.Conv1d(channels, in_features, kernel_size=3, padding=1),
            nn.Tanh()
        )
    def forward(self, x):
        identity = x
        x_permuted = x.permute(0, 2, 1)
        refinement_permuted = self.model(x_permuted)
        refinement = refinement_permuted.permute(0, 2, 1)
        return identity + refinement

# --- DAE (Diffusion) Refiner Definition ---
class SimplePoseRefinerNet(nn.Module):
    def __init__(self, in_channels, out_channels, hidden_dim=512):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, hidden_dim, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(hidden_dim, hidden_dim*2, kernel_size=3, padding=1)
        self.conv3 = nn.Conv1d(hidden_dim*2, hidden_dim*2, kernel_size=3, padding=1)
        self.conv4 = nn.Conv1d(hidden_dim*2, hidden_dim, kernel_size=3, padding=1)
        self.conv5 = nn.Conv1d(hidden_dim, out_channels, kernel_size=3, padding=1)
        self.norm1 = nn.GroupNorm(8, hidden_dim)
        self.norm2 = nn.GroupNorm(8, hidden_dim*2)
        self.norm3 = nn.GroupNorm(8, hidden_dim*2)
        self.norm4 = nn.GroupNorm(8, hidden_dim)
        self.act = nn.SiLU()

    def forward(self, x):
        h1 = self.act(self.norm1(self.conv1(x)))
        h2 = self.act(self.norm2(self.conv2(h1)))
        h3 = self.act(self.norm3(self.conv3(h2)))
        h4 = self.act(self.norm4(self.conv4(h3 + h2)))
        out = self.conv5(h4 + h1) + x
        return out

class DiffusionRefiner(L.LightningModule):
    def __init__(self, pose_dim: int, window_size: int, learning_rate: float = 1e-4):
        super().__init__()
        self.save_hyperparameters()
        self.pose_dim = pose_dim
        self.window_size = window_size
        self.learning_rate = learning_rate
        self.model = SimplePoseRefinerNet(
            in_channels=pose_dim,
            out_channels=pose_dim,
            hidden_dim=512
        )
    
    def _transpose_data(self, x: Tensor) -> Tensor:
        return x.permute(0, 2, 1)
    
    def _transpose_output(self, x: Tensor) -> Tensor:
        return x.permute(0, 2, 1)

    @torch.no_grad()
    def run_inference(self, noisy_condition: Tensor) -> Tensor:
        noisy_input_t = self._transpose_data(noisy_condition)
        refined_pose_t = self.model(noisy_input_t)
        return self._transpose_output(refined_pose_t)
    
    # Add dummy methods needed by L.LightningModule
    def training_step(self, batch, batch_idx):
        pass
    def validation_step(self, batch, batch_idx):
        pass
    def configure_optimizers(self):
        pass

# ---
# MAIN ASSET LOADING FUNCTION
# ---

@st.cache_resource(show_spinner="Loading all AI models, please wait...")
def load_all_assets():
    """
    Loads all models, configs, and vocabs into memory on CPU
    and caches them for the Streamlit app.
    """
    
    # --- 1. Set Device ---
    device = torch.device("cpu")
    assets = {}

    # --- 2. Load VQ-VAE Model & Codebook ---
    try:
        print("Loading VQ-VAE model...")
        # --- FIX: Use your file structure ---
        vq_model_dir = Path("./models/vq_models/phix_codebook")
        vq_config = load_config(vq_model_dir / "config.yaml")
        # --- FIX: Use exact checkpoint name ---
        vq_checkpoint_path = vq_model_dir / "vq_vae_model.ckpt"
        print(f"  Found VQ checkpoint: {vq_checkpoint_path.name}")

        pose_input_size = POSE_FEATURES 
        pose_fps = 12
        
        assets["pose_dim"] = pose_input_size
        assets["fps"] = pose_fps

        vq_model = VQ_Transformer(
            config=vq_config,
            train_batch_size=1, dev_batch_size=1,
            dataset=None,
            input_size=pose_input_size,
            model_dir=str(vq_model_dir),
            fps=pose_fps,
            loggers={},
        )
        
        # Load with weights_only=False
        vq_checkpoint = torch.load(vq_checkpoint_path, map_location=device, weights_only=False)
        
        vq_model.load_state_dict(vq_checkpoint["state_dict"], strict=True)
        vq_model = vq_model.to(device).eval()
        
        assets["codebook_pose"] = vq_model.get_codebook_pose().to(device)
        assets["vq_model"] = vq_model
        assets["vq_config"] = vq_config
        print("✅ VQ-VAE Model loaded.")

    except Exception as e:
        print(f"🔥 ERROR loading VQ-VAE model: {e}")
        st.error(f"Error loading VQ-VAE model: {e}. Check console for details.")
        return None

    # --- 3. Load Translation Model ---
    try:
        print("Loading Translation model...")
        # --- FIX: Use your file structure ---
        trans_model_dir = Path("./models/translation_models/phix_translation")
        trans_config = load_config(trans_model_dir / "config.yaml")
        # --- FIX: Use exact checkpoint name ---
        trans_checkpoint_path = trans_model_dir / "model-epoch=282-val_loss=0.00-val_MSE=0.00.ckpt"
        print(f"  Found Translation checkpoint: {trans_checkpoint_path.name}")

        text_vocab_path = trans_model_dir / "text_vocab.txt"
        with open(text_vocab_path, 'r', encoding='utf-8') as f:
            text_vocab_list = [line.strip() for line in f.readlines()]
        text_vocab = {word: i for i, word in enumerate(text_vocab_list)}
        
        trans_model = Transformer(
            config=trans_config,
            save_path=trans_model_dir,
            train_batch_size=1, dev_batch_size=1,
            src_vocab=text_vocab,
            output_size=assets["codebook_pose"].shape[0] + 4, 
            fps=assets["fps"],
            ground_truth_text={},
            codebook_pose=assets["codebook_pose"],
        )
        
        # Load with weights_only=False
        trans_checkpoint = torch.load(trans_checkpoint_path, map_location=device, weights_only=False)
        
        trans_model.load_state_dict(trans_checkpoint["state_dict"], strict=True)
        trans_model = trans_model.to(device).eval()
        
        assets["trans_model"] = trans_model
        assets["trans_config"] = trans_config
        assets["text_vocab"] = text_vocab
        print("✅ Translation Model loaded.")

    except Exception as e:
        print(f"🔥 ERROR loading Translation model: {e}")
        st.error(f"Error loading Translation model: {e}. Check console for details.")
        return None

    # --- 4. Load GAN Generator ---
    try:
        print("Loading GAN model...")
        # --- FIX: Use your file structure ---
        gan_model_path = Path("./models/generator_model.pth")
        if not gan_model_path.exists():
             raise FileNotFoundError(f"GAN model not found at {gan_model_path}")

        gan_generator = Generator(
            in_features=assets["pose_dim"], 
            seq_len=SEQUENCE_LENGTH_GAN
        ).to(device)
        
        # Load with weights_only=False
        gan_state_dict = torch.load(gan_model_path, map_location=device, weights_only=False)
        gan_generator.load_state_dict(gan_state_dict)
        
        gan_generator.eval()
        
        assets["gan_model"] = gan_generator
        print("✅ GAN Generator loaded.")

    except Exception as e:
        print(f"🔥 ERROR loading GAN model: {e}")
        st.error(f"Error loading GAN model: {e}. Check console for details.")
        return None

    # --- 5. Load Diffusion (DAE) Refiner ---
    try:
        print("Loading DAE model...")
        # --- FIX: Use your file structure ---
        dae_checkpoint_path = Path("./models/best_refiner.ckpt")
        if not dae_checkpoint_path.exists():
            raise FileNotFoundError(f"DAE model not found at {dae_checkpoint_path}")
        print(f"  Found DAE checkpoint: {dae_checkpoint_path.name}")

        # Load with weights_only=False
        dae_refiner = DiffusionRefiner.load_from_checkpoint(
            dae_checkpoint_path,
            map_location=device,
            pose_dim=assets["pose_dim"],
            window_size=assets["vq_config"]["data"]["window_size"],
            weights_only=False 
        ).to(device)
        
        dae_refiner.eval()
        
        assets["dae_model"] = dae_refiner
        print("✅ Diffusion (DAE) Refiner loaded.")
        
    except Exception as e:
        print(f"🔥 ERROR loading DAE model: {e}")
        st.error(f"Error loading DAE model: {e}. Check console for details.")
        return None

    # --- 6. Load Normalization Stats (FOR GAN ONLY) ---
    norm_stats = {
        "global_min": -0.5088222026824951,
        "global_max": 0.7907042503356934
    }
    assets["norm_stats"] = norm_stats
    print("✅ GAN Normalization stats loaded.")

    print("\n--- All assets loaded successfully! ---")
    return assets