import torch
import string
import pathlib
from pathlib import Path
from deep_translator import GoogleTranslator

# --- Imports from your copied repo files ---
from constants import PAD_ID, BOS_ID, EOS_ID
from stitch import stitch_poses, apply_low_pass_filter, interpolate_pose
from plot import make_pose_video

# --- Helper functions for GAN Normalization ---
# We define these here so pipeline.py is self-contained.
# These are from your GAN notebook.

def normalize_pose_tensor(tensor, g_min, g_max):
    """Normalizes a pose tensor from [min, max] to [-1, 1]"""
    normalized = (tensor - g_min) / (g_max - g_min)
    return (normalized * 2) - 1

def denormalize_pose_tensor(tensor, g_min, g_max):
    """De-normalizes a pose tensor from [-1, 1] to [min, max]"""
    unscaled = (tensor + 1) / 2
    return (unscaled * (g_max - g_min)) + g_min


# --- Pipeline Function 1: Text to Tokens ---

def translate_and_tokenize(text_input: str, text_vocab: dict):
    """
    1. Translates English text to German.
    2. Pre-processes and tokenizes the German text.
    3. Returns tensors ready for the Transformer model.
    """
    try:
        translator = GoogleTranslator(source='en', target='de')
        german_sentence = translator.translate(text_input)
    except Exception as e:
        print(f"Translation failed: {e}")
        german_sentence = text_input # Fallback

    # Pre-processing from your notebook
    text = german_sentence.lower()
    text = text.replace("-", " ")
    remove_chars = (
        string.punctuation.replace(".", "") + "„“…–’‘”‘‚´" + "0123456789€"
    )
    text = "".join(ch for ch in text if ch not in remove_chars).split()

    # Tokenize
    token_indices = [text_vocab.get(w, text_vocab["<unk>"]) for w in text]
    token_indices.append(EOS_ID)  # Add End-of-Sequence token

    # Create batch tensors (batch size 1, on CPU)
    device = torch.device("cpu")
    src = torch.tensor(token_indices, dtype=torch.long, device=device).unsqueeze(0)
    src_length = torch.tensor([len(token_indices)], dtype=torch.long, device=device)
    src_mask = (src != PAD_ID).unsqueeze(-2).to(device)

    return src, src_length, src_mask


# --- Pipeline Function 2: Tokens to Baseline Pose ---

@torch.no_grad()
def run_text_to_baseline(src, src_length, src_mask, trans_model, vq_config):
    """
    1. Runs the Transformer to get VQ token indices.
    2. Looks up indices in the codebook to get pose segments.
    3. Stitches segments into a full "robotic" pose.
    
    Returns:
        - pose_segments (torch.Tensor): The *un-stitched* segments (for DAE).
        - stitched_baseline_pose (torch.Tensor): The *stitched* pose (for GAN/video).
        - confidence (float): The average probability score of the prediction.
    """
    device = torch.device("cpu")
    model_settings = trans_model.hparams.config["model"]["beam_setting"]

    # 1. Run Transformer (Text -> VQ Tokens)
    # --- ETHICAL UPDATE: Unpack confidence score ---
    vq_tokens, confidence = trans_model.greedy_decode(
        src=src,
        src_length=src_length,
        src_mask=src_mask,
        max_output_length=model_settings["max_output_length"],
    )

    # 2. Post-process VQ Tokens
    vq_tokens = vq_tokens.squeeze(0).cpu().numpy()
    eos_index = (vq_tokens == EOS_ID).nonzero()
    if eos_index[0].size > 0:
        eos_index = eos_index[0][0]
        vq_tokens = vq_tokens[:eos_index]
    
    # Shift tokens (remove 4 special tokens)
    vq_tokens = vq_tokens[vq_tokens >= 4] - 4

    if len(vq_tokens) == 0:
        return None, None, 0.0  # Failed to generate

    # 3. Convert VQ Tokens to Poses (Baseline)
    # pose_segments shape: (N_tokens, window_size, pose_dim)
    pose_segments = trans_model.codebook_pose[vq_tokens].to(device)

    # 4. Stitch Poses
    window_size = vq_config["data"]["window_size"]
    stitch_config = trans_model.hparams.config["stitch"]
    
    # Reshape for stitching: (N_tokens, window_size, num_joints, 3)
    pose_segments_reshaped = pose_segments.reshape(
        pose_segments.shape[0], window_size, -1, 3
    )
    
    # Stitch (this function uses numpy, so we send to CPU)
    stitched_baseline_pose = stitch_poses(
        poses=pose_segments_reshaped.cpu(),
        stitch_config=stitch_config
    )
    
    # Ensure output is a tensor
    if not isinstance(stitched_baseline_pose, torch.Tensor):
        stitched_baseline_pose = torch.from_numpy(stitched_baseline_pose)

    return pose_segments, stitched_baseline_pose, confidence


# --- Pipeline Function 3: GAN Refinement ---

@torch.no_grad()
def run_gan_refiner(stitched_pose, gan_model, norm_stats, pose_dim, seq_len=64):
    """
    1. Normalizes and chunks the full baseline pose.
    2. Runs the GAN Generator on each chunk.
    3. De-normalizes and stitches the refined chunks back together.
    """
    device = torch.device("cpu")
    
    # 1. Flatten and Normalize
    # (T, J, 3) -> (T, F)
    pose_flat = stitched_pose.reshape(stitched_pose.shape[0], -1) 
    pose_normalized = normalize_pose_tensor(
        pose_flat, norm_stats["global_min"], norm_stats["global_max"]
    ).to(device)

    # 2. Pad and Chunk
    original_length = pose_normalized.shape[0]
    pad_len = (seq_len - (original_length % seq_len)) % seq_len
    if pad_len > 0:
        padding = torch.zeros(pad_len, pose_dim, device=device)
        pose_normalized = torch.cat([pose_normalized, padding], dim=0)
        
    pose_chunks = pose_normalized.view(-1, seq_len, pose_dim)

    # 3. Run Generator
    refined_chunks = gan_model(pose_chunks)

    # 4. Re-assemble and De-normalize
    refined_pose_flat = refined_chunks.view(-1, pose_dim)
    refined_pose_flat = refined_pose_flat[:original_length] # Trim padding
    
    refined_pose_denorm = denormalize_pose_tensor(
        refined_pose_flat, norm_stats["global_min"], norm_stats["global_max"]
    )
    
    # 5. Reshape back to (T, J, 3)
    stitched_gan_pose = refined_pose_denorm.reshape(original_length, -1, 3)
    
    return stitched_gan_pose


# --- Pipeline Function 4: DAE (Diffusion) Refinement ---

@torch.no_grad()
def run_dae_refiner(pose_segments, dae_model, vq_config, trans_config):
    """
    1. Runs the DAE (Refiner) model on the *un-stitched* segments.
    2. Stitches the *refined* segments together.
    
    Note: This model operates on un-normalized data.
    """
    device = torch.device("cpu")
    
    # 1. Run Refiner (DAE)
    # pose_segments shape is (N_tokens, T, C), which is (B, T, C) for the model
    refined_segments = dae_model.run_inference(pose_segments.to(device))
    
    # 2. Stitch Refined Segments
    window_size = vq_config["data"]["window_size"]
    stitch_config = trans_config["stitch"]

    refined_segments_reshaped = refined_segments.reshape(
        refined_segments.shape[0], window_size, -1, 3
    )
    
    stitched_dae_pose = stitch_poses(
        poses=refined_segments_reshaped.cpu(),
        stitch_config=stitch_config
    )

    if not isinstance(stitched_dae_pose, torch.Tensor):
        stitched_dae_pose = torch.from_numpy(stitched_dae_pose)

    return stitched_dae_pose


# --- Pipeline Function 5: Video Saving ---

def save_video(pose_tensor: torch.Tensor, 
               file_path: str, 
               fps: int, 
               smoothed: bool = False):
    """
    Saves a pose tensor to an .mp4 video file.
    If 'smoothed' is True, applies filtering and interpolation.
    """
    
    output_pose = pose_tensor.cpu()
    output_fps = fps
    
    # This is the "Refinement" post-processing
    if smoothed:
        # 1. Apply smoothing filter
        output_pose = apply_low_pass_filter(
            output_pose, cutoff_freq=2.0, fs=fps
        )
        # 2. Interpolate to double the frames
        new_length = output_pose.shape[0] * 2
        output_pose = interpolate_pose(
            output_pose, num_sample_pts=new_length
        )
        # 3. Double the FPS to maintain original speed
        output_fps = fps * 2
        
    # Generate video
    video_path = Path(file_path).parent
    video_name = Path(file_path).name
    
    make_pose_video(
        poses=[output_pose],
        names=["Prediction"],
        video_name=video_name,
        save_dir=video_path,
        fps=output_fps,
        overwrite=True
    )
    print(f"Video saved to {file_path}")