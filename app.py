# --- AGGRESSIVE PosixPath FIX ---
# This MUST be the very first block of code in app.py
import sys
import pathlib
import os

if os.name == 'nt':  # Check if running on Windows
    # This is a monkey-patch.
    # It tells Python's unpickler that whenever it sees a
    # 'pathlib.PosixPath' object in the file, it should
    # actually create a 'pathlib.WindowsPath' object instead.
    pathlib.PosixPath = pathlib.WindowsPath
# --- END FIX ---

import torch  # <-- This torch import is now safe
import streamlit as st
import pandas as pd
import streamlit.components.v1 as components
from pathlib import Path
import numpy as np
import base64  # <-- 1. IMPORTED BASE64

# --- Core App Imports ---
import load_models
import pipeline

# ---
# 1. APPLICATION SETUP & MODEL LOADING
# ---
st.set_page_config(
    layout="wide",
    page_title="Text-to-Sign Generation",
    page_icon="🤖"
)

# Create a directory for temporary videos
TEMP_VIDEO_DIR = Path("./temp_videos")
TEMP_VIDEO_DIR.mkdir(exist_ok=True)

# Define file paths for the live demo videos
LIVE_BASELINE_VIDEO = TEMP_VIDEO_DIR / "live_baseline.mp4"
LIVE_GAN_VIDEO = TEMP_VIDEO_DIR / "live_gan.mp4"
LIVE_DAE_VIDEO = TEMP_VIDEO_DIR / "live_dae.mp4"

# Load all models, configs, vocabs, and stats.
assets = load_models.load_all_assets()

# Store W&B plot links
wandb_plots = {
    "vae": '<iframe src="https://wandb.ai/siddheshkotwal379-na/VQ%20SignVqTransformer/reports/VQ-VAE--VmlldzoxNDk0ODQ2Nw" style="border:none;height:1024px;width:100%"></iframe>',
    "transformer": '<iframe src="https://wandb.ai/siddheshkotwal379-na/Translation%20SignVqTransformer/reports/Transformer--VmlldzoxNDk0ODU4Ng" style="border:none;height:1024px;width:100%"></iframe>',
    "gan": '<iframe src="https://wandb.ai/siddheshkotwal379-na/sign-vq-transformer/reports/GAN-Pose-Refinement--VmlldzoxNDk0ODYxMw" style="border:none;height:1024px;width:100%"></iframe>',
    "diffusion": '<iframe src="https://wandb.ai/siddheshkotwal379-na/sign-pose-refinement/reports/Diffusion-Pose-Refinement--VmlldzoxNDk0ODY3NQ" style="border:none;height:1024px;width:100%"></iframe>',
}

# ---
# 2. SIDEBAR NAVIGATION
# ---
st.sidebar.title("Text to Sign Language 🤖")
st.sidebar.markdown("A multi-stage pipeline for sign language generation using VAEs, Transformers, GANs, and Diffusion.")

page = st.sidebar.radio(
    "Navigation",
    [
        "Live Demo",
        "Model Showcase",
        "Quantitative Metrics",
        "Model Comparison",
        "Training Plots",
        "Pipeline Explained"
    ],
    index=0
)

# ---
# 3. UI PAGES
# ---

# -----------------------------------------------------------------
# PAGE 1: LIVE DEMO
# -----------------------------------------------------------------
if page == "Live Demo":
    st.title("Live Text-to-Sign Generation")
    st.markdown("Enter English text to generate a sign language video. This runs on CPU and may take 20-30 seconds.")

    if assets:
        text_input = st.text_area(
            "Enter English text:",
            "Temperatures tonight will be between four and nine degrees."
        )

        if st.button("Generate Baseline Sign", type="primary"):
            if 'refined_video_bytes' in st.session_state:  # --- FIX 1 ---
                del st.session_state['refined_video_bytes']
            
            # Clear the "all" panel state if we generate a single baseline
            if 'all_baseline_b64_str' in st.session_state:
                del st.session_state['all_baseline_b64_str']
            if 'all_gan_b64_str' in st.session_state:
                del st.session_state['all_gan_b64_str']
            if 'all_dae_b64_str' in st.session_state:
                del st.session_state['all_dae_b64_str']

            with st.spinner("Generating... (This runs the Transformer & VQ-VAE lookup)"):
                try:
                    # 1. Text -> Tokens
                    src, src_length, src_mask = pipeline.translate_and_tokenize(
                        text_input, text_vocab=assets["text_vocab"]
                    )
                    
                    # 2. Tokens -> Poses (Baseline)
                    segments, baseline_pose = pipeline.run_text_to_baseline(
                        src, src_length, src_mask,
                        trans_model=assets["trans_model"],
                        vq_config=assets["vq_config"]
                    )
                    
                    if baseline_pose is None:
                        st.error("The model predicted an empty sequence. Please try a different sentence.")
                    else:
                        # 3. Save Baseline Video (UNSMOOTHED)
                        pipeline.save_video(
                            baseline_pose,
                            file_path=LIVE_BASELINE_VIDEO,
                            fps=assets["fps"],
                            smoothed=False
                        )
                        
                        # --- FIX 2: Read bytes and store them ---
                        with open(LIVE_BASELINE_VIDEO, "rb") as f:
                            video_bytes = f.read()
                        st.session_state.baseline_video_bytes = video_bytes
                        # --- END FIX ---
                        
                        # Save poses for refiners to use
                        st.session_state.pose_segments = segments
                        st.session_state.stitched_baseline_pose = baseline_pose

                except Exception as e:
                    st.error(f"An error occurred during baseline generation: {e}")
                    if 'st.session_state' in locals():
                        if 'baseline_video_bytes' in st.session_state:
                            del st.session_state['baseline_video_bytes']

        # --- Display Baseline Video ---
        if 'baseline_video_bytes' in st.session_state:  # --- FIX 3 ---
            st.subheader("1. Baseline Generation (VQ-VAE + Transformer)")
            st.markdown("This is the raw, 'robotic' output from the main Transformer model.")
            vid_cols = st.columns([0.2, 0.6, 0.2])
            with vid_cols[1]:
                st.video(st.session_state.baseline_video_bytes)
            
            st.divider()
            st.subheader("2. Refine Pose")
            st.markdown("Apply a refiner model to make the movement smoother and more realistic.")
            
            col1, col2 = st.columns(2)
            
            with col1:
                if st.button("Refine with GAN"):
                    # Clear the "all" panel state
                    if 'all_baseline_b64_str' in st.session_state:
                        del st.session_state['all_baseline_b64_str']

                    with st.spinner("Refining with GAN..."):
                        try:
                            gan_pose = pipeline.run_gan_refiner(
                                st.session_state.stitched_baseline_pose,
                                assets["gan_model"],
                                assets["norm_stats"],
                                assets["pose_dim"]
                            )
                            pipeline.save_video(
                                gan_pose,
                                file_path=LIVE_GAN_VIDEO,
                                fps=assets["fps"],
                                smoothed=True
                            )
                            # --- Read bytes for GAN video ---
                            with open(LIVE_GAN_VIDEO, "rb") as f:
                                video_bytes = f.read()
                            st.session_state.refined_video_bytes = video_bytes
                            st.session_state.refined_type = "GAN"
                        except Exception as e:
                            st.error(f"An error occurred during GAN refinement: {e}")

            with col2:
                if st.button("Refine with Diffusion"):
                    # Clear the "all" panel state
                    if 'all_baseline_b64_str' in st.session_state:
                        del st.session_state['all_baseline_b64_str']
                        
                    with st.spinner("Refining with Diffusion..."):
                        try:
                            dae_pose = pipeline.run_dae_refiner(
                                st.session_state.pose_segments,
                                assets["dae_model"],
                                assets["vq_config"],
                                assets["trans_config"]
                            )
                            pipeline.save_video(
                                dae_pose,
                                file_path=LIVE_DAE_VIDEO,
                                fps=assets["fps"],
                                smoothed=True
                            )
                            # --- Read bytes for DAE video ---
                            with open(LIVE_DAE_VIDEO, "rb") as f:
                                video_bytes = f.read()
                            st.session_state.refined_video_bytes = video_bytes
                            st.session_state.refined_type = "Diffusion"
                        except Exception as e:
                            st.error(f"An error occurred during Diffusion refinement: {e}")

        # --- Display Refined Video ---
        if 'refined_video_bytes' in st.session_state:  # --- FIX 5 ---
            st.subheader(f"3. Refined Output ({st.session_state.refined_type})")
            st.markdown("This video has been post-processed with the selected refiner and includes frame interpolation for a more natural look.")
            vid_cols = st.columns([0.2, 0.6, 0.2])
            with vid_cols[1]:
                st.video(st.session_state.refined_video_bytes)
        
        # -----------------------------------------------------------------
        # --- NEW SECTION: ALL-IN-ONE GENERATION ---
        # -----------------------------------------------------------------
        st.divider()
        st.subheader("3. All-in-One Generation & Comparison")
        st.markdown("Generate all three video outputs (Baseline, GAN-refined, and Diffusion-refined) at once for a direct comparison.")
        
        if st.button("Generate All Models"):
            # Clear single-view state
            if 'baseline_video_bytes' in st.session_state:
                del st.session_state['baseline_video_bytes']
            if 'refined_video_bytes' in st.session_state:
                del st.session_state['refined_video_bytes']
                
            # Clear all three video byte keys for this specific panel
            if 'all_baseline_b64_str' in st.session_state:
                del st.session_state['all_baseline_b64_str']
            if 'all_gan_b64_str' in st.session_state:
                del st.session_state['all_gan_b64_str']
            if 'all_dae_b64_str' in st.session_state:
                del st.session_state['all_dae_b64_str']

            with st.spinner("Generating all three videos... This may take a minute."):
                try:
                    # --- 1. BASELINE ---
                    st.toast("Generating Baseline...")
                    src, src_length, src_mask = pipeline.translate_and_tokenize(
                        text_input, text_vocab=assets["text_vocab"]
                    )
                    segments, baseline_pose = pipeline.run_text_to_baseline(
                        src, src_length, src_mask,
                        trans_model=assets["trans_model"],
                        vq_config=assets["vq_config"]
                    )
                    
                    if baseline_pose is None:
                        st.error("The model predicted an empty sequence. Please try a different sentence.")
                    else:
                        # Save baseline video
                        pipeline.save_video(
                            baseline_pose,
                            file_path=LIVE_BASELINE_VIDEO,
                            fps=assets["fps"],
                            smoothed=False
                        )
                        with open(LIVE_BASELINE_VIDEO, "rb") as f:
                            baseline_bytes = f.read()
                            # <-- 2. ENCODE TO BASE64 -->
                            st.session_state.all_baseline_b64_str = base64.b64encode(baseline_bytes).decode()
                        
                        # Store intermediate poses
                        st.session_state.pose_segments = segments
                        st.session_state.stitched_baseline_pose = baseline_pose

                        # --- 2. GAN REFINER ---
                        st.toast("Refining with GAN...")
                        gan_pose = pipeline.run_gan_refiner(
                            st.session_state.stitched_baseline_pose,
                            assets["gan_model"],
                            assets["norm_stats"],
                            assets["pose_dim"]
                        )
                        pipeline.save_video(
                            gan_pose,
                            file_path=LIVE_GAN_VIDEO,
                            fps=assets["fps"],
                            smoothed=True
                        )
                        with open(LIVE_GAN_VIDEO, "rb") as f:
                            gan_bytes = f.read()
                            # <-- 2. ENCODE TO BASE64 -->
                            st.session_state.all_gan_b64_str = base64.b64encode(gan_bytes).decode()

                        # --- 3. DIFFUSION REFINER ---
                        st.toast("Refining with Diffusion...")
                        dae_pose = pipeline.run_dae_refiner(
                            st.session_state.pose_segments,
                            assets["dae_model"],
                            assets["vq_config"],
                            assets["trans_config"]
                        )
                        pipeline.save_video(
                            dae_pose,
                            file_path=LIVE_DAE_VIDEO,
                            fps=assets["fps"],
                            smoothed=True
                        )
                        with open(LIVE_DAE_VIDEO, "rb") as f:
                            dae_bytes = f.read()
                            # <-- 2. ENCODE TO BASE64 -->
                            st.session_state.all_dae_b64_str = base64.b64encode(dae_bytes).decode()
                        
                        st.toast("All videos generated!")

                except Exception as e:
                    st.error(f"An error occurred during 'Generate All' process: {e}")
                    # Clear keys on failure
                    if 'all_baseline_b64_str' in st.session_state:
                        del st.session_state['all_baseline_b64_str']
                    if 'all_gan_b64_str' in st.session_state:
                        del st.session_state['all_gan_b64_str']
                    if 'all_dae_b64_str' in st.session_state:
                        del st.session_state['all_dae_b64_str']

        # --- 3. REPLACED DISPLAY LOGIC ---
        # Check for the new base64 string keys
        if 'all_baseline_b64_str' in st.session_state and 'all_gan_b64_str' in st.session_state and 'all_dae_b64_str' in st.session_state:
            
            # Craft the custom HTML component
            html_player = f"""
                <style>
                    .controls {{
                        text-align: center;
                        margin: 15px 0;
                        display: flex;
                        justify-content: center;
                        gap: 10px;
                    }}
                    .controls button {{
                        padding: 8px 16px;
                        font-size: 16px;
                        border: none;
                        border-radius: 5px;
                        cursor: pointer;
                        background-color: #f0f2f6;
                        color: #31333F;
                        transition: background-color 0.2s;
                    }}
                    .controls button:hover {{
                        background-color: #e6e8eb;
                    }}

                    /* --- UPDATED STYLES --- */
                    .video-container {{
                        display: flex;
                        justify-content: space-around;
                        align-items: flex-start; /* Align to top */
                        gap: 15px; /* Added a bit more gap */
                    }}
                    .video-column {{
                        flex: 1; /* Each column will take equal space */
                        min-width: 0; /* Prevent flex overflow */
                        text-align: center; /* Center the label */
                    }}
                    .video-label {{
                        text-align: center;
                        font-size: 1.1em;
                        font-weight: 600;
                        margin-bottom: 8px; /* Added margin */
                    }}
                    .video-player {{
                        width: 100%; /* Video fills its column */
                        height: auto; /* Maintain aspect ratio */
                        border-radius: 5px;
                        background-color: #000; /* Added for visual*/
                    }}
                    /* --- END UPDATED STYLES --- */
                </style>

                <div class="controls">
                    <button id="playBtn">▶️ Play All</button>
                    <button id="pauseBtn">⏸️ Pause All</button>
                    <button id="resetBtn">⏮️ Reset All</button>
                </div>

                <div class="video-container">
                    <div class="video-column">
                        <p class="video-label">Baseline (Raw)</p>
                        <video id="vid1" class="video-player" controls muted loop src="data:video/mp4;base64,{st.session_state.all_baseline_b64_str}"></video>
                    </div>
                    <div class="video-column">
                        <p class="video-label">Refined (GAN)</p>
                        <video id="vid2" class="video-player" controls muted loop src="data:video/mp4;base64,{st.session_state.all_gan_b64_str}"></video>
                    </div>
                    <div class="video-column">
                        <p class="video-label">Refined (Diffusion)</p>
                        <video id="vid3" class="video-player" controls muted loop src="data:video/mp4;base64,{st.session_state.all_dae_b64_str}"></video>
                    </div>
                </div>
                <script>
                    const vid1 = document.getElementById("vid1");
                    const vid2 = document.getElementById("vid2");
                    const vid3 = document.getElementById("vid3");
                    const videos = [vid1, vid2, vid3];

                    document.getElementById("playBtn").onclick = () => videos.forEach(v => v.play());
                    document.getElementById("pauseBtn").onclick = () => videos.forEach(v => v.pause());
                    document.getElementById("resetBtn").onclick = () => videos.forEach(v => {{
                        v.pause();
                        v.currentTime = 0;
                    }});
                </script>
            """
            
            # Render the component
            # Adjust height as needed, 450 might be too small now
            components.html(html_player, height=500)

        # -----------------------------------------------------------------
        # --- END OF NEW SECTION ---
        # -----------------------------------------------------------------

    else:
        st.error("Models are not loaded. The app cannot function. Please check the logs.")

# -----------------------------------------------------------------
# PAGE 2: MODEL SHOWCASE
# -----------------------------------------------------------------
elif page == "Model Showcase":
    st.title("Model Showcase: Pre-Generated Example")
    st.markdown("This page shows a pre-generated example of each model's output to demonstrate its specific role in the pipeline.")

    st.header("1. VQ-VAE: Pose Reconstruction")
    st.markdown("""
    The VQ-VAE's job is to create a "dictionary" of pose "words" (a codebook).
    It's trained by taking a **real** pose segment (left) and reconstructing it (right).
    This process creates the "robotic" but accurate baseline motion.
    """)
    if Path("static/vae_example.mp4").exists():
        st.video("static/vae_example.mp4")
    else:
        st.warning("static/vae_example.mp4 not found.")

    st.header("2. Transformer: Text-to-Sign (Baseline)")
    st.markdown("""
    The Transformer is the "translator." It learns to map text to a sequence of "pose words" from the VQ-VAE's dictionary. 
    When we stitch these pose words together, we get this **Baseline Video**. 
    **Note:** No smoothing is applied. This is the raw, "robotic" output.
    """)
    if Path("static/baseline_example.mp4").exists():
        vid_cols = st.columns([0.2, 0.6, 0.2])
        with vid_cols[1]:
            if Path("static/baseline_example.mp4").exists():
                st.video("static/baseline_example.mp4")
            else:
                st.warning("static/baseline_example.mp4 not found.")
    else:
        st.warning("static/baseline_example.mp4 not found.")

    st.header("3. Refinement Models (GAN vs. Diffusion)")
    st.markdown("""
    Finally, the "robotic" baseline video is fed into a refiner model. 
    This model's only job is to make the motion smoother and more realistic.
    """)
    
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Refined with GAN")
        if Path("static/gan_example.mp4").exists():
            st.video("static/gan_example.mp4")
        st.caption("The GAN refiner adds realistic 'style' and 'texture' to the motion.")
    with col2:
        st.subheader("Refined with Diffusion")
        if Path("static/dae_example.mp4").exists():
            st.video("static/dae_example.mp4")
        st.caption("The Diffusion refiner 'denoises' the robotic motion, resulting in a very smooth pose.")

# -----------------------------------------------------------------
# PAGE 3: QUANTITATIVE METRICS
# -----------------------------------------------------------------
elif page == "Quantitative Metrics":
    st.title("Quantitative Model Performance")
    st.markdown("Metrics comparing the **Baseline (Transformer)** against the two **Refined** models (GAN and Diffusion) on the test dataset.")

    st.header("Comparative Performance")
    st.markdown("This table compares the models using only the metrics that are common to all three. **Lower is better** for MPJPE and WER. **Higher is better** for BLEU.")
    
    # Data for the common metrics
    data_common = {
        'Metric': [
            'MPJPE (DTW) ↓', 'WER (Back-Translation) ↓', 'BLEU-1 (Back-Translation) ↑', 'BLEU-4 (Back-Translation) ↑'
        ],
        'Baseline (Transformer)': [
            0.0394, 97.0909, 25.9924, 8.0193
        ],
        'Refined (GAN)': [
            0.0309, 98.7441, 24.1520, 7.0871
        ],
        'Refined (Diffusion)': [
            0.0391, 95.8862, 25.8783, 7.9180
        ]
    }
    df_common = pd.DataFrame(data_common)
    
    st.dataframe(
        df_common,
        hide_index=True,
        use_container_width=True,
        column_config={
            "Baseline (Transformer)": st.column_config.NumberColumn(format="%.4f"),
            "Refined (GAN)": st.column_config.NumberColumn(format="%.4f"),
            "Refined (Diffusion)": st.column_config.NumberColumn(format="%.4f"),
        }
    )

    st.divider()

    st.header("Detailed Model Metrics")
    st.markdown("These tables show the complete test metrics for each individual model in the pipeline.")

    tab1, tab2, tab3, tab4 = st.tabs(["VQ-VAE", "Transformer (Baseline)", "GAN Refiner", "Diffusion Refiner"])

    with tab1:
        st.subheader("VQ-VAE (Pose Quantization)")
        data_vq = {
            "Metric": [
                "Test Loss",
                "Test Pose MSE",
                "Test Reconstruction Loss",
                "Test Counter MSE",
                "Test Counter Loss"
            ],
            "Value": [
                1.342257e-06,
                0.00016830,
                1.325529e-06,
                0.00212635,
                1.672819e-05
            ]
        }
        df_vq = pd.DataFrame(data_vq)
        st.dataframe(
            df_vq,
            hide_index=True,
            use_container_width=True,
            column_config={"Value": st.column_config.NumberColumn(format="%.8f")}
        )

    with tab2:
        st.subheader("Transformer (Text-to-Pose Baseline)")
        data_tf = {
            "Metric": [
                "MPJPE (DTW) ↓",
                "WER (Back-Translation) ↓",
                "BLEU-1 ↑",
                "BLEU-2 ↑",
                "BLEU-3 ↑",
                "BLEU-4 ↑"
            ],
            "Value": [
                0.0394, 97.0909, 25.9924, 14.6043, 10.2652, 8.0193
            ]
        }
        df_tf = pd.DataFrame(data_tf)
        st.dataframe(
            df_tf,
            hide_index=True,
            use_container_width=True,
            column_config={"Value": st.column_config.NumberColumn(format="%.4f")}
        )

    with tab3:
        st.subheader("GAN (Pose Refinement)")
        data_gan = {
            "Metric": [
                "MPJPE (DTW) ↓",
                "WER (Back-Translation) ↓",
                "BLEU-1 ↑",
                "BLEU-4 ↑"
            ],
            "Value": [
                0.0309, 98.7441, 24.1520, 7.0871
            ]
        }
        df_gan = pd.DataFrame(data_gan)
        st.dataframe(
            df_gan,
            hide_index=True,
            use_container_width=True,
            column_config={"Value": st.column_config.NumberColumn(format="%.4f")}
        )
    
    with tab4:
        st.subheader("Diffusion (Pose Refinement)")
        data_diff = {
            "Metric": [
                "MPJPE (DTW) ↓",
                "WER (Back-Translation) ↓",
                "BLEU-1 ↑",
                "BLEU-2 ↑",
                "BLEU-3 ↑",
                "BLEU-4 ↑"
            ],
            "Value": [
                0.0391, 95.8862, 25.8783, 14.5032, 10.1766, 7.9180
            ]
        }
        df_diff = pd.DataFrame(data_diff)
        st.dataframe(
            df_diff,
            hide_index=True,
            use_container_width=True,
            column_config={"Value": st.column_config.NumberColumn(format="%.4f")}
        )

# -----------------------------------------------------------------
# PAGE 4: MODEL COMPARISON
# -----------------------------------------------------------------
elif page == "Model Comparison":
    st.title("Model Architecture & Role Comparison")
    st.markdown("This project integrates four distinct deep learning models, each with a specialized role in the text-to-sign pipeline.")

    # We need pandas to create the transposed table
    import pandas as pd

    # Define the comparison data with models as keys (columns)
    comparison_data = {
        'VQ-VAE': {
            'Primary Role': "Pose Quantizer / Discretizer",
            'Model Type': "Vector-Quantized Variational Autoencoder",
            'Core Architecture': "1D CNN Encoder/Decoder + Discrete Codebook",
            'Operating Domain': "Continuous Pose Segments -> Discrete Latent Space",
            'Training Objective': "Reconstruct pose segments with high fidelity after forcing them through the discrete codecodebook bottleneck.",
            'Key Loss Function(s)': "L1/MSE Reconstruction Loss + Codebook Commitment Loss",
            'Inference Input': "N/A (Its learned codebook is used)",
            'Inference Output': "A 4000-vector 'Codebook' of pose micro-motions"
        },
        'Transformer': {
            'Primary Role': "Text-to-Motion Translator",
            'Model Type': "Sequence-to-Sequence (Seq2Seq) Model",
            'Core Architecture': "Encoder-Decoder Transformer",
            'Operating Domain': "Tokenized Text -> Discrete Codebook Indices",
            'Training Objective': "Predict the correct *sequence* of codebook indices (pose-words) that corresponds to a given input text.",
            'Key Loss Function(s)': "Cross-Entropy Loss",
            'Inference Input': "Tokenized text sentence",
            'Inference Output': "A sequence of integer indices (e.g., [42, 1024, 9])"
        },
        'GAN': {
            'Primary Role': "Pose Refiner (Style Transfer)",
            'Model Type': "Generative Adversarial Network",
            'Core Architecture': "Generator (1D CNN) + PatchGAN Discriminator",
            'Operating Domain': "Full, Stitched Pose (Continuous Space)",
            'Training Objective': "The Generator learns to refine poses to be indistinguishable from real poses, as judged by the Discriminator.",
            'Key Loss Function(s)': "Adversarial (BCE) Loss + L1 Identity Loss",
            'Inference Input': "Full 'robotic' baseline pose (Normalized to [-1, 1])",
            'Inference Output': "A full, 'stylized' refined pose (Normalized)"
        },
        'Diffusion': {
            'Primary Role': "Pose Refiner (Denoising)",
            'Model Type': "Diffusion Model",
            'Core Architecture': "1D U-Net (CNN-based)",
            'Operating Domain': "Un-stitched Pose Segments (Continuous Space)",
            'Training Objective': "Given a 'noisy' (VQ-quantized) pose segment, predict the original 'clean' (ground truth) pose segment.",
            'Key Loss Function(s)': "L1 / MSE Loss",
            'Inference Input': "Un-stitched 'robotic' pose segments (Un-normalized)",
            'Inference Output': "A batch of 'denoised' refined pose segments"
        }
    }
    
    # Create DataFrame from the dictionary
    df_comparison = pd.DataFrame(comparison_data)
    
    # Set the index name
    df_comparison.index.name = "Feature"
    
    # Display the transposed dataframe
    st.dataframe(df_comparison, use_container_width=True)

# -----------------------------------------------------------------
# PAGE 5: TRAINING PLOTS
# -----------------------------------------------------------------
elif page == "Training Plots":
    st.title("Training & Validation Plots")
    st.markdown("These are live, interactive plots embedded directly from Weights & Biases.")

    tab1, tab2, tab3, tab4 = st.tabs(["VQ-VAE", "Transformer", "GAN Refiner", "Diffusion Refiner"])
    
    with tab1:
        st.subheader("VQ-VAE Model")
        components.html(wandb_plots["vae"], height=700, scrolling=True)
        
    with tab2:
        st.subheader("Transformer Model")
        components.html(wandb_plots["transformer"], height=700, scrolling=True)
        
    with tab3:
        st.subheader("GAN Refiner")
        components.html(wandb_plots["gan"], height=700, scrolling=True)

    with tab4:
        st.subheader("Diffusion Refiner")
        components.html(wandb_plots["diffusion"], height=700, scrolling=True)

# -----------------------------------------------------------------
# PAGE 6: PIPELINE EXPLAINED
# -----------------------------------------------------------------
elif page == "Pipeline Explained":
    st.title("Project Pipeline Explained")
    st.markdown("This project is a multi-stage pipeline. Here is an overview of how your text becomes a video.")
    
    if Path("static/pipeline_diagram.png").exists():
        st.image("static/pipeline_diagram.png", caption="The complete Text-to-Sign generation pipeline.")
    else:
        st.warning("static/pipeline_diagram.png not found.")

    st.header("Step-by-Step Breakdown")
    
    with st.expander("Step 1: VQ-VAE (Offline Training)"):
        st.markdown("""
        This is the **Discrete Representation Learner**. The VQ-VAE (Vector-Quantized Variational Autoencoder) is trained on thousands of short, continuous 3D pose segments from the dataset.
        
        * Its goal is to learn a "bottleneck" that forces the pose data into a **discrete latent space**.
        * This process, **Vector Quantization**, maps each pose segment to the closest vector in a learned "codebook."
        * In this project, the codebook contains **4000 unique vectors**. Each vector is a "pose-word" that represents a fundamental micro-motion.
        
        This offline step is crucial: it converts complex, continuous motion into a finite vocabulary, turning the problem into a sequence-to-sequence task.
        
        
        """)
        
    with st.expander("Step 2: Transformer (Live Inference)"):
        st.markdown("""
        This is the **Sequence-to-Sequence Translator**. This model performs the cross-domain mapping from text to motion.
        
        * It uses a standard **Encoder-Decoder Transformer** architecture.
        * The **Encoder** consumes the tokenized input text, creating a high-level semantic representation of its meaning.
        * The **Decoder** attends to this representation and auto-regressively generates a new sequence. Instead of outputting text, it predicts the *sequence of indices* from the VQ-VAE's 4000-word codebook.
        
        The output is a list of integers (e.g., `[42, 1024, 9, 500, ...]`) that represents the sequence of "pose-words" needed to perform the sign.
        """)
        
    with st.expander("Step 3: Baseline Pose Generation (Live)"):
        st.markdown("""
        This is the **Index Lookup & Stitching** phase. The list of indices from the Transformer is used to generate the first draft of the video.
        
        1.  **Lookup:** Each index (e.g., `42`) is used to retrieve its corresponding vector (a 4-frame pose segment) from the VQ-VAE's codebook.
        2.  **Stitching:** The `stitch_poses` function intelligently concatenates these 4-frame segments end-to-end, handling the temporal boundaries between them.
        
        The result is the **Baseline Video**. This video is "robotic" because it's built from a finite set of parts. It suffers from **temporal discontinuity** (jerky transitions) and **quantization error** (the "pose-words" are approximations, not perfect).
        """)
        
    with st.expander("Step 4: Pose Refinement (Live)"):
        st.markdown("""
        This is the **Refinement & Stylization** stage. This is a pose-to-pose task where the "robotic" baseline is the input, and a more realistic pose is the output.
        
        * **GAN Refiner:** The **Generator** (a 1D-CNN) takes the *entire* stitched baseline pose and attempts to "re-draw" it with realistic motion artifacts. It's trained adversarially against a **PatchGAN Discriminator** that learns to distinguish real poses from refined ones.
        
        * **Diffusion Refiner:** The **Diffusion Model** was trained on (Noisy VQ-VAE Pose, Clean Real Pose) pairs. It takes the *un-stitched* baseline segments and runs them through a U-Net-like architecture to remove quantization noise. This maps the "robotic" motion back to a smoother, continuous representation.
        """)
        
    with st.expander("Step 5: Video Generation (Live)"):
        st.markdown("""
        This is the final rendering step. The **refined pose sequence** from the selected model (GAN or Diffusion) is directly compiled and encoded into the final video file. 
        
        This process assembles the individual 3D pose frames into a coherent `.mp4` video, resulting in the smooth and natural motion displayed in the app.
        """)