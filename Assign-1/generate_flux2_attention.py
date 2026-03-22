"""
FLUX.2 cosine similarity between word embedding and attention outputs.
Compute cosine similarity between concept word's text embedding and
the attention layer output at each spatial position.
"""

import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

from diffusers import Flux2KleinPipeline
from diffusers.models.transformers.transformer_flux2 import (
    Flux2AttnProcessor, Flux2Attention, Flux2TransformerBlock,
    _get_qkv_projections
)
from diffusers.models.attention_dispatch import dispatch_attention_fn
from diffusers.models.embeddings import apply_rotary_emb

MODEL_PATH = "../models/FLUX.2-klein-base-4B"

DEVICE = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {DEVICE}")

# Target concept and prompt
CONCEPT = "bird"
PROMPT = "A serene mountain landscape at sunset with a lake reflecting the orange sky, a bird flying to the right, a small wooden boat floating on the lake"

# Load pipeline
pipe = Flux2KleinPipeline.from_pretrained(
    MODEL_PATH,
    torch_dtype=torch.bfloat16,
)
pipe.to("cpu")

# Tokenizer and text encoder setup
tokenizer = pipe.tokenizer
tokens = tokenizer(PROMPT, return_tensors="pt", padding=True)
input_ids = tokens.input_ids[0]
print(f"\nPrompt: {PROMPT}")
print(f"Total tokens: {len(input_ids)}")

# Find token position for concept
def find_token_positions(input_ids, concept):
    positions = []
    concept_lower = concept.lower()
    n = len(input_ids)
    for start in range(n):
        for length in range(1, n - start + 1):
            tokens_here = input_ids[start:start + length]
            decoded = tokenizer.decode(tokens_here).lower().strip()
            decoded_clean = decoded.replace('<|endoftext|>', '').replace('<|pad|>', '').strip()
            if concept_lower in decoded_clean or decoded_clean in concept_lower:
                if decoded_clean == concept_lower:
                    for p in range(start, start + length):
                        if p not in positions:
                            positions.append(p)
    return positions

bird_positions = find_token_positions(input_ids, CONCEPT)
decoded = [tokenizer.decode([input_ids[p]]).strip() for p in bird_positions]
print(f"Concept '{CONCEPT}': positions={bird_positions}, decoded={decoded}")

# Enable sequential CPU offload for heavy transformer
pipe.enable_sequential_cpu_offload()

# Storage: per-head cross-attention weights
# Structure: block_id -> list of per-head text-to-image maps
per_head_attn = {}  # block_id -> list of [B, heads, text_len, img_len]


class PerHeadAttnCaptureProcessor(Flux2AttnProcessor):
    """Captures per-head cross-attention weights (text → image)."""

    def __init__(self, block_id, attn_store):
        super().__init__()
        self.block_id = block_id
        self.attn_store = attn_store

    def __call__(
        self,
        attn: "Flux2Attention",
        hidden_states: torch.Tensor,
        encoder_hidden_states=None,
        attention_mask=None,
        image_rotary_emb=None,
    ):
        query, key, value, encoder_query, encoder_key, encoder_value = _get_qkv_projections(
            attn, hidden_states, encoder_hidden_states
        )

        query = query.unflatten(-1, (attn.heads, -1))
        key = key.unflatten(-1, (attn.heads, -1))
        value = value.unflatten(-1, (attn.heads, -1))

        query = attn.norm_q(query)
        key = attn.norm_k(key)

        text_len = 0
        if attn.added_kv_proj_dim is not None:
            encoder_query = encoder_query.unflatten(-1, (attn.heads, -1))
            encoder_key = encoder_key.unflatten(-1, (attn.heads, -1))
            encoder_value = encoder_value.unflatten(-1, (attn.heads, -1))

            encoder_query = attn.norm_added_q(encoder_query)
            encoder_key = attn.norm_added_k(encoder_key)

            text_len = encoder_query.shape[1]
            img_len = hidden_states.shape[1]

            query = torch.cat([encoder_query, query], dim=1)
            key = torch.cat([encoder_key, key], dim=1)
            value = torch.cat([encoder_value, value], dim=1)

        if image_rotary_emb is not None:
            query = apply_rotary_emb(query, image_rotary_emb, sequence_dim=1)
            key = apply_rotary_emb(key, image_rotary_emb, sequence_dim=1)

        # Compute per-head attention weights manually (SDPA doesn't return weights)
        # query: [B, seq, heads, head_dim] -> [B, heads, seq, dim]
        query_p = query.permute(0, 2, 1, 3)
        key_p = key.permute(0, 2, 1, 3)
        scale = 1.0 / np.sqrt(query_p.shape[-1])
        attn_weights = torch.matmul(query_p, key_p.transpose(-2, -1)) * scale
        attn_weights = torch.softmax(attn_weights, dim=-1)  # [B, heads, seq, seq]

        # Extract image → text attention: [B, heads, img_len, text_len]
        # This answers: how does each image token attend to each text token?
        if text_len > 0 and img_len > 0:
            img_to_text = attn_weights[:, :, text_len:text_len + img_len, :text_len]
            self.attn_store[self.block_id].append(img_to_text.detach().float().cpu())

        # Run actual attention
        hidden_states_out = dispatch_attention_fn(
            query, key, value,
            attn_mask=attention_mask,
            backend=self._attention_backend,
            parallel_config=self._parallel_config,
        )
        hidden_states_out = hidden_states_out.flatten(2, 3)
        hidden_states_out = hidden_states_out.to(query.dtype)

        if encoder_hidden_states is not None:
            encoder_hidden_states_out, hidden_states_out = hidden_states_out.split_with_sizes(
                [encoder_hidden_states.shape[1], hidden_states_out.shape[1] - encoder_hidden_states.shape[1]], dim=1
            )
            encoder_hidden_states_out = attn.to_add_out(encoder_hidden_states_out)

        hidden_states_out = attn.to_out[0](hidden_states_out)
        hidden_states_out = attn.to_out[1](hidden_states_out)

        if encoder_hidden_states is not None:
            return hidden_states_out, encoder_hidden_states_out
        return hidden_states_out
        return hidden_states_out


# Install per-head capture processors on double-stream blocks
transformer = pipe.transformer

print("\nInstalling per-head attention capture processors...")
for i, block in enumerate(transformer.transformer_blocks):
    processor = PerHeadAttnCaptureProcessor(
        block_id=f"double_{i}",
        attn_store=per_head_attn
    )
    block.attn.set_processor(processor)
    per_head_attn[f"double_{i}"] = []
    print(f"  Double block {i}: heads={block.attn.heads}, dim_head={block.attn.head_dim}")

print(f"Total blocks instrumented: {len(per_head_attn)}")

# Run generation
print("\nRunning denoising with attention capture (4 steps)...")

with torch.no_grad():
    result = pipe(
        prompt=PROMPT,
        num_inference_steps=4,
        guidance_scale=3.5,
        height=512,
        width=512,
        generator=torch.Generator(device="cpu").manual_seed(42),
    )

print(f"\nDenoising complete.")

# Detect dimensions from captured data
sorted_blocks = sorted(per_head_attn.keys())
sample_maps = per_head_attn[sorted_blocks[0]]
num_steps = len(sample_maps)
sample_map = sample_maps[0]
num_heads = sample_map.shape[1]  # [B, heads, img_len, text_len] (image→text)
img_len = sample_map.shape[2]
text_len = sample_map.shape[3]
LATENT_SIZE = int(np.sqrt(img_len))

print(f"  Blocks: {len(sorted_blocks)}, Steps: {num_steps}, Heads: {num_heads}")
print(f"  Image tokens: {img_len} ({LATENT_SIZE}x{LATENT_SIZE}), Text tokens: {text_len}")
print(f"  (Capturing IMAGE → TEXT attention: each img token attends to text tokens)")
print(f"  Bird token position: {bird_positions[0]}")

# Select "bird" token position
bird_token_idx = bird_positions[0]
if bird_token_idx >= text_len:
    print(f"  WARNING: bird token idx {bird_token_idx} >= text_len {text_len}, using 0")
    bird_token_idx = 0

# Extract per-head maps for bird token across all blocks/steps
# Structure: (block, step, head) -> spatial map

# Figure 1: ALL per-head attention maps
# Total: n_blocks × n_steps × n_heads = 5 × 8 × 24 = 960 maps
print(f"\nGenerating all per-head attention maps: {len(sorted_blocks)}x{num_steps}x{num_heads}={len(sorted_blocks)*num_steps*num_heads} maps")
fig1, axes1 = plt.subplots(
    len(sorted_blocks) * num_heads, num_steps,
    figsize=(3.5 * num_steps, 3 * len(sorted_blocks) * num_heads)
)

for b_idx, block_id in enumerate(sorted_blocks):
    maps = per_head_attn[block_id]
    for step_idx in range(num_steps):
        for h_idx in range(num_heads):
            row = b_idx * num_heads + h_idx
            ax = axes1[row, step_idx]
            attn_slice = maps[step_idx][0, h_idx, :, bird_token_idx]  # [img_len] - how img token attends to bird token
            spatial = attn_slice.reshape(LATENT_SIZE, LATENT_SIZE).numpy()
            spatial = (spatial - spatial.min()) / (spatial.max() - spatial.min() + 1e-8)
            ax.imshow(spatial, cmap='hot', interpolation='bilinear')
            if row == 0:
                ax.set_title(f"Step {step_idx}", fontsize=9)
            if step_idx == 0:
                ax.set_ylabel(f"{block_id}\nH{h_idx}", fontsize=7)
            else:
                ax.set_yticklabels([])
            ax.set_xticks([])
            ax.set_xticklabels([])

plt.suptitle(f"Image→Text Attention: each pixel's attention to '{CONCEPT}'\n(Token {bird_token_idx}, rows=blocks×heads, cols=steps)", fontsize=11, y=1.01)
plt.tight_layout()
plt.savefig('img2text/attention_img2text_per_head_all_maps.png', dpi=100, bbox_inches='tight')
plt.close()
print("Saved: img2text/attention_img2text_per_head_all_maps.png")

# Figure 2: Bird attention across all heads (one row per block), averaged over steps
print("\nGenerating per-head averages over steps...")
fig2, axes2 = plt.subplots(len(sorted_blocks), 1, figsize=(4 * num_heads, 3 * len(sorted_blocks)))
if len(sorted_blocks) == 1:
    axes2 = [axes2]

for b_idx, block_id in enumerate(sorted_blocks):
    maps = per_head_attn[block_id]
    # Average over steps for each head
    head_maps = []
    for h_idx in range(num_heads):
        step_slices = [maps[s][0, h_idx, :, bird_token_idx] for s in range(num_steps)]
        avg_spatial = torch.stack(step_slices).mean(dim=0).reshape(LATENT_SIZE, LATENT_SIZE).numpy()
        avg_spatial = (avg_spatial - avg_spatial.min()) / (avg_spatial.max() - avg_spatial.min() + 1e-8)
        head_maps.append(avg_spatial)

    axes2[b_idx].imshow(
        np.concatenate(head_maps, axis=1),
        cmap='hot', interpolation='bilinear', aspect='auto'
    )
    axes2[b_idx].set_ylabel(f"{block_id}", fontsize=9)
    axes2[b_idx].set_xticks([])
    axes2[b_idx].set_yticks([])
    for h_idx in range(num_heads):
        axes2[b_idx].axvline(x=(h_idx + 1) * LATENT_SIZE - 0.5, color='white', linewidth=0.5)

plt.suptitle(f"Image→Text Attention: each pixel's attention to '{CONCEPT}' (averaged over steps)\nWhite lines separate heads", fontsize=11)
plt.tight_layout()
plt.savefig('img2text/attention_img2text_per_head_avg_steps.png', dpi=120, bbox_inches='tight')
plt.close()
print("Saved: img2text/attention_img2text_per_head_avg_steps.png")

# Figure 3: For each block (row) and denoising step (col), average ALL heads after normalizing each
# Each denoising step produces 2 forward passes (captures), so group them
print("\nGenerating block-step heatmaps (avg of 24 normalized heads, 5 blocks × 4 steps)...")
sorted_blocks = sorted(per_head_attn.keys())
n_blocks = len(sorted_blocks)
n_denoise_steps = 4
pairs_per_step = 2

fig3, axes3 = plt.subplots(n_blocks, n_denoise_steps, figsize=(5 * n_denoise_steps, 4 * n_blocks))
if n_blocks == 1:
    axes3 = axes3.reshape(1, -1)

for b_idx, block_id in enumerate(sorted_blocks):
    maps = per_head_attn[block_id]
    for step_idx in range(n_denoise_steps):
        ax = axes3[b_idx, step_idx]

        cap0 = step_idx * pairs_per_step
        cap1 = cap0 + 1

        # Collect all head heatmaps for both captures, normalize each, then average
        all_normalized = []
        max_vals = []

        for cap_idx in [cap0, cap1]:
            for h_idx in range(num_heads):
                attn_slice = maps[cap_idx][0, h_idx, :, bird_token_idx]  # [img_len]
                spatial = attn_slice.reshape(LATENT_SIZE, LATENT_SIZE).numpy()
                max_vals.append(spatial.max())
                # Normalize this head
                s_min, s_max = spatial.min(), spatial.max()
                if s_max > s_min:
                    spatial_norm = (spatial - s_min) / (s_max - s_min + 1e-8)
                else:
                    spatial_norm = spatial
                all_normalized.append(torch.from_numpy(spatial_norm))

        # Average all 24 heads (or 48 if 2 captures)
        stacked = torch.stack(all_normalized)  # [N, H, W]
        avg_heatmap = stacked.mean(dim=0).numpy()  # [H, W]

        ax.imshow(avg_heatmap, cmap='hot', interpolation='bilinear')
        if b_idx == 0:
            ax.set_title(f"Denoise Step {step_idx}", fontsize=12)
        if step_idx == 0:
            ax.set_ylabel(f"{block_id}", fontsize=11)
        ax.set_xticks([])
        ax.set_yticks([])

plt.suptitle(f"Avg of Normalized Heads per Block × Step (Image→Text, '{CONCEPT}')\n5 blocks × 4 steps = 20 heatmaps (each = avg of 24 normalized heads)", fontsize=13, y=1.02)
plt.tight_layout()
plt.savefig('img2text/best_per_block_step.png', dpi=120, bbox_inches='tight')
plt.close()
print(f"Saved: img2text/best_per_block_step.png ({n_blocks}x{n_denoise_steps}={n_blocks*n_denoise_steps} heatmaps)")

# Figure 4: Overlay of MAX across all heads/steps on image
print("\nGenerating overlay...")
all_bird_maps = []
for block_id in sorted_blocks:
    maps = per_head_attn[block_id]
    for step_idx in range(num_steps):
        for h_idx in range(num_heads):
            all_bird_maps.append(maps[step_idx][0, h_idx, :, bird_token_idx])

stacked_all = torch.stack(all_bird_maps)  # [N, img_len]
max_bird = stacked_all.max(dim=0)[0]
max_spatial = max_bird.reshape(LATENT_SIZE, LATENT_SIZE).numpy()
max_spatial = (max_spatial - max_spatial.min()) / (max_spatial.max() - max_spatial.min() + 1e-8)

fig3, ax3 = plt.subplots(1, 1, figsize=(6, 6))
gen_img = result.images[0]

attn_tensor = torch.from_numpy(max_spatial).unsqueeze(0).unsqueeze(0).float()
attn_up = F.interpolate(attn_tensor, size=(512, 512), mode='bilinear', align_corners=False)
attn_up = attn_up.squeeze().numpy()

ax3.imshow(gen_img)
ax3.imshow(attn_up, cmap='hot', alpha=0.6, interpolation='bilinear')
ax3.set_title(f"Image→Text: each pixel's attention to '{CONCEPT}'\n(MAX across {len(all_bird_maps)} head×step maps)", fontsize=11)
ax3.axis('off')
plt.tight_layout()
plt.savefig('img2text/attention_img2text_overlay.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved: attention_img2text_overlay.png")

print(f"\nDone!")
