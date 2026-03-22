"""
FLUX.2 (Klein Base 4B) image generation using diffusers.
Strategy: Load on CPU, use sequential CPU offload, manually move small
components (VAE, text_encoder) to MPS for faster inference.
"""

import torch
from diffusers import Flux2KleinPipeline

MODEL_PATH = "../models/FLUX.2-klein-base-4B"

DEVICE = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {DEVICE}")

pipe = Flux2KleinPipeline.from_pretrained(
    MODEL_PATH,
    torch_dtype=torch.bfloat16,
)
pipe.to("cpu")

# Sequential CPU offload: each module is moved to CPU when not needed,
# no device placement required (doesn't call .to(device)).
pipe.enable_sequential_cpu_offload()

# Move small, fast components to device for speed
# VAE decode is only ~80MB and benefits from acceleration
try:
    pipe.vae.to(DEVICE)
    print(f"VAE on {DEVICE}")
except Exception as e:
    print(f"VAE on {DEVICE} failed: {e}")

# Text encoder is ~1GB — move to device if there's room
try:
    pipe.text_encoder.to(DEVICE)
    print(f"Text encoder on {DEVICE}")
except Exception as e:
    print(f"Text encoder on {DEVICE} failed: {e}")


def generate(
    prompt: str,
    num_inference_steps: int = 4,
    guidance_scale: float = 3.5,
    height: int = 256,
    width: int = 256,
    seed: int | None = None,
    output_path: str = "output.png",
):
    generator = (
        [torch.Generator(device="cpu").manual_seed(seed)] if seed is not None else None
    )

    result = pipe(
        prompt=prompt,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
        height=height,
        width=width,
        generator=generator,
    )

    result.images[0].save(output_path)
    print(f"Saved: {output_path} ({height}x{width}, {num_inference_steps} steps)")
    return result.images[0]


if __name__ == "__main__":
    generate(
        prompt="A serene mountain landscape at sunset with a lake reflecting the orange sky, a bird flying to the right, a small wooden boat floating on the lake",
        num_inference_steps=20,
        guidance_scale=3.5,
        height=512,
        width=512,
        seed=42,
        output_path="flux2_output.png",
    )
