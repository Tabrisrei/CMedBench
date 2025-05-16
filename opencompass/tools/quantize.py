from llmcompressor.modifiers.quantization import GPTQModifier
from llmcompressor import oneshot
import os

# Define paths and settings
model_path = "/mnt/sda/sulixian/gsb/base_models/meta-llama/Meta-Llama-3-8B"
output_dir = "./Meta-Llama-3-8B-4bit"
dataset = "open_platypus"  # Calibration dataset
max_seq_length = 2048
num_calibration_samples = 512

# Ensure output directory exists
os.makedirs(output_dir, exist_ok=True)

# Define quantization recipe
recipe = [
    GPTQModifier(
        scheme="W4A16",  # 4-bit weights, 16-bit activations
        targets="Linear",  # Apply to all Linear layers
        ignore=["lm_head"],  # Skip the language model head
        bits=4,  # Explicitly set to 4-bit quantization
        quantize_bias=False,  # Do not quantize biases
        damping=0.03,  # Damping factor for GPTQ
        block_size=128,  # Block size for quantization
        disable_quip=True,  # Disable QUIP optimization for standard GPTQ
    )
]

# Perform one-shot quantization
oneshot(
    model=model_path,
    dataset=dataset,
    recipe=recipe,
    output_dir=output_dir,
    max_seq_length=max_seq_length,
    num_calibration_samples=num_calibration_samples,
)

print(f"Quantized model saved to {output_dir}")