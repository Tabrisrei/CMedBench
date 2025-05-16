import csv
import os
import sys
import time
import numpy as np
import psutil
import torch
import gc
from glob import glob
from threading import Thread
import pynvml
import uuid
import atexit
import json
from transformers import AutoTokenizer, AutoModelForCausalLM
from vllm import LLM, SamplingParams

# Set environment variables
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"


# Configure model paths and output directory
MODEL_PATHS = [
    'path/to/your/model/vllm_quant_model'
]
MODEL_PATHS = list(set(MODEL_PATHS))
LOG_DIR = "path/to/log"
os.makedirs(LOG_DIR, exist_ok=True)


# Synthesize a prompt with the given input length.
# Set batch size and prompts
batch_size = 16
real_world_prompts = [
    "Explain the difference between machine learning and deep learning in simple terms.",
    "Summarize the key points of the theory of relativity in one paragraph.",
    "Translate the following sentence into French: 'Knowledge is power, but wisdom is knowing how to use it.'",
    "What are the potential risks of artificial general intelligence?",
    "Generate a short story about a robot who wants to become human.",
    "Describe how a rocket engine works using analogies a child could understand.",
    "List three strategies to improve time management for remote workers.",
    "What is the role of mitochondria in human cells?",
    "Write a haiku about the feeling of solitude in winter.",
    "Give a brief history of the printing press and its impact on society.",
    "Explain how photosynthesis works in plants, step by step.",
    "Translate this into Chinese: 'The early bird catches the worm, but the second mouse gets the cheese.'",
    "Compare and contrast the philosophies of Plato and Aristotle.",
    "How does blockchain technology ensure security and immutability?",
    "Why is sleep important for memory consolidation in humans?",
    "Describe the structure of the United Nations and its main functions.",
]
prompts = real_world_prompts[:batch_size]

# Sampling parameters
sampling_params = {
    "temperature": 0.8,
    "top_p": 0.95,
    "max_tokens": 200,
    "stop_token_ids": []
}

def get_vram_usage(baseline_vram=0.0):
    """Get VRAM usage in MB for the current GPU using pynvml, relative to baseline."""
    try:
        pynvml.nvmlInit()
        # Map CUDA_VISIBLE_DEVICES to pynvml index
        visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(",")
        gpu_index = int(visible_devices[0])
        handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_index)
        mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
        vram_used = mem_info.used / 1024**2
        vram_delta = max(0.0, vram_used - baseline_vram)
        # print(f"Debug: Raw VRAM: {vram_used:.2f} MB, Baseline: {baseline_vram:.2f} MB, Delta: {vram_delta:.2f} MB")
        pynvml.nvmlShutdown()
        return vram_delta
    except Exception as e:
        print(f"Debug: Error getting VRAM with pynvml: {e}")
        return 0.0

def calculate_params(model_path):
    """Calculate parameter count and memory size in GB using HF on GPU.

    Args:
        model_path (str): Path to the model.

    Returns:
        tuple: (parameter count in billions, memory in GB, quantization bits).
    """
    print(f"Debug: Entering calculate_params for {model_path}")
    start_time = time.time()  # Define start_time
    try:
        total_params = 0
        quant_bits = 16  # Default to float16

        # Check quantization from config.json
        config_path = os.path.join(model_path, 'config.json')
        if os.path.exists(config_path):
            print("Debug: Reading model config")
            try:
                with open(config_path, 'r') as f:
                    config = json.load(f)
                # Check compression_config (e.g., compressed-tensors)
                if 'compression_config' in config:
                    quant_config = config['compression_config']
                    if quant_config.get('quant_method') == 'compressed-tensors':
                        for group in quant_config.get('config_groups', {}).values():
                            weights = group.get('weights', {})
                            if weights.get('type') == 'int':
                                quant_bits = weights.get('num_bits', 4)
                                print(f"Debug: Detected compressed-tensors quantization: {quant_bits}-bit")
                                break
                # Check quant_config (e.g., AWQ)
                elif 'quant_config' in config:
                    quant_config = config['quant_config']
                    if quant_config.get('quant_method') == 'awq':
                        quant_bits = quant_config.get('bits', 4)
                        print(f"Debug: Detected AWQ quantization: {quant_bits}-bit")
                # Check BitsAndBytes
                elif 'quantization_config' in config:
                    quant_config = config['quantization_config']
                    if quant_config.get('quant_method') == 'bitsandbytes':
                        quant_bits = 8 if quant_config.get('load_in_8bit') else 4
                        print(f"Debug: Detected BitsAndBytes quantization: {quant_bits}-bit")
            except Exception as e:
                print(f"Debug: Failed to read config: {e}")

        # Estimate parameters from config if possible
        if os.path.exists(config_path):
            try:
                with open(config_path, 'r') as f:
                    config = json.load(f)
                layers = config.get('num_hidden_layers', 32)
                hidden_size = config.get('hidden_size', 4096)
                vocab_size = config.get('vocab_size', 128256)
                intermediate_size = config.get('intermediate_size', 14336)
                total_params = (
                    layers * (hidden_size * hidden_size * 4 +  # Attention
                              hidden_size * intermediate_size * 2) +  # FFN
                    vocab_size * hidden_size  # Embedding
                )
                print(f"Debug: Config-based params: {total_params/1e9:.2f}B")
            except Exception as e:
                print(f"Debug: Failed to estimate params from config: {e}")

        # Load model on CPU if config-based params are unavailable
        if total_params == 0:
            print("Debug: Loading model with HF on CPU")
            try:
                hf_model = AutoModelForCausalLM.from_pretrained(
                    model_path,
                    torch_dtype=torch.bfloat16,
                    device_map='cpu',
                )
                for name, param in hf_model.named_parameters():
                    total_params += param.numel()

                # Double-check quantization via weights
                if quant_bits == 16:
                    sample_param = next(hf_model.parameters())
                    if sample_param.dtype == torch.int8:
                        quant_bits = 8
                        print("Debug: Detected 8-bit weights")
                    elif 'awq' in model_path.lower() or 'compressed-tensors' in model_path.lower():
                        quant_bits = 4
                        print("Debug: Assumed 4-bit based on model path")

                del hf_model
                gc.collect()
                torch.cuda.empty_cache()
                print(f"Debug: HF-based params: {total_params/1e9:.2f}B")
            except Exception as e:
                print(f"Debug: HF loading failed: {e}. Using default parameters")
                total_params = 8e9  # Default to 8B for LLaMA-3-8B

        # Calculate memory
        memory_bits = total_params * quant_bits
        memory_bytes = memory_bits / 8
        memory_gb = memory_bytes / (1024**3)
        param_billions = total_params / 1e9

        print(f"Debug: Calculated {param_billions:.2f}B params, {memory_gb:.2f} GB memory, {quant_bits}-bit, took {time.time() - start_time:.2f}s")
        return param_billions, memory_gb, quant_bits
    except Exception as e:
        print(f"Debug: Error calculating parameters: {e}")
        default_params = 8.0
        default_memory = 8e9 * 4 / (8 * 1024**3)  # Assume 4-bit default
        print(f"Debug: Using default: {default_params}B params, {default_memory:.2f} GB memory, 4-bit")
        return default_params, default_memory, 4
    finally:
        print("Debug: Exiting calculate_params")
        gc.collect()
        torch.cuda.empty_cache()

def estimate_flops(param_billions, completion_time, token_count):
    """Estimate FLOPs in TFLOPs based on parameter count and inference time."""
    params = param_billions * 1e9
    flops = 2 * params * token_count / 1e12
    flops_per_second = flops / completion_time if completion_time > 0 else 0
    return flops, flops_per_second

def calculate_metrics(results, total_runs):
    """Calculate average and standard deviation for metrics."""
    metric_keys = ["ttft", "completion_time", "tokens_per_second", "flops", "flops_per_second", "vram_peak"]
    metrics = {k: [r[k] for r in results if k in r] for k in metric_keys}
    
    output = {
        "successful_runs": len(results),
        "total_runs": total_runs,
        "peak_vram_used_mb": max(metrics["vram_peak"]) if metrics["vram_peak"] else 0.0,
        "peak_vram_used_gb": max(metrics["vram_peak"]) / 1024 if metrics["vram_peak"] else 0.0
    }
    
    for k in metric_keys:
        values = metrics.get(k, [])
        output[f"avg_{k}"] = float(np.mean(values)) if values else 0.0
        output[f"std_{k}"] = float(np.std(values)) if values else 0.0
    
    return output

def monitor_vram(vram_samples, stop_flag, baseline_vram):
    """Continuously monitor VRAM usage in a separate thread."""
    while not stop_flag[0]:
        vram_samples.append(get_vram_usage(baseline_vram))
        time.sleep(0.01)

# Enhanced cleanup function
def cleanup_processes(framework):
    current_pid = os.getpid()
    for proc in psutil.process_iter(['pid', 'cmdline']):
        try:
            if proc.pid == current_pid:
                continue
            cmdline = proc.info['cmdline'] or []
            if any(framework.lower() in str(arg).lower() for arg in cmdline) or 'vllm' in str(cmdline).lower():
                print(f"Terminating residual {framework} process: PID {proc.pid}")
                proc.terminate()
                try:
                    proc.wait(timeout=10)  # Increased timeout
                except psutil.TimeoutExpired:
                    print(f"Force killing process {proc.pid}")
                    proc.kill()
        except Exception as e:
            print(f"Error terminating process {proc.pid}: {e}")

def global_cleanup():
    """Global cleanup for NCCL and processes."""
    print("Debug: Running global cleanup")
    try:
        cleanup_processes("vLLM")
        cleanup_processes("SGLang")
        cleanup_processes("AutoAWQ")
        cleanup_processes("HuggingFace")
        if torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()
        torch.cuda.empty_cache()
        gc.collect()
    except Exception as e:
        print(f"Debug: Global cleanup error: {e}")

atexit.register(global_cleanup)

def calculate_params(model_path):
    """Calculate parameter count and memory size in GB using HF on GPU.

    Args:
        model_path (str): Path to the model.

    Returns:
        tuple: (parameter count in billions, memory in GB, quantization bits).
    """
    print(f"Debug: Entering calculate_params for {model_path}")
    start_time = time.time()
    try:
        total_params = 0
        quant_bits = 16  # Default to float16

        # Check quantization from config.json
        config_path = os.path.join(model_path, 'config.json')
        if os.path.exists(config_path):
            print("Debug: Reading model config")
            try:
                with open(config_path, 'r') as f:
                    config = json.load(f)
                # Check compression_config (e.g., compressed-tensors)
                if 'compression_config' in config:
                    quant_config = config['compression_config']
                    if quant_config.get('quant_method') == 'compressed-tensors':
                        for group in quant_config.get('config_groups', {}).values():
                            weights = group.get('weights', {})
                            if weights.get('type') == 'int':
                                quant_bits = weights.get('num_bits', 4)
                                print(f"Debug: Detected compressed-tensors quantization: {quant_bits}-bit")
                                break
                # Check quant_config (e.g., AWQ)
                elif 'quant_config' in config:
                    quant_config = config['quant_config']
                    if quant_config.get('quant_method') == 'awq':
                        quant_bits = quant_config.get('bits', 4)
                        print(f"Debug: Detected AWQ quantization: {quant_bits}-bit")
                # Check BitsAndBytes
                elif 'quantization_config' in config:
                    quant_config = config['quantization_config']
                    if quant_config.get('quant_method') == 'bitsandbytes':
                        quant_bits = 8 if quant_config.get('load_in_8bit') else 4
                        print(f"Debug: Detected BitsAndBytes quantization: {quant_bits}-bit")
                # Check model path for AWQ/GPTQ hints
                elif 'awq' in model_path.lower() or 'gptq' in model_path.lower():
                    quant_bits = 4
                    print(f"Debug: Assumed 4-bit quantization based on model path")
            except Exception as e:
                print(f"Debug: Failed to read config: {e}")

        # Estimate parameters from config if possible
        if os.path.exists(config_path):
            try:
                with open(config_path, 'r') as f:
                    config = json.load(f)
                layers = config.get('num_hidden_layers', 32)
                hidden_size = config.get('hidden_size', 4096)
                vocab_size = config.get('vocab_size', 128256)
                intermediate_size = config.get('intermediate_size', 14336)
                total_params = (
                    layers * (hidden_size * hidden_size * 4 +  # Attention
                              hidden_size * intermediate_size * 2) +  # FFN
                    vocab_size * hidden_size  # Embedding
                )
                print(f"Debug: Config-based params: {total_params/1e9:.2f}B")
            except Exception as e:
                print(f"Debug: Failed to estimate params from config: {e}")

        # Load model on CPU if config-based params are unavailable
        if total_params == 0:
            print("Debug: Loading model with HF on CPU")
            try:
                hf_model = AutoModelForCausalLM.from_pretrained(
                    model_path,
                    torch_dtype=torch.bfloat16,
                    device_map='cpu',
                )
                for name, param in hf_model.named_parameters():
                    total_params += param.numel()

                # Double-check quantization via weights
                if quant_bits == 16:
                    sample_param = next(hf_model.parameters())
                    if sample_param.dtype == torch.int8:
                        quant_bits = 8
                        print("Debug: Detected 8-bit weights")
                    elif 'awq' in model_path.lower() or 'gptq' in model_path.lower():
                        quant_bits = 4
                        print("Debug: Assumed 4-bit based on model path")

                del hf_model
                gc.collect()
                torch.cuda.empty_cache()
                print(f"Debug: HF-based params: {total_params/1e9:.2f}B")
            except Exception as e:
                print(f"Debug: HF loading failed: {e}. Using default parameters")
                total_params = 8e9  # Default to 8B for LLaMA-3-8B

        # Calculate memory
        memory_bits = total_params * quant_bits
        memory_bytes = memory_bits / 8
        memory_gb = memory_bytes / (1024**3)
        param_billions = total_params / 1e9

        print(f"Debug: Calculated {param_billions:.2f}B params, {memory_gb:.2f} GB memory, {quant_bits}-bit, took {time.time() - start_time:.2f}s")
        return param_billions, memory_gb, quant_bits
    except Exception as e:
        print(f"Debug: Error calculating parameters: {e}")
        default_params = 8.0
        default_memory = 8e9 * 4 / (8 * 1024**3)  # Assume 4-bit default
        print(f"Debug: Using default: {default_params}B params, {default_memory:.2f} GB memory, 4-bit")
        return default_params, default_memory, 4
    finally:
        print("Debug: Exiting calculate_params")
        gc.collect()
        torch.cuda.empty_cache()

def test_vllm(model_path, tokenizer, param_billions, memory_gb):
    """Test vLLM model performance (compatible with v0.8.2)."""
    framework = "vLLM"
    results = []
    start_time = time.time()
    print("Debug: Starting vLLM test")
    baseline_vram = get_vram_usage()
    print(f"Debug: Baseline VRAM set: {baseline_vram:.2f} MB")
    llm = None

    # Fix: Check that tokenizer is an actual tokenizer, not a bool or something else
    if not hasattr(tokenizer, 'encode'):
        print(f"Debug: Invalid tokenizer input, loading from {model_path}")
        try:
            tokenizer = AutoTokenizer.from_pretrained(
                model_path,
                use_fast=False,
                trust_remote_code=True
            )
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
                print("Debug: Set tokenizer pad_token to eos_token")
        except Exception as e:
            print(f"Debug: Failed to load tokenizer from {model_path}: {e}, falling back to default")
            tokenizer = AutoTokenizer.from_pretrained(
                "/home/gsb/LLMCMed/abase_models/meta-llama/Meta-Llama-3-8B",
                use_fast=False,
                trust_remote_code=True
            )
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
                print("Debug: Set default tokenizer pad_token to eos_token")

    try:
        for attempt in range(3):
            try:
                llm = LLM(
                    model=model_path,
                    gpu_memory_utilization=0.9,
                    dtype='auto',
                    disable_custom_all_reduce=True,
                    block_size=32,
                    num_gpu_blocks_override=1024,
                )
                print("Debug: vLLM model loaded successfully")
                break
            except Exception as e:
                print(f"Error loading {model_path} (attempt {attempt + 1}): {e}")
                if attempt < 2:
                    cleanup_processes(framework)
                    torch.cuda.empty_cache()
                    gc.collect()
                    time.sleep(2)
                else:
                    print(f"Failed to load {model_path}. Skipping.")
                    return results

        print(f"Debug: vLLM initialization took {time.time() - start_time:.2f}s")
        vllm_params = SamplingParams(**sampling_params)
        single_token_params = SamplingParams(temperature=sampling_params["temperature"], top_p=sampling_params["top_p"], max_tokens=1, stop_token_ids=[])

        print("Debug: Starting batched TTFT generation")
        vram_samples = []
        stop_flag = [False]
        vram_thread = Thread(target=monitor_vram, args=(vram_samples, stop_flag, baseline_vram))
        vram_thread.start()
        start_time = time.time()
        try:
            ttft_outputs = llm.generate(prompts, single_token_params)
            ttft_end_time = time.time()
            ttfts = [(ttft_end_time - start_time) / len(prompts) for _ in prompts]
            vram_samples.append(get_vram_usage(baseline_vram))
            print(f"Debug: Batched TTFT took {ttft_end_time - start_time:.2f}s")
        except Exception as e:
            print(f"Error measuring batched TTFT: {e}")
            ttfts = [None] * len(prompts)

        print("Debug: Starting batched full generation")
        start_time = time.time()
        try:
            outputs = llm.generate(prompts, vllm_params)
            generation_end_time = time.time()
            print(f"Debug: Batched full generation took {generation_end_time - start_time:.2f}s")
        except Exception as e:
            print(f"Error in batched full generation: {e}")
            outputs = []
        stop_flag[0] = True
        vram_thread.join()

        for i, (prompt, output) in enumerate(zip(prompts, outputs)):
            try:
                output_text = output.outputs[0].text
                completion_time = (generation_end_time - start_time) / len(prompts)
                input_len = len(tokenizer.encode(prompt))
                token_count = len(output.outputs[0].token_ids)
                if token_count != 200:
                    print(f"Warning: Generated {token_count} tokens instead of 200 for prompt {i+1}")
                tokens_per_second = token_count / completion_time if completion_time > 0 else 0
                peak_vram = max(vram_samples) if vram_samples else 0.0
                flops, flops_per_sec = estimate_flops(param_billions, completion_time, token_count)

                ttft = ttfts[i]
                if not ttft and token_count > 0:
                    ttft = completion_time / token_count

                results.append({
                    "ttft": ttft,
                    "completion_time": completion_time,
                    "tokens_per_second": tokens_per_second,
                    "vram_peak": peak_vram,
                    "flops": flops,
                    "flops_per_second": flops_per_sec
                })
                print(f"Prompt: {prompt}\nOutput: {output_text}\nTTFT: {ttft*1000:.2f}ms, Time: {completion_time:.2f}s, Tokens/s: {tokens_per_second:.2f}, VRAM Peak: {peak_vram:.2f} MB")
            except Exception as e:
                print(f"Error processing prompt '{prompt}': {e}")

    finally:
        print("Debug: Cleaning up vLLM")
        try:
            if llm is not None:
                del llm
            cleanup_processes(framework)
            torch.cuda.empty_cache()
            gc.collect()
            if torch.distributed.is_initialized():
                torch.distributed.destroy_process_group()
        except Exception as e:
            print(f"Error cleaning up {framework}: {e}")

    print(f"Debug: vLLM test completed, total time: {time.time() - start_time:.2f}s")
    return results

def main():
    all_results = []
    start_time = time.time()
    print("Debug: Starting main")
    
    # Load default tokenizer with error handling
    default_tokenizer = None
    try:
        default_tokenizer = AutoTokenizer.from_pretrained(
            "path/to/your/model",
            use_fast=False,
            trust_remote_code=True
        )
        if default_tokenizer.pad_token is None:
            default_tokenizer.pad_token = default_tokenizer.eos_token
            print("Debug: Set default tokenizer pad_token to eos_token")
    except Exception as e:
        print(f"Error loading default tokenizer: {e}. Exiting.")
        sys.exit(1)

    param_cache = {}
    for model_path in MODEL_PATHS:
        model_type = os.path.basename(model_path)
        framework = None
        if 'vllm_quant_model' in model_path.lower():
            framework = "vLLM"

        print(f"\nRunning {framework} tests for model: {model_path} ({model_type})")
        tokenizer = default_tokenizer
        try:
            print("Debug: Loading tokenizer")
            model_tokenizer = AutoTokenizer.from_pretrained(
                '/path/to/your/model',
                use_fast=False,
                trust_remote_code=True
            )
            if model_tokenizer.pad_token is None:
                model_tokenizer.pad_token = model_tokenizer.eos_token
                print("Debug: Set tokenizer pad_token to eos_token")
            tokenizer = model_tokenizer
            print("Debug: Tokenizer loaded")
        except Exception as e:
            print(f"Error loading tokenizer for {model_path}: {e}. Using default tokenizer.")

        # Ensure tokenizer is valid
        if not hasattr(tokenizer, 'encode'):
            print(f"Debug: Tokenizer is invalid for {model_path}, using default tokenizer")
            tokenizer = default_tokenizer

        print("Debug: Calculating parameters before backend")
        start_time_params = time.time()
        if model_path not in param_cache:
            param_billions, memory_gb, quant_bits = calculate_params(model_path)
            param_cache[model_path] = (param_billions, memory_gb, quant_bits)
        else:
            param_billions, memory_gb, quant_bits = param_cache[model_path]
            print(f"Debug: Using cached parameters: {param_billions}B, {memory_gb} GB, {quant_bits}-bit")
        print(f"Debug: Parameters: {param_billions}B, {memory_gb} GB, {quant_bits}-bit, took {time.time() - start_time_params:.2f}s")

        # Force cleanup before running tests
        cleanup_processes(framework)
        torch.cuda.empty_cache()
        gc.collect()

        results = []
        print(f"Debug: Starting test for framework {framework}")
        if framework == "vLLM":
            results = test_vllm(model_path, tokenizer, param_billions, memory_gb)

        if results:
            all_results.append({
                "model_path": model_path,
                "framework": framework,
                "model_type": model_type,
                "param_billions": param_billions,
                "memory_gb": memory_gb,
                "quant_bits": quant_bits,
                **calculate_metrics(results, total_runs=len(prompts))
            })

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    csv_file = os.path.join(LOG_DIR, f"performance_batch_{timestamp}.csv")
    with open(csv_file, 'w', newline='') as f:
        if all_results:
            writer = csv.DictWriter(f, fieldnames=all_results[0].keys())
            writer.writeheader()
            writer.writerows(all_results)
        else:
            print("No results to write to CSV.")

    print(f"All performance metrics written to {csv_file}")
    print(f"Debug: Main completed, total time: {time.time() - start_time:.2f}s")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("Debug: KeyboardInterrupt detected, cleaning up")
        global_cleanup()
        sys.exit(1)
    except Exception as e:
        print(f"Debug: Unexpected error in main: {e}")
        global_cleanup()
        raise
    finally:
        print("Debug: Final cleanup")
        global_cleanup()