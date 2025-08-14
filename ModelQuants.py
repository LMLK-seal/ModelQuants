#!/usr/bin/env python3
"""
HuggingFace Model Quantizer - Production Grade Version
A professional GUI application for quantizing HuggingFace models to lower bit precision.
Enhanced with optimized quantization methods, better error handling, and production features.
"""

import os
import sys
import threading
import traceback
import json
import hashlib
import shutil
import time
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple, Callable
import logging
from logging.handlers import RotatingFileHandler
from datetime import datetime
from dataclasses import dataclass
from enum import Enum
import psutil

import customtkinter as ctk
from tkinter import filedialog, messagebox
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    AutoConfig,
    pipeline
)
from accelerate import init_empty_weights, load_checkpoint_and_dispatch
from accelerate.utils import get_balanced_memory
import gc

# Configure advanced logging
class ColoredFormatter(logging.Formatter):
    COLORS = {'DEBUG': '\033[36m', 'INFO': '\033[32m', 'WARNING': '\033[33m', 'ERROR': '\033[31m', 'CRITICAL': '\033[35m', 'RESET': '\033[0m'}
    def format(self, record):
        log_color = self.COLORS.get(record.levelname, self.COLORS['RESET'])
        if sys.platform == "win32": os.system("")
        record.levelname = f"{log_color}{record.levelname}{self.COLORS['RESET']}"
        return super().format(record)

def setup_logging():
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    log_file = 'quantizer.log'
    if os.path.exists(log_file):
        try: os.remove(log_file)
        except OSError: pass
    file_handler = RotatingFileHandler(log_file, maxBytes=10*1024*1024, backupCount=5)
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s'))
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(ColoredFormatter('%(asctime)s - %(levelname)s - %(message)s'))
    if not logger.handlers:
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)
    return logger

logger = setup_logging()


def enable_perf_tweaks():
    """Enable safe perf tweaks when CUDA is available."""
    try:
        if torch.cuda.is_available():
            try:
                torch.backends.cuda.matmul.allow_tf32 = True  # TF32 for Ampere+
            except Exception:
                pass
            try:
                torch.set_float32_matmul_precision("high")
            except Exception:
                pass
    except Exception as _e:
        logger.warning(f"Perf tweaks not applied: {_e}")


class QuantizationMethod(Enum):
    NF4_RECOMMENDED = "4-bit (NF4) - Production Ready"
    NF4_HIGH_PRECISION = "4-bit (NF4) + BF16 - High Precision"
    FP4_FAST = "4-bit (FP4) - Fast Inference"
    INT4_MAX_COMPRESSION = "4-bit (Int4) - Maximum Compression"
    INT8_BALANCED = "8-bit (Int8) - Balanced Quality"
    INT8_CPU_OFFLOAD = "8-bit + CPU Offload - Large Models"
    DYNAMIC_INT8 = "Dynamic 8-bit (GPTQ-style)"
    MIXED_BF16 = "Mixed Precision (BF16) - GPU Optimized"
    MIXED_FP16 = "Mixed Precision (FP16) - Universal"
    CPU_ONLY = "CPU-Only (FP32) - No Quantization"
    EXTREME_COMPRESSION = "Extreme 4-bit (Experimental)"

@dataclass
class QuantizationConfig:
    method: QuantizationMethod
    config: Dict[str, Any]
    description: str
    memory_reduction: str
    quality: str
    supported_models: List[str]
    min_gpu_memory: str
    inference_speed: str
    stability: str
    production_ready: bool

class SystemProfiler:
    @staticmethod
    def get_system_info() -> Dict[str, Any]:
        info = {'cpu_count': psutil.cpu_count(), 'cpu_freq': psutil.cpu_freq().current if psutil.cpu_freq() else 0, 'memory_total': psutil.virtual_memory().total, 'memory_available': psutil.virtual_memory().available, 'disk_free': shutil.disk_usage('.').free, 'platform': sys.platform, 'python_version': sys.version}
        if torch.cuda.is_available():
            info['cuda_version'], info['gpu_count'], info['gpu_info'] = torch.version.cuda, torch.cuda.device_count(), []
            for i in range(torch.cuda.device_count()):
                props = torch.cuda.get_device_properties(i)
                free_mem, _ = torch.cuda.mem_get_info(i)
                info['gpu_info'].append({'name': props.name, 'memory_total': props.total_memory, 'memory_free': free_mem, 'compute_capability': f"{props.major}.{props.minor}", 'multiprocessor_count': props.multi_processor_count})
        else: info['cuda_version'], info['gpu_count'] = None, 0
        return info

    @staticmethod
    def recommend_quantization_method(model_size_gb: float, available_memory_gb: float) -> QuantizationMethod:
        if not torch.cuda.is_available():
            return QuantizationMethod.CPU_ONLY
        if model_size_gb > available_memory_gb * 0.8: return QuantizationMethod.INT4_MAX_COMPRESSION
        elif model_size_gb > available_memory_gb * 0.5: return QuantizationMethod.NF4_RECOMMENDED
        else: return QuantizationMethod.NF4_HIGH_PRECISION

class ModelQuantizer:
    QUANTIZATION_CONFIGS = {
        QuantizationMethod.NF4_RECOMMENDED: QuantizationConfig(method=QuantizationMethod.NF4_RECOMMENDED, config={"load_in_4bit": True, "bnb_4bit_quant_type": "nf4", "bnb_4bit_use_double_quant": True, "bnb_4bit_compute_dtype": torch.bfloat16, "device_map": "auto", "trust_remote_code": True}, description="Production-ready 4-bit quantization with optimal quality/size balance", memory_reduction="75%", quality="High", supported_models=["llama", "mistral", "phi", "qwen", "gemma"], min_gpu_memory="6GB", inference_speed="Fast", stability="Stable", production_ready=True),
        QuantizationMethod.NF4_HIGH_PRECISION: QuantizationConfig(method=QuantizationMethod.NF4_HIGH_PRECISION, config={"load_in_4bit": True, "bnb_4bit_quant_type": "nf4", "bnb_4bit_use_double_quant": True, "bnb_4bit_compute_dtype": torch.bfloat16, "device_map": "auto", "trust_remote_code": True, "attn_implementation": "flash_attention_2"}, description="High-precision 4-bit with Flash Attention for maximum quality", memory_reduction="70%", quality="Very High", supported_models=["llama", "mistral", "phi", "qwen"], min_gpu_memory="8GB", inference_speed="Very Fast", stability="Stable", production_ready=True),
        QuantizationMethod.FP4_FAST: QuantizationConfig(method=QuantizationMethod.FP4_FAST, config={"load_in_4bit": True, "bnb_4bit_quant_type": "fp4", "bnb_4bit_use_double_quant": True, "bnb_4bit_compute_dtype": torch.float16, "device_map": "auto", "trust_remote_code": True}, description="Fast 4-bit quantization optimized for inference speed", memory_reduction="75%", quality="Good", supported_models=["llama", "mistral", "phi", "qwen", "gemma"], min_gpu_memory="6GB", inference_speed="Very Fast", stability="Stable", production_ready=True),
        QuantizationMethod.INT4_MAX_COMPRESSION: QuantizationConfig(method=QuantizationMethod.INT4_MAX_COMPRESSION, config={"load_in_4bit": True, "bnb_4bit_quant_type": "nf4", "bnb_4bit_use_double_quant": False, "bnb_4bit_compute_dtype": torch.float16, "device_map": "auto", "trust_remote_code": True, "max_memory": {0: "8GiB"}, "offload_folder": "temp_offload"}, description="Maximum compression 4-bit for memory-constrained environments", memory_reduction="80%", quality="Good", supported_models=["llama", "mistral", "phi", "qwen", "gemma"], min_gpu_memory="4GB", inference_speed="Fast", stability="Stable", production_ready=True),
        QuantizationMethod.INT8_BALANCED: QuantizationConfig(method=QuantizationMethod.INT8_BALANCED, config={"load_in_8bit": True, "device_map": "auto", "trust_remote_code": True}, description="8-bit quantization with excellent quality preservation", memory_reduction="50%", quality="Very High", supported_models=["llama", "mistral", "phi", "qwen", "gemma", "falcon"], min_gpu_memory="8GB", inference_speed="Fast", stability="Very Stable", production_ready=True),
        QuantizationMethod.INT8_CPU_OFFLOAD: QuantizationConfig(method=QuantizationMethod.INT8_CPU_OFFLOAD, config={"load_in_8bit": True, "device_map": "auto", "offload_folder": "cpu_offload", "offload_state_dict": True, "trust_remote_code": True}, description="8-bit with CPU offloading for very large models", memory_reduction="60%", quality="Very High", supported_models=["llama", "mistral", "falcon", "bloom"], min_gpu_memory="6GB", inference_speed="Moderate", stability="Stable", production_ready=True),
        QuantizationMethod.DYNAMIC_INT8: QuantizationConfig(method=QuantizationMethod.DYNAMIC_INT8, config={"load_in_8bit": True, "device_map": "auto", "trust_remote_code": True}, description="GPTQ-style dynamic 8-bit quantization (currently uses standard 8-bit).", memory_reduction="50%", quality="High (Experimental)", supported_models=["llama", "mistral", "phi", "falcon"], min_gpu_memory="8GB", inference_speed="Fast", stability="Experimental", production_ready=False),
        QuantizationMethod.MIXED_BF16: QuantizationConfig(method=QuantizationMethod.MIXED_BF16, config={"torch_dtype": torch.bfloat16, "device_map": "auto", "trust_remote_code": True, "attn_implementation": "flash_attention_2"}, description="BFloat16 mixed precision with Flash Attention", memory_reduction="50%", quality="Very High", supported_models=["llama", "mistral", "phi", "qwen", "gemma", "falcon"], min_gpu_memory="12GB", inference_speed="Very Fast", stability="Very Stable", production_ready=True),
        QuantizationMethod.MIXED_FP16: QuantizationConfig(method=QuantizationMethod.MIXED_FP16, config={"torch_dtype": torch.float16, "device_map": "auto", "trust_remote_code": True}, description="Half precision floating point for universal compatibility", memory_reduction="50%", quality="High", supported_models=["llama", "mistral", "phi", "qwen", "gemma", "falcon", "bloom"], min_gpu_memory="10GB", inference_speed="Very Fast", stability="Very Stable", production_ready=True),
        QuantizationMethod.CPU_ONLY: QuantizationConfig(method=QuantizationMethod.CPU_ONLY, config={"device_map": {"": "cpu"}, "torch_dtype": torch.float32, "trust_remote_code": True}, description="Loads the model on the CPU in full precision (FP32). This is not a quantization method but a loading strategy for machines without a dedicated GPU.", memory_reduction="0%", quality="Full (Original)", supported_models=["all"], min_gpu_memory="N/A", inference_speed="Slow", stability="Very Stable", production_ready=True),
        QuantizationMethod.EXTREME_COMPRESSION: QuantizationConfig(method=QuantizationMethod.EXTREME_COMPRESSION, config={"load_in_4bit": True, "bnb_4bit_quant_type": "fp4", "bnb_4bit_use_double_quant": False, "bnb_4bit_compute_dtype": torch.float16, "device_map": "auto", "trust_remote_code": True, "max_memory": {0: "7GiB"}, "offload_folder": "extreme_offload"}, description="Experimental extreme compression - use with caution", memory_reduction="85%", quality="Experimental", supported_models=["llama", "mistral"], min_gpu_memory="3GB", inference_speed="Moderate", stability="Experimental", production_ready=False)
    }

    @staticmethod
    def validate_model_compatibility(model_path: str, quantization_method: QuantizationMethod) -> Tuple[bool, str]:
        try:
            config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
            model_type = getattr(config, 'model_type', '').lower()
            quant_config = ModelQuantizer.QUANTIZATION_CONFIGS[quantization_method]
            if quant_config.supported_models != ["all"] and not any(supported in model_type for supported in quant_config.supported_models):
                return False, f"Model type '{model_type}' may not be fully supported with {quantization_method.value}"
            if quantization_method in [QuantizationMethod.NF4_HIGH_PRECISION, QuantizationMethod.MIXED_BF16]:
                try: import flash_attn; logger.info("Flash Attention detected")
                except ImportError: logger.warning("Flash Attention not available")
            return True, f"Model '{model_type}' is compatible with {quantization_method.value}"
        except Exception as e: return False, f"Error validating model compatibility: {str(e)}"

    @staticmethod
    def estimate_quantization_time(param_count_billions: float, quantization_method: QuantizationMethod) -> str:
        if not param_count_billions or param_count_billions <= 0: return "N/A"
        base_time = {QuantizationMethod.NF4_RECOMMENDED: 0.5, QuantizationMethod.NF4_HIGH_PRECISION: 0.6, QuantizationMethod.FP4_FAST: 0.4, QuantizationMethod.INT4_MAX_COMPRESSION: 0.5, QuantizationMethod.INT8_BALANCED: 0.3, QuantizationMethod.INT8_CPU_OFFLOAD: 1.0, QuantizationMethod.DYNAMIC_INT8: 0.4, QuantizationMethod.MIXED_BF16: 0.2, QuantizationMethod.MIXED_FP16: 0.2, QuantizationMethod.CPU_ONLY: 1.5, QuantizationMethod.EXTREME_COMPRESSION: 0.7}
        time_minutes = param_count_billions * base_time.get(quantization_method, 0.5)
        if time_minutes < 1: return "< 1 minute"
        elif time_minutes < 60: return f"~{int(round(time_minutes))} minutes"
        else: return f"~{int(time_minutes // 60)}h {int(time_minutes % 60)}m"

    @staticmethod
    def create_backup_config(model_path: str, output_path: str) -> str:
        backup_dir = Path(output_path) / "original_backup"
        backup_dir.mkdir(exist_ok=True)
        for config_file in ["config.json", "tokenizer_config.json", "generation_config.json"]:
            if (src := Path(model_path) / config_file).exists():
                shutil.copy2(src, backup_dir / config_file); logger.info(f"Backed up {config_file}")
        return str(backup_dir)

    @staticmethod
    def verify_quantized_model(model_path: str, quantization_method: QuantizationMethod) -> Tuple[bool, str]:
        try:
            logger.info("Starting model verification...")
            quant_config_data = ModelQuantizer.QUANTIZATION_CONFIGS[quantization_method].config

            is_bnb_quant = "load_in_4bit" in quant_config_data or "load_in_8bit" in quant_config_data
            if is_bnb_quant and not torch.cuda.is_available():
                msg = "Verification SKIPPED: BitsAndBytes quantization requires a CUDA GPU to load."
                logger.warning(msg)
                return True, msg

            tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
            if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token

            # Base config, works for both quantized and non-quantized models
            test_config = {"trust_remote_code": True, "device_map": quant_config_data.get("device_map", "auto")}
            
            # Add quantization config ONLY if it's a bnb method
            if is_bnb_quant:
                bnb_config_dict = {k: v for k, v in quant_config_data.items() if k.startswith('bnb_') or 'load_in' in k}
                if quantization_method == QuantizationMethod.INT8_CPU_OFFLOAD:
                     bnb_config_dict['llm_int8_enable_fp32_cpu_offload'] = True
                test_config["quantization_config"] = BitsAndBytesConfig(**bnb_config_dict)
                logger.info(f"Verifying with Quantization Config: {test_config['quantization_config'].to_dict()}")
            else: # For CPU_ONLY (FP32)
                test_config["torch_dtype"] = quant_config_data.get("torch_dtype", torch.float32)

            logger.info(f"Verifying with config: {test_config}")
            model = AutoModelForCausalLM.from_pretrained(model_path, **test_config)
            model.eval()

            inputs = tokenizer("Hello, this is a test", return_tensors="pt")
            if "cpu" in str(test_config.get("device_map", "")):
                 logger.info("Moving verification inputs to CPU.")
                 inputs = inputs.to("cpu")

            with torch.inference_mode():
                outputs = model.generate(inputs.input_ids, max_length=inputs.input_ids.shape[1] + 10, do_sample=False, pad_token_id=tokenizer.eos_token_id)

            logger.info(f"Verification successful. Generated: {tokenizer.decode(outputs[0], skip_special_tokens=True)[:50]}...")
            del model, tokenizer; gc.collect()
            if torch.cuda.is_available(): torch.cuda.empty_cache()
            return True, "Model verification successful"
        except Exception as e:
            logger.error(f"Model verification failed: {str(e)}"); logger.error(traceback.format_exc())
            return False, f"Verification failed: {str(e)}"

    @staticmethod
    def quantize_model(model_path: str, output_path: str, quantization_method: QuantizationMethod, progress_callback: Optional[Callable[[str, float], None]] = None, enable_optimizations: bool = True, create_backup: bool = True, verify_output: bool = True) -> Tuple[bool, str]:
        start_time = time.time()
        try:
            if not os.path.exists(model_path): return False, f"Model path does not exist: {model_path}"
            quant_config = ModelQuantizer.QUANTIZATION_CONFIGS[quantization_method]
            
            # Guardrail for BitsAndBytes methods on non-CUDA systems
            is_bnb_quant = "load_in_4bit" in quant_config.config or "load_in_8bit" in quant_config.config
            if is_bnb_quant and not torch.cuda.is_available():
                raise RuntimeError(
                    "BitsAndBytes quantization (4-bit or 8-bit) requires a CUDA-enabled GPU. "
                    "Please select the 'CPU-Only (FP32)' method or run on a machine with CUDA."
                )

            if progress_callback: progress_callback("Initializing...", 0.05)
            
            is_compatible, compat_msg = ModelQuantizer.validate_model_compatibility(model_path, quantization_method)
            if not is_compatible: logger.warning(compat_msg)
            os.makedirs(output_path, exist_ok=True)
            backup_path = ModelQuantizer.create_backup_config(model_path, output_path) if create_backup else None
            config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
            tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, use_fast=True)
            if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token
            if progress_callback: progress_callback("Preparing config...", 0.25)
            
            model_kwargs = quant_config.config.copy()
            if is_bnb_quant:
                bnb_config_dict = {k: v for k, v in quant_config.config.items() if k.startswith('bnb_') or 'load_in' in k}
                if quantization_method == QuantizationMethod.INT8_CPU_OFFLOAD:
                    bnb_config_dict['llm_int8_enable_fp32_cpu_offload'] = True
                model_kwargs['quantization_config'] = BitsAndBytesConfig(**bnb_config_dict)

            if enable_optimizations and quantization_method != QuantizationMethod.CPU_ONLY:
                model_kwargs.update({'use_cache': True, 'use_safetensors': True})
                if quantization_method in [QuantizationMethod.NF4_HIGH_PRECISION, QuantizationMethod.MIXED_BF16]:
                    try: import flash_attn; model_kwargs['attn_implementation'] = 'flash_attention_2'; logger.info("Flash Attention 2 enabled")
                    except ImportError: logger.warning("Flash Attention not available"); model_kwargs.pop('attn_implementation', None)

            if progress_callback: progress_callback("Loading model...", 0.30)
            logger.info(f"Loading model with config: {quantization_method.value}")
            
            model = AutoModelForCausalLM.from_pretrained(model_path, **model_kwargs)
                
            if enable_optimizations and hasattr(model, 'gradient_checkpointing_enable'): model.gradient_checkpointing_enable(); logger.info("Gradient checkpointing enabled")
            
            if progress_callback: progress_callback("Saving model...", 0.70)
            model.save_pretrained(save_directory=output_path, safe_serialization=True, max_shard_size="5GB"); logger.info("Model saved")
            
            if hasattr(model, "quantization_config") and model.quantization_config:
                model.quantization_config.save_pretrained(output_path)
                logger.info("Quantization config saved to output directory.")

            if progress_callback: progress_callback("Saving tokenizer & metadata...", 0.85)
            tokenizer.save_pretrained(output_path)
            
            quantization_metadata = {"quantization_method": quantization_method.value, "quantization_config": {k: str(v) for k, v in quant_config.config.items()}, "model_info": {"model_type": getattr(config, 'model_type', 'unknown'), "hidden_size": getattr(config, 'hidden_size', 'unknown'), "num_layers": getattr(config, 'num_hidden_layers', 'unknown'), "vocab_size": getattr(config, 'vocab_size', 'unknown')}, "quantization_stats": {"memory_reduction": quant_config.memory_reduction, "quality": quant_config.quality, "inference_speed": quant_config.inference_speed, "stability": quant_config.stability, "production_ready": quant_config.production_ready}, "system_info": SystemProfiler.get_system_info(), "timestamp": datetime.now().isoformat(), "processing_time_seconds": time.time() - start_time, "optimizations_enabled": enable_optimizations, "backup_created": backup_path is not None, "backup_path": backup_path, "compatibility_check": compat_msg}
            with open(Path(output_path) / "quantization_info.json", "w") as f: json.dump(quantization_metadata, f, indent=2, default=str)
            
            if verify_output:
                if progress_callback: progress_callback("Verifying model...", 0.95)
                success, msg = ModelQuantizer.verify_quantized_model(output_path, quantization_method)
                quantization_metadata["verification"] = {"success": success, "message": msg}
                with open(Path(output_path) / "quantization_info.json", "w") as f: json.dump(quantization_metadata, f, indent=2, default=str)
            
            del model, tokenizer, config; gc.collect()
            if torch.cuda.is_available(): torch.cuda.empty_cache()
            if progress_callback: progress_callback("Process complete!", 1.0)
            return True, f"Process completed in {time.time() - start_time:.1f}s"
        except Exception as e:
            error_msg = f"Process failed: {str(e)}"
            logger.error(f"{error_msg}\n{traceback.format_exc()}")
            if progress_callback: progress_callback(f"Error: {error_msg}", 0.0)
            return False, error_msg

enable_perf_tweaks()

class ModelQuantizerGUI:
    def __init__(self):
        ctk.set_appearance_mode("dark"); ctk.set_default_color_theme("blue")
        self.root = ctk.CTk(); self.root.title("ModelQuantizer Pro"); self.root.geometry("1200x1200"); self.root.minsize(1000, 800)
        self.system_info = SystemProfiler.get_system_info()
        self.model_path = ctk.StringVar(); self.output_path = ctk.StringVar()
        
        default_method = QuantizationMethod.NF4_RECOMMENDED.value
        if not torch.cuda.is_available():
            default_method = QuantizationMethod.CPU_ONLY.value
        self.quantization_method = ctk.StringVar(value=default_method)

        self.enable_optimizations = ctk.BooleanVar(value=True); self.create_backup = ctk.BooleanVar(value=True); self.verify_output = ctk.BooleanVar(value=True)
        self.is_processing = False; self.current_model_info = {}
        self.setup_ui(); self.center_window(); self.display_system_info()
        logger.info("ModelQuantizer Pro GUI initialized")

    def center_window(self):
        self.root.update_idletasks()
        w, h = self.root.winfo_width(), self.root.winfo_height()
        sw, sh = self.root.winfo_screenwidth(), self.root.winfo_screenheight()
        self.root.geometry(f"+{ (sw // 2) - (w // 2)}+{max(50, (sh // 2) - (h // 2) - 50)}")

    def setup_ui(self):
        self.main_frame = ctk.CTkScrollableFrame(self.root); self.main_frame.pack(fill="both", expand=True, padx=15, pady=15)
        self.setup_header_section(self.main_frame)
        self.setup_system_info_section(self.main_frame)
        self.setup_model_input_section(self.main_frame)
        self.setup_output_section(self.main_frame)
        self.setup_advanced_quantization_section(self.main_frame)
        self.setup_model_analysis_section(self.main_frame)
        self.setup_advanced_options_section(self.main_frame)
        self.setup_progress_section(self.main_frame)
        self.setup_status_bar()
        self.on_quantization_method_change(self.quantization_method.get())

    def setup_header_section(self, parent):
        header = ctk.CTkFrame(parent, fg_color="transparent"); header.pack(fill="x", padx=10, pady=(10, 20))
        ctk.CTkLabel(header, text="ModelQuantizer Pro", font=("Arial", 28, "bold"), text_color="#00D4FF").pack(pady=(10, 5))
        ctk.CTkLabel(header, text="Production-Grade AI Model Quantization Suite", font=("Arial", 14), text_color="#A0A0A0").pack(pady=(0, 10))

    def setup_system_info_section(self, parent):
        frame = ctk.CTkFrame(parent); frame.pack(fill="x", padx=10, pady=(0, 15))
        ctk.CTkLabel(frame, text="System Information:", font=("Arial", 16, "bold")).pack(anchor="w", padx=20, pady=(15, 5))
        self.system_info_text = ctk.CTkTextbox(frame, height=100, font=("Consolas", 11), fg_color="#1E1E1E", border_color="#404040")
        self.system_info_text.pack(fill="x", padx=20, pady=(0, 15)); self.system_info_text.configure(state="disabled")

    def setup_model_input_section(self, parent):
        frame = ctk.CTkFrame(parent); frame.pack(fill="x", padx=10, pady=(0, 15))
        header = ctk.CTkFrame(frame, fg_color="transparent"); header.pack(fill="x", padx=20, pady=(15, 5))
        ctk.CTkLabel(header, text="Model Input Path:", font=("Arial", 16, "bold")).pack(side="left")
        self.model_status_label = ctk.CTkLabel(header, text="", font=("Arial", 12), text_color="#FF6B6B"); self.model_status_label.pack(side="right")
        path_frame = ctk.CTkFrame(frame, fg_color="transparent"); path_frame.pack(fill="x", padx=20, pady=(0, 10))
        self.model_entry = ctk.CTkEntry(path_frame, textvariable=self.model_path, placeholder_text="Select or enter HuggingFace model folder path...", height=40, font=("Arial", 12))
        self.model_entry.pack(side="left", fill="x", expand=True, padx=(0, 10))
        ctk.CTkButton(path_frame, text="📁 Browse", command=self.browse_model_folder, width=120, height=40, font=("Arial", 12, "bold"), hover_color="#2B5CE6").pack(side="right")
        quick_frame = ctk.CTkFrame(frame, fg_color="transparent"); quick_frame.pack(fill="x", padx=20, pady=(0, 15))
        ctk.CTkLabel(quick_frame, text="Quick Access:", font=("Arial", 12, "bold")).pack(side="left", padx=(0, 10))
        ctk.CTkButton(quick_frame, text="HF Cache", command=self.browse_hf_cache, width=80, height=30, font=("Arial", 10)).pack(side="left")
        self.model_path.trace_add("write", self.on_model_path_change)

    def setup_output_section(self, parent):
        frame = ctk.CTkFrame(parent); frame.pack(fill="x", padx=10, pady=(0, 15))
        ctk.CTkLabel(frame, text="Output Configuration:", font=("Arial", 16, "bold")).pack(anchor="w", padx=20, pady=(15, 5))
        path_frame = ctk.CTkFrame(frame, fg_color="transparent"); path_frame.pack(fill="x", padx=20, pady=(0, 10))
        self.output_entry = ctk.CTkEntry(path_frame, textvariable=self.output_path, placeholder_text="Output folder for processed model...", height=40, font=("Arial", 12))
        self.output_entry.pack(side="left", fill="x", expand=True, padx=(0, 10))
        ctk.CTkButton(path_frame, text="📁 Browse", command=self.browse_output_folder, width=120, height=40, font=("Arial", 12, "bold"), hover_color="#2B5CE6").pack(side="right")
        auto_frame = ctk.CTkFrame(frame, fg_color="transparent"); auto_frame.pack(fill="x", padx=20, pady=(0, 15))
        self.auto_output_checkbox = ctk.CTkCheckBox(auto_frame, text="Auto-generate output path", font=("Arial", 12), command=self.on_auto_output_change)
        self.auto_output_checkbox.pack(side="left"); self.auto_output_checkbox.select()

    def setup_advanced_quantization_section(self, parent):
        frame = ctk.CTkFrame(parent); frame.pack(fill="x", padx=10, pady=(0, 15))
        header = ctk.CTkFrame(frame, fg_color="transparent"); header.pack(fill="x", padx=20, pady=(15, 5))
        ctk.CTkLabel(header, text="Processing Method:", font=("Arial", 16, "bold")).pack(side="left")
        self.recommend_btn = ctk.CTkButton(header, text="🎯 Auto-Recommend", command=self.auto_recommend_quantization, width=140, height=30, font=("Arial", 11), fg_color="#28A745", hover_color="#1E7B34"); self.recommend_btn.pack(side="right")
        method_frame = ctk.CTkFrame(frame, fg_color="transparent"); method_frame.pack(fill="x", padx=20, pady=(5, 10))
        self.quant_menu = ctk.CTkOptionMenu(method_frame, values=[m.value for m in QuantizationMethod], variable=self.quantization_method, height=40, font=("Arial", 12), command=self.on_quantization_method_change)
        self.quant_menu.pack(side="left", fill="x", expand=True, padx=(0, 10))
        self.production_indicator = ctk.CTkLabel(method_frame, text="", font=("Arial", 12, "bold"), width=120); self.production_indicator.pack(side="right")
        info_frame = ctk.CTkFrame(frame); info_frame.pack(fill="x", padx=20, pady=(0, 15))
        self.method_info_text = ctk.CTkTextbox(info_frame, height=120, font=("Consolas", 11), fg_color="#1E1E1E", border_color="#404040")
        self.method_info_text.pack(fill="x", padx=15, pady=15); self.method_info_text.configure(state="disabled")

    def setup_model_analysis_section(self, parent):
        frame = ctk.CTkFrame(parent); frame.pack(fill="x", padx=10, pady=(0, 15))
        ctk.CTkLabel(frame, text="Model Analysis & Size Estimation:", font=("Arial", 16, "bold")).pack(anchor="w", padx=20, pady=(15, 5))
        self.analysis_text = ctk.CTkTextbox(frame, height=100, font=("Consolas", 11), fg_color="#1E1E1E", border_color="#404040")
        self.analysis_text.pack(fill="x", padx=20, pady=(0, 15)); self.analysis_text.configure(state="disabled")

    def setup_advanced_options_section(self, parent):
        frame = ctk.CTkFrame(parent); frame.pack(fill="x", padx=10, pady=(0, 15))
        ctk.CTkLabel(frame, text="Advanced Options:", font=("Arial", 16, "bold")).pack(anchor="w", padx=20, pady=(15, 10))
        grid = ctk.CTkFrame(frame, fg_color="transparent"); grid.pack(fill="x", padx=20, pady=(0, 15))
        self.optimization_checkbox = ctk.CTkCheckBox(grid, text="Enable Optimizations", variable=self.enable_optimizations, font=("Arial", 12)); self.optimization_checkbox.pack(side="left", padx=(0, 20))
        self.backup_checkbox = ctk.CTkCheckBox(grid, text="Create Backup", variable=self.create_backup, font=("Arial", 12)); self.backup_checkbox.pack(side="left", padx=(0, 20))
        self.verify_checkbox = ctk.CTkCheckBox(grid, text="Verify Processed Model", variable=self.verify_output, font=("Arial", 12)); self.verify_checkbox.pack(side="left")

    def setup_progress_section(self, parent):
        frame = ctk.CTkFrame(parent); frame.pack(fill="x", padx=10, pady=(0, 15))
        ctk.CTkLabel(frame, text="Processing Control:", font=("Arial", 16, "bold")).pack(anchor="w", padx=20, pady=(15, 10))
        progress_frame = ctk.CTkFrame(frame); progress_frame.pack(fill="x", padx=20, pady=(0, 15))
        header = ctk.CTkFrame(progress_frame, fg_color="transparent"); header.pack(fill="x", padx=15, pady=(15, 5))
        self.progress_label = ctk.CTkLabel(header, text="Ready to process...", font=("Arial", 13), anchor="w"); self.progress_label.pack(side="left")
        self.progress_percentage = ctk.CTkLabel(header, text="0%", font=("Arial", 13, "bold"), anchor="e"); self.progress_percentage.pack(side="right")
        self.progress = ctk.CTkProgressBar(progress_frame, height=20, border_width=2); self.progress.pack(fill="x", padx=15, pady=(0, 10)); self.progress.set(0)
        self.time_estimate_label = ctk.CTkLabel(progress_frame, text="Estimated Time: N/A", font=("Arial", 11), text_color="#A0A0A0"); self.time_estimate_label.pack(padx=15, pady=(0, 15))
        btn_frame = ctk.CTkFrame(frame, fg_color="transparent"); btn_frame.pack(pady=(0, 20))
        self.quantize_btn = ctk.CTkButton(btn_frame, text="🚀 Start Processing", command=self.start_quantization, height=50, width=200, font=("Arial", 16, "bold"), fg_color="#28A745", hover_color="#1E7B34"); self.quantize_btn.pack(side="left", padx=(20, 15))
        self.cancel_btn = ctk.CTkButton(btn_frame, text="⛔ Cancel", command=self.cancel_quantization, height=50, width=120, font=("Arial", 14, "bold"), fg_color="#DC3545", hover_color="#A71E2A", state="disabled"); self.cancel_btn.pack(side="left")

    def setup_status_bar(self):
        frame = ctk.CTkFrame(self.root, height=40, corner_radius=0); frame.pack(side="bottom", fill="x"); frame.pack_propagate(False)
        self.status_label = ctk.CTkLabel(frame, text="Ready - ModelQuantizer Pro", font=("Arial", 11), anchor="w"); self.status_label.pack(side="left", padx=15, pady=10)
        self.system_stats_label = ctk.CTkLabel(frame, text="", font=("Arial", 10), anchor="e", text_color="#A0A0A0"); self.system_stats_label.pack(side="right", padx=15, pady=10); self.update_system_stats()

    def display_system_info(self):
        lines = [f"CPU: {self.system_info['cpu_count']} cores | RAM: {self.system_info['memory_available']/(1024**3):.1f}GB / {self.system_info['memory_total']/(1024**3):.1f}GB"]
        if self.system_info['gpu_count'] > 0:
            lines.append(f"CUDA: v{self.system_info['cuda_version']} | GPUs: {self.system_info['gpu_count']}")
            for i, gpu in enumerate(self.system_info['gpu_info']): lines.append(f"  GPU {i}: {gpu['name']} ({gpu['memory_total']/(1024**3):.1f}GB)")
        else: lines.append("GPU: No CUDA devices detected")
        self.system_info_text.configure(state="normal"); self.system_info_text.delete("1.0", "end"); self.system_info_text.insert("1.0", "\n".join(lines)); self.system_info_text.configure(state="disabled")

    def update_system_stats(self):
        try:
            gpu_text = ""
            if torch.cuda.is_available() and self.system_info['gpu_count'] > 0:
                free, total = torch.cuda.mem_get_info(0); gpu_text = f" | GPU Mem: {(total - free)/total*100:.0f}%"
            self.system_stats_label.configure(text=f"RAM: {psutil.virtual_memory().percent:.0f}%{gpu_text}")
            self.root.after(5000, self.update_system_stats)
        except Exception as e: logger.error(f"Error updating stats: {e}"); self.root.after(10000, self.update_system_stats)

    def browse_model_folder(self):
        if folder := filedialog.askdirectory(title="Select Model Folder"): self.model_path.set(folder)

    def browse_output_folder(self):
        if folder := filedialog.askdirectory(title="Select Output Folder"): self.output_path.set(folder); self.auto_output_checkbox.deselect()

    def browse_hf_cache(self):
        try:
            from huggingface_hub import HfFolder
            if (models_dir := Path(HfFolder.get_cache_dir()) / 'models').exists():
                if folder := filedialog.askdirectory(title="Select Model from Cache", initialdir=models_dir): self.model_path.set(folder)
            else: messagebox.showinfo("Info", "HuggingFace cache not found.")
        except Exception as e: messagebox.showerror("Error", f"Could not access HF cache: {e}")

    def on_model_path_change(self, *args):
        if os.path.isdir(path := self.model_path.get()):
            self.model_status_label.configure(text=""); self.current_model_info = self.get_enhanced_model_info(path)
            self.update_analysis_display(); self.update_time_estimate()
            if self.auto_output_checkbox.get(): self.update_auto_output_path()
        else: self.model_status_label.configure(text="Invalid Path", text_color="#FF6B6B"); self.current_model_info = {}; self.update_analysis_display()

    def get_enhanced_model_info(self, path):
        try:
            config = json.load(open(Path(path) / "config.json", 'r', encoding='utf-8'))
            params = config.get('num_parameters') or (config.get('num_hidden_layers', 0) * (12 * config.get('hidden_size', 0)**2) + config.get('vocab_size', 0) * config.get('hidden_size', 0) if 'hidden_size' in config else None)
            return {'model_type': config.get('model_type', 'N/A'), 'param_count': params, 'disk_size_gb': sum(f.stat().st_size for f in Path(path).glob('**/*') if f.is_file()) / (1024**3)}
        except Exception as e: logger.error(f"Failed to analyze model: {e}"); return {"error": f"Failed to analyze model: {e}"}

    def update_analysis_display(self):
        lines = ["No model selected."]
        if self.current_model_info and "error" not in self.current_model_info:
            lines = [f"Model Type:     {self.current_model_info.get('model_type', 'N/A')}"]
            if isinstance(params := self.current_model_info.get('param_count'), (int, float)): lines.append(f"Parameters:     ~{params/1e9:.2f}B")
            else: lines.append("Parameters:     N/A")
            lines.append(f"Size on Disk:   {self.current_model_info.get('disk_size_gb', 0):.2f} GB")
        elif self.current_model_info: lines = [self.current_model_info["error"]]
        self.analysis_text.configure(state="normal"); self.analysis_text.delete("1.0", "end"); self.analysis_text.insert("1.0", "\n".join(lines)); self.analysis_text.configure(state="disabled")

    def on_quantization_method_change(self, method_value):
        method = next(m for m in QuantizationMethod if m.value == method_value)
        config = ModelQuantizer.QUANTIZATION_CONFIGS[method]
        self.method_info_text.configure(state="normal"); self.method_info_text.delete("1.0", "end"); self.method_info_text.insert("1.0", f"Description: {config.description}\n\nMemory Reduction: {config.memory_reduction: <20} Quality: {config.quality}\nInference Speed:  {config.inference_speed: <20} Stability: {config.stability}\nMin GPU Memory:   {config.min_gpu_memory}"); self.method_info_text.configure(state="disabled")
        self.production_indicator.configure(text="✓ Production Ready" if config.production_ready else "✗ Experimental", text_color="#28A745" if config.production_ready else "#FFC107")
        if self.auto_output_checkbox.get(): self.update_auto_output_path()
        self.update_time_estimate()

    def update_time_estimate(self):
        if self.current_model_info and "error" not in self.current_model_info and (params := self.current_model_info.get('param_count')):
            method = next(m for m in QuantizationMethod if m.value == self.quantization_method.get())
            self.time_estimate_label.configure(text=f"Estimated Time: {ModelQuantizer.estimate_quantization_time(params / 1e9, method)}")
        else: self.time_estimate_label.configure(text="Estimated Time: N/A")

    def on_auto_output_change(self):
        if self.auto_output_checkbox.get(): self.output_entry.configure(state="disabled"); self.update_auto_output_path()
        else: self.output_entry.configure(state="normal")

    def update_auto_output_path(self):
        if model_path := self.model_path.get():
            name = Path(model_path).name; short_name = "".join(filter(str.isalnum, self.quantization_method.get().split('(')[0].strip())).lower()
            self.output_path.set(str(Path(model_path).parent / f"{name}-{short_name}-processed"))

    def auto_recommend_quantization(self):
        if not self.current_model_info or "error" in self.current_model_info: messagebox.showwarning("Warning", "Select a valid model first."); return
        try:
            available_mem = (self.system_info['gpu_info'][0]['memory_total'] if torch.cuda.is_available() else self.system_info['memory_available']) / (1024**3)
            method = SystemProfiler.recommend_quantization_method(self.current_model_info.get('disk_size_gb', 0), available_mem)
            self.quantization_method.set(method.value); messagebox.showinfo("Recommendation", f"Recommended: {method.value}")
        except Exception as e: messagebox.showerror("Error", f"Could not generate recommendation: {e}")

    def start_quantization(self):
        if self.is_processing: messagebox.showwarning("Warning", "Process already running."); return
        if not (model_path := self.model_path.get()) or not os.path.isdir(model_path): messagebox.showerror("Error", "Invalid model path."); return
        if not (output_path := self.output_path.get()): messagebox.showerror("Error", "Invalid output path."); return
        if os.path.exists(output_path) and os.listdir(output_path) and not messagebox.askyesno("Warning", "Output directory not empty. Overwrite?"): return
        self.is_processing = True; self.toggle_ui_state(False)
        self.quantization_thread = threading.Thread(target=self._quantize_thread_target, args=(model_path, output_path), daemon=True); self.quantization_thread.start()

    def _quantize_thread_target(self, model_path, output_path):
        method = next(m for m in QuantizationMethod if m.value == self.quantization_method.get())
        success, message = ModelQuantizer.quantize_model(model_path, output_path, method, self.update_progress, self.enable_optimizations.get(), self.create_backup.get(), self.verify_output.get())
        self.root.after(0, self.on_quantization_finished, success, message)

    def on_quantization_finished(self, success, message):
        self.is_processing = False; self.toggle_ui_state(True)
        if success: messagebox.showinfo("Success", message); self.status_label.configure(text=f"Success: {message}")
        else: messagebox.showerror("Error", message); self.status_label.configure(text=f"Error: {message}")

    def update_progress(self, message: str, value: float):
        self.root.after(0, lambda: (self.progress_label.configure(text=message), self.progress.set(value), self.progress_percentage.configure(text=f"{int(value * 100)}%")))

    def cancel_quantization(self): messagebox.showinfo("Cancel", "Cancellation not implemented. Please close the application to stop.")

    def toggle_ui_state(self, enabled: bool):
        state = "normal" if enabled else "disabled"
        for widget in [self.quantize_btn, self.model_entry, self.quant_menu, self.recommend_btn, self.auto_output_checkbox, self.optimization_checkbox, self.backup_checkbox, self.verify_checkbox]: widget.configure(state=state)
        self.cancel_btn.configure(state="disabled" if enabled else "normal")
        if not self.auto_output_checkbox.get() or not enabled: self.output_entry.configure(state=state)
        else: self.output_entry.configure(state="disabled")

    def run(self): self.root.mainloop()

if __name__ == "__main__":
    try:
        app = ModelQuantizerGUI()
        app.run()
    except Exception as e:
        logger.critical(f"A critical error occurred: {e}\n{traceback.format_exc()}")
        try:
            root = ctk.CTk(); root.withdraw()
            messagebox.showerror("Critical Error", f"Application failed.\nCheck 'quantizer.log' for details.\n\nError: {e}")
        finally: sys.exit(1)
