#!/usr/bin/env python3
"""
HuggingFace Model Quantizer
A professional GUI application for quantizing HuggingFace models to lower bit precision.
"""

import os
import sys
import threading
import traceback
from pathlib import Path
from typing import Optional, Dict, Any
import logging
from datetime import datetime

import customtkinter as ctk
from tkinter import filedialog, messagebox
import torch
from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer, 
    BitsAndBytesConfig,
    AutoConfig
)
from accelerate import init_empty_weights
import gc

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('quantizer.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class ModelQuantizer:
    """Core model quantization functionality."""
    
    QUANTIZATION_CONFIGS = {
        "4-bit (NF4)": {
            "load_in_4bit": True,
            "bnb_4bit_quant_type": "nf4",
            "bnb_4bit_use_double_quant": True,
            "bnb_4bit_compute_dtype": torch.bfloat16
        },
        "4-bit (FP4)": {
            "load_in_4bit": True,
            "bnb_4bit_quant_type": "fp4",
            "bnb_4bit_use_double_quant": True,
            "bnb_4bit_compute_dtype": torch.bfloat16
        },
        "8-bit": {
            "load_in_8bit": True,
            "device_map": "auto"
        }
    }
    
    @staticmethod
    def validate_model_path(model_path: str) -> tuple[bool, str]:
        """Validate if the given path contains a valid HuggingFace model."""
        try:
            path = Path(model_path)
            if not path.exists():
                return False, f"Path does not exist: {model_path}"
            
            if not path.is_dir():
                return False, f"Path is not a directory: {model_path}"
            
            # Check for essential model files
            config_file = path / 'config.json'
            if not config_file.exists():
                return False, "Missing config.json file"
            
            # Check for model weight files (more comprehensive)
            model_patterns = [
                'pytorch_model.bin',
                'model.safetensors', 
                'pytorch_model-*.bin',
                'model-*.safetensors',
                'pytorch_model.bin.index.json',
                'model.safetensors.index.json'
            ]
            
            model_files_found = []
            for pattern in model_patterns:
                found_files = list(path.glob(pattern))
                model_files_found.extend(found_files)
            
            if not model_files_found:
                # List available files for debugging
                available_files = [f.name for f in path.iterdir() if f.is_file()]
                return False, f"No model weight files found. Available files: {', '.join(available_files[:10])}"
            
            # Try to load config to ensure it's valid
            try:
                config = AutoConfig.from_pretrained(model_path)
                logger.info(f"Valid model found: {config.model_type}")
            except Exception as e:
                return False, f"Invalid config.json: {str(e)}"
            
            return True, "Valid HuggingFace model"
            
        except Exception as e:
            return False, f"Error validating model: {str(e)}"
    
    @staticmethod
    def get_model_info(model_path: str) -> Dict[str, Any]:
        """Extract model information."""
        try:
            config = AutoConfig.from_pretrained(model_path)
            
            # Calculate approximate model size
            param_count = getattr(config, 'n_parameters', None)
            if param_count is None:
                # Estimate based on hidden size and layers
                hidden_size = getattr(config, 'hidden_size', 0)
                num_layers = getattr(config, 'num_hidden_layers', 0)
                vocab_size = getattr(config, 'vocab_size', 0)
                
                if all([hidden_size, num_layers, vocab_size]):
                    # Rough estimation
                    param_count = (hidden_size * hidden_size * 12 * num_layers) + (vocab_size * hidden_size * 2)
            
            return {
                'model_type': getattr(config, 'model_type', 'unknown'),
                'hidden_size': getattr(config, 'hidden_size', 'unknown'),
                'num_layers': getattr(config, 'num_hidden_layers', 'unknown'),
                'vocab_size': getattr(config, 'vocab_size', 'unknown'),
                'param_count': param_count,
                'architectures': getattr(config, 'architectures', ['unknown'])
            }
        except Exception as e:
            logger.error(f"Error reading model info: {e}")
            return {}
    
    @staticmethod
    def quantize_model(model_path: str, output_path: str, quantization_type: str, 
                      progress_callback=None) -> bool:
        """Quantize the model and save to output path."""
        try:
            if progress_callback:
                progress_callback("Initializing quantization...")
            
            # Get quantization config
            quant_config = ModelQuantizer.QUANTIZATION_CONFIGS[quantization_type]
            
            if progress_callback:
                progress_callback("Loading model configuration...")
            
            # Create BitsAndBytesConfig
            if "load_in_4bit" in quant_config:
                bnb_config = BitsAndBytesConfig(**quant_config)
            else:
                bnb_config = BitsAndBytesConfig(**quant_config)
            
            if progress_callback:
                progress_callback("Loading tokenizer...")
            
            # Load tokenizer
            tokenizer = AutoTokenizer.from_pretrained(model_path)
            
            if progress_callback:
                progress_callback("Loading and quantizing model (this may take a while)...")
            
            # Load and quantize model
            model = AutoModelForCausalLM.from_pretrained(
                model_path,
                quantization_config=bnb_config,
                device_map="auto",
                torch_dtype=torch.bfloat16,
                trust_remote_code=True
            )
            
            if progress_callback:
                progress_callback("Saving quantized model...")
            
            # Create output directory
            os.makedirs(output_path, exist_ok=True)
            
            # Save model and tokenizer
            model.save_pretrained(output_path)
            tokenizer.save_pretrained(output_path)
            
            # Clean up memory
            del model
            del tokenizer
            gc.collect()
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
            
            if progress_callback:
                progress_callback("Quantization completed successfully!")
            
            logger.info(f"Model successfully quantized and saved to {output_path}")
            return True
            
        except Exception as e:
            error_msg = f"Quantization failed: {str(e)}"
            logger.error(error_msg)
            logger.error(traceback.format_exc())
            if progress_callback:
                progress_callback(f"Error: {error_msg}")
            return False

class ModelQuantizerGUI:
    """Professional GUI for the model quantizer."""
    
    def __init__(self):
        # Configure customtkinter
        ctk.set_appearance_mode("dark")
        ctk.set_default_color_theme("blue")
        
        # Initialize main window
        self.root = ctk.CTk()
        self.root.title("ModelQuants - AI Model Quantizer")
        self.root.geometry("800x900")
        self.root.minsize(700, 500)
        
        # Variables
        self.model_path = ctk.StringVar()
        self.output_path = ctk.StringVar()
        self.quantization_type = ctk.StringVar(value="4-bit (NF4)")
        self.is_processing = False
        
        self.setup_ui()
        self.center_window()
        
    def center_window(self):
        """Center the window on screen."""
        self.root.update_idletasks()
        x = (self.root.winfo_screenwidth() // 2) - (self.root.winfo_width() // 2)
        y = (self.root.winfo_screenheight() // 2) - (self.root.winfo_height() // 2)
        self.root.geometry(f"+{x}+{y}")
    
    def setup_ui(self):
        """Setup the user interface."""
        # Main container
        main_frame = ctk.CTkFrame(self.root)
        main_frame.pack(fill="both", expand=True, padx=20, pady=20)
        
        # Title
        title_label = ctk.CTkLabel(
            main_frame, 
            text="ModelQuants - AI Model Quantizer",
            font=ctk.CTkFont(size=24, weight="bold")
        )
        title_label.pack(pady=(20, 30))
        
        # Model input section
        self.setup_model_input_section(main_frame)
        
        # Output section
        self.setup_output_section(main_frame)
        
        # Quantization options
        self.setup_quantization_section(main_frame)
        
        # Model info section
        self.setup_model_info_section(main_frame)
        
        # Progress and controls
        self.setup_progress_section(main_frame)
        
        # Status bar
        self.setup_status_bar()
    
    def setup_model_input_section(self, parent):
        """Setup model input section."""
        input_frame = ctk.CTkFrame(parent)
        input_frame.pack(fill="x", padx=20, pady=(0, 15))
        
        ctk.CTkLabel(
            input_frame, 
            text="Model Input Path:",
            font=ctk.CTkFont(size=14, weight="bold")
        ).pack(anchor="w", padx=20, pady=(15, 5))
        
        path_frame = ctk.CTkFrame(input_frame)
        path_frame.pack(fill="x", padx=20, pady=(0, 15))
        
        self.model_entry = ctk.CTkEntry(
            path_frame,
            textvariable=self.model_path,
            placeholder_text="Select HuggingFace model folder...",
            height=35
        )
        self.model_entry.pack(side="left", fill="x", expand=True, padx=(10, 5), pady=10)
        
        browse_btn = ctk.CTkButton(
            path_frame,
            text="Browse",
            command=self.browse_model_folder,
            width=100,
            height=35
        )
        browse_btn.pack(side="right", padx=(5, 10), pady=10)
        
        # Bind path change event
        self.model_path.trace_add("write", self.on_model_path_change)
    
    def setup_output_section(self, parent):
        """Setup output section."""
        output_frame = ctk.CTkFrame(parent)
        output_frame.pack(fill="x", padx=20, pady=(0, 15))
        
        ctk.CTkLabel(
            output_frame, 
            text="Output Path:",
            font=ctk.CTkFont(size=14, weight="bold")
        ).pack(anchor="w", padx=20, pady=(15, 5))
        
        output_path_frame = ctk.CTkFrame(output_frame)
        output_path_frame.pack(fill="x", padx=20, pady=(0, 15))
        
        self.output_entry = ctk.CTkEntry(
            output_path_frame,
            textvariable=self.output_path,
            placeholder_text="Select output folder for quantized model...",
            height=35
        )
        self.output_entry.pack(side="left", fill="x", expand=True, padx=(10, 5), pady=10)
        
        output_browse_btn = ctk.CTkButton(
            output_path_frame,
            text="Browse",
            command=self.browse_output_folder,
            width=100,
            height=35
        )
        output_browse_btn.pack(side="right", padx=(5, 10), pady=10)
    
    def setup_quantization_section(self, parent):
        """Setup quantization options section."""
        quant_frame = ctk.CTkFrame(parent)
        quant_frame.pack(fill="x", padx=20, pady=(0, 15))
        
        ctk.CTkLabel(
            quant_frame, 
            text="Quantization Type:",
            font=ctk.CTkFont(size=14, weight="bold")
        ).pack(anchor="w", padx=20, pady=(15, 5))
        
        self.quant_menu = ctk.CTkOptionMenu(
            quant_frame,
            values=list(ModelQuantizer.QUANTIZATION_CONFIGS.keys()),
            variable=self.quantization_type,
            height=35
        )
        self.quant_menu.pack(anchor="w", padx=20, pady=(0, 15))
    
    def setup_model_info_section(self, parent):
        """Setup model information section."""
        info_frame = ctk.CTkFrame(parent)
        info_frame.pack(fill="both", expand=True, padx=20, pady=(0, 15))
        
        ctk.CTkLabel(
            info_frame, 
            text="Model Information:",
            font=ctk.CTkFont(size=14, weight="bold")
        ).pack(anchor="w", padx=20, pady=(15, 5))
        
        self.info_text = ctk.CTkTextbox(
            info_frame,
            height=120,
            font=ctk.CTkFont(family="Consolas", size=11)
        )
        self.info_text.pack(fill="both", expand=True, padx=20, pady=(0, 15))
        self.info_text.insert("1.0", "No model selected...")
        self.info_text.configure(state="disabled")
    
    def setup_progress_section(self, parent):
        """Setup progress and control section."""
        control_frame = ctk.CTkFrame(parent)
        control_frame.pack(fill="x", padx=20, pady=(0, 15))
        
        # Progress bar
        self.progress = ctk.CTkProgressBar(control_frame)
        self.progress.pack(fill="x", padx=20, pady=(15, 10))
        self.progress.set(0)
        
        # Progress label
        self.progress_label = ctk.CTkLabel(
            control_frame,
            text="Ready to quantize model...",
            font=ctk.CTkFont(size=12)
        )
        self.progress_label.pack(padx=20, pady=(0, 10))
        
        # Control buttons
        btn_frame = ctk.CTkFrame(control_frame, fg_color="transparent")
        btn_frame.pack(pady=(0, 15))
        
        self.quantize_btn = ctk.CTkButton(
            btn_frame,
            text="Start Quantization",
            command=self.start_quantization,
            height=40,
            width=150,
            font=ctk.CTkFont(size=14, weight="bold")
        )
        self.quantize_btn.pack(side="left", padx=(20, 10))
        
        self.cancel_btn = ctk.CTkButton(
            btn_frame,
            text="Cancel",
            command=self.cancel_quantization,
            height=40,
            width=100,
            fg_color="gray",
            hover_color="dark gray",
            state="disabled"
        )
        self.cancel_btn.pack(side="left", padx=(10, 20))
        
        # Debug button for troubleshooting
        self.debug_btn = ctk.CTkButton(
            btn_frame,
            text="Debug Path",
            command=self.debug_model_path,
            height=40,
            width=100,
            fg_color="orange",
            hover_color="dark orange"
        )
        self.debug_btn.pack(side="left", padx=(10, 20))
    
    def setup_status_bar(self):
        """Setup status bar."""
        self.status_label = ctk.CTkLabel(
            self.root,
            text="Ready",
            font=ctk.CTkFont(size=10),
            anchor="w"
        )
        self.status_label.pack(side="bottom", fill="x", padx=20, pady=(0, 10))
    
    def browse_model_folder(self):
        """Browse for model folder."""
        if self.is_processing:
            return
            
        folder = filedialog.askdirectory(
            title="Select HuggingFace Model Folder",
            initialdir=os.path.expanduser("~")
        )
        if folder:
            self.model_path.set(folder)
    
    def browse_output_folder(self):
        """Browse for output folder."""
        if self.is_processing:
            return
            
        folder = filedialog.askdirectory(
            title="Select Output Folder",
            initialdir=os.path.expanduser("~")
        )
        if folder:
            self.output_path.set(folder)
    
    def on_model_path_change(self, *args):
        """Handle model path change."""
        path = self.model_path.get()
        
        if not path:
            self.update_model_info("No model selected...")
            return
        
        # Validate model path with detailed error message
        is_valid, message = ModelQuantizer.validate_model_path(path)
        if not is_valid:
            self.update_model_info(f"Invalid model path: {message}")
            return
        
        # Get model info
        info = ModelQuantizer.get_model_info(path)
        if info:
            info_text = self.format_model_info(info)
            self.update_model_info(info_text)
            
            # Auto-suggest output path
            if not self.output_path.get():
                model_name = Path(path).name
                suggested_output = str(Path(path).parent / f"{model_name}_quantized")
                self.output_path.set(suggested_output)
        else:
            self.update_model_info("Could not read model information.")
    
    def format_model_info(self, info: Dict[str, Any]) -> str:
        """Format model information for display."""
        lines = []
        lines.append(f"Model Type: {info.get('model_type', 'Unknown')}")
        lines.append(f"Architecture: {', '.join(info.get('architectures', ['Unknown']))}")
        lines.append(f"Hidden Size: {info.get('hidden_size', 'Unknown')}")
        lines.append(f"Number of Layers: {info.get('num_layers', 'Unknown')}")
        lines.append(f"Vocabulary Size: {info.get('vocab_size', 'Unknown'):,}")
        
        if info.get('param_count'):
            param_count = info['param_count']
            if param_count > 1e9:
                param_str = f"{param_count / 1e9:.1f}B"
            elif param_count > 1e6:
                param_str = f"{param_count / 1e6:.1f}M"
            else:
                param_str = f"{param_count:,}"
            lines.append(f"Estimated Parameters: {param_str}")
        
        return "\n".join(lines)
    
    def update_model_info(self, text: str):
        """Update model information display."""
        self.info_text.configure(state="normal")
        self.info_text.delete("1.0", "end")
        self.info_text.insert("1.0", text)
        self.info_text.configure(state="disabled")
    
    def update_progress(self, message: str, progress: float = None):
        """Update progress display."""
        self.progress_label.configure(text=message)
        if progress is not None:
            self.progress.set(progress)
        self.root.update_idletasks()
    
    def start_quantization(self):
        """Start the quantization process."""
        # Validate inputs
        if not self.model_path.get():
            messagebox.showerror("Error", "Please select a model folder.")
            return
        
        if not self.output_path.get():
            messagebox.showerror("Error", "Please select an output folder.")
            return
        
        # Validate model path with detailed error message
        is_valid, error_message = ModelQuantizer.validate_model_path(self.model_path.get())
        if not is_valid:
            messagebox.showerror("Error", f"Invalid model folder: {error_message}")
            return
        
        # Check if output folder exists and warn user
        if os.path.exists(self.output_path.get()) and os.listdir(self.output_path.get()):
            result = messagebox.askyesno(
                "Output Folder Not Empty", 
                "The output folder already contains files. Do you want to continue?"
            )
            if not result:
                return
        
        # Start quantization in separate thread
        self.is_processing = True
        self.quantize_btn.configure(state="disabled")
        self.cancel_btn.configure(state="normal")
        self.progress.set(0)
        
        self.quantization_thread = threading.Thread(target=self.quantization_worker)
        self.quantization_thread.daemon = True
        self.quantization_thread.start()
    
    def quantization_worker(self):
        """Worker thread for quantization."""
        try:
            def progress_callback(message):
                self.root.after(0, lambda: self.update_progress(message))
            
            success = ModelQuantizer.quantize_model(
                self.model_path.get(),
                self.output_path.get(),
                self.quantization_type.get(),
                progress_callback
            )
            
            # Update UI on completion
            self.root.after(0, lambda: self.quantization_complete(success))
            
        except Exception as e:
            error_msg = f"Quantization failed: {str(e)}"
            logger.error(error_msg)
            self.root.after(0, lambda: self.quantization_complete(False, error_msg))
    
    def quantization_complete(self, success: bool, error_msg: str = ""):
        """Handle quantization completion."""
        self.is_processing = False
        self.quantize_btn.configure(state="normal")
        self.cancel_btn.configure(state="disabled")
        
        if success:
            self.progress.set(1.0)
            self.update_progress("Quantization completed successfully!")
            messagebox.showinfo(
                "Success", 
                f"Model quantized successfully!\nOutput saved to: {self.output_path.get()}"
            )
        else:
            self.progress.set(0)
            self.update_progress("Quantization failed.")
            messagebox.showerror("Error", error_msg or "Quantization failed. Check logs for details.")
    
    def cancel_quantization(self):
        """Cancel the quantization process."""
        # Note: This is a simplified cancel - in practice, you'd need more sophisticated
        # thread management to cleanly cancel the quantization process
        self.is_processing = False
        self.quantize_btn.configure(state="normal")
        self.cancel_btn.configure(state="disabled")
        self.progress.set(0)
        self.update_progress("Quantization cancelled.")
    
    def debug_model_path(self):
        """Debug the current model path to help diagnose issues."""
        path = self.model_path.get()
        if not path:
            messagebox.showinfo("Debug", "No model path selected.")
            return
        
        debug_info = []
        try:
            path_obj = Path(path)
            debug_info.append(f"Path: {path}")
            debug_info.append(f"Exists: {path_obj.exists()}")
            debug_info.append(f"Is Directory: {path_obj.is_dir()}")
            
            if path_obj.exists():
                debug_info.append(f"Is Symlink: {path_obj.is_symlink()}")
                if path_obj.is_symlink():
                    try:
                        debug_info.append(f"Symlink Target: {path_obj.resolve()}")
                    except:
                        debug_info.append("Symlink Target: Could not resolve")
                
                if path_obj.is_dir():
                    files = [f.name for f in path_obj.iterdir() if f.is_file()]
                    debug_info.append(f"Files in directory ({len(files)}): {', '.join(files[:15])}")
                    if len(files) > 15:
                        debug_info.append("... (and more)")
                    
                    # Check for specific files
                    config_exists = (path_obj / 'config.json').exists()
                    debug_info.append(f"config.json exists: {config_exists}")
                    
                    if config_exists:
                        try:
                            with open(path_obj / 'config.json', 'r') as f:
                                import json
                                config_data = json.load(f)
                                debug_info.append(f"Model type: {config_data.get('model_type', 'unknown')}")
                                debug_info.append(f"Architectures: {config_data.get('architectures', 'unknown')}")
                        except Exception as e:
                            debug_info.append(f"Error reading config.json: {str(e)}")
            
        except Exception as e:
            debug_info.append(f"Error during debug: {str(e)}")
        
        # Show debug info in a dialog
        debug_text = "\n".join(debug_info)
        
        # Create a simple dialog to show debug info
        debug_window = ctk.CTkToplevel(self.root)
        debug_window.title("Model Path Debug Information")
        debug_window.geometry("600x400")
        debug_window.transient(self.root)
        debug_window.grab_set()
        
        debug_textbox = ctk.CTkTextbox(debug_window, font=ctk.CTkFont(family="Consolas", size=11))
        debug_textbox.pack(fill="both", expand=True, padx=20, pady=20)
        debug_textbox.insert("1.0", debug_text)
        debug_textbox.configure(state="disabled")
        
        close_btn = ctk.CTkButton(debug_window, text="Close", command=debug_window.destroy)
        close_btn.pack(pady=(0, 20))
    
    def run(self):
        """Run the application."""
        self.root.mainloop()

def main():
    """Main entry point."""
    try:
        # Check for required dependencies
        required_packages = ['torch', 'transformers', 'accelerate', 'bitsandbytes', 'customtkinter']
        missing_packages = []
        
        for package in required_packages:
            try:
                __import__(package)
            except ImportError:
                missing_packages.append(package)
        
        if missing_packages:
            print(f"Missing required packages: {', '.join(missing_packages)}")
            print("Please install them using: pip install " + " ".join(missing_packages))
            sys.exit(1)
        
        # Initialize and run GUI
        app = ModelQuantizerGUI()
        app.run()
        
    except Exception as e:
        logger.error(f"Application failed to start: {e}")
        logger.error(traceback.format_exc())
        messagebox.showerror("Error", f"Failed to start application: {e}")

if __name__ == "__main__":
    main()