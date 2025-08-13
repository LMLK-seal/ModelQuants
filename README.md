# 🚀 ModelQuants

[![Python Version](https://img.shields.io/badge/python-3.8%2B-blue.svg)](https://python.org)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Platform](https://img.shields.io/badge/platform-Windows%20%7C%20Linux%20%7C%20macOS-lightgrey.svg)](https://github.com)
[![HuggingFace](https://img.shields.io/badge/🤗-HuggingFace-yellow.svg)](https://huggingface.co)

**Professional Model Quantization Converter for HuggingFace Transformers**

ModelQuants is a state-of-the-art GUI application designed for AI researchers, engineers, and enthusiasts who need to efficiently quantize large language models. Convert your BF16/FP16 models to optimized 4-bit or 8-bit formats with a single click, dramatically reducing memory usage while maintaining model performance.

![ModelQuants Screenshot](https://github.com/LMLK-seal/ModelQuants/blob/main/screenshot.png?raw=true)

---

## ✨ Features

### 🎯 **Core Functionality**
- **🔧 Advanced Quantization**: Support for 4-bit (NF4/FP4) and 8-bit quantization using BitsAndBytesConfig
- **📊 Real-time Progress**: Live progress tracking with detailed status updates
- **🛡️ Model Validation**: Comprehensive model structure validation before processing
- **💾 Memory Optimization**: Automatic memory cleanup and CUDA cache management
- **🔍 Debug Tools**: Built-in diagnostic tools for troubleshooting model paths

### 🖥️ **Professional Interface**
- **🎨 Modern Dark Theme**: Sleek customtkinter-based GUI with professional aesthetics
- **📁 Smart Path Management**: Auto-suggestion of output paths and intelligent folder selection
- **📈 Model Information Display**: Automatic detection and display of model architecture details
- **⚡ Threaded Processing**: Non-blocking UI with background quantization processing
- **🚨 Error Handling**: Robust error management with user-friendly notifications

### 🔧 **Technical Excellence**
- **📝 Comprehensive Logging**: Detailed logging to both file and console for debugging
- **🔒 Thread Safety**: Safe multi-threaded operations with proper synchronization
- **💡 Intelligent Validation**: Deep model structure analysis and file integrity checks
- **🎯 Precision Control**: Fine-tuned quantization parameters for optimal results

---

## 🚀 Quick Start

### Prerequisites

- **Python 3.8+** 🐍
- **CUDA-compatible GPU** (recommended) ⚡
- **8GB+ System RAM** (16GB+ recommended for large models) 💾

### 📦 Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/LMLK-seal/ModelQuants.git
   cd ModelQuants
   ```

2. install manually:
   ```bash
   pip install torch transformers accelerate bitsandbytes customtkinter
   ```

3. **Run ModelQuants:**
   ```bash
   python ModelQuants.py
   ```

---

## 📖 Usage Guide

### 🎯 **Basic Workflow**

1. **📂 Select Model**: Choose your HuggingFace model folder
2. **📍 Set Output**: Specify where to save the quantized model
3. **⚙️ Choose Quantization**: Select your preferred quantization type
4. **🚀 Start Process**: Click "Start Quantization" and monitor progress

### 🔧 **Quantization Options**

| Type | Description | Use Case | Memory Reduction |
|------|-------------|----------|------------------|
| **4-bit (NF4)** | 🥇 Normalized Float 4-bit | **Recommended** - Best quality/size ratio | ~75% |
| **4-bit (FP4)** | 🥈 Float 4-bit | Alternative 4-bit option | ~75% |
| **8-bit** | 🥉 Integer 8-bit | Conservative quantization | ~50% |

### 📊 **Model Information Display**

ModelQuants automatically detects and displays:
- 🏗️ **Model Architecture**: Type and structure information
- 🔢 **Parameter Count**: Total model parameters (B/M format)
- 📏 **Hidden Size**: Model dimension specifications
- 🧠 **Layer Count**: Number of transformer layers
- 📚 **Vocabulary Size**: Tokenizer vocabulary information

---

## 🛠️ Advanced Features

### 🔍 **Debug Tools**

Use the **Debug Path** button to diagnose model loading issues:
- 📁 Directory structure analysis
- 🔗 Symlink resolution
- 📄 File listing and validation
- ⚙️ Configuration file inspection

### 📝 **Logging System**

ModelQuants maintains comprehensive logs:
- 📊 **Console Output**: Real-time processing information
- 📁 **File Logging**: Persistent logs saved to `quantizer.log`
- 🚨 **Error Tracking**: Detailed error traces for debugging

### 🎛️ **Configuration**

Advanced users can modify quantization parameters in the source code:
```python
QUANTIZATION_CONFIGS = {
    "4-bit (NF4)": {
        "load_in_4bit": True,
        "bnb_4bit_quant_type": "nf4",
        "bnb_4bit_use_double_quant": True,
        "bnb_4bit_compute_dtype": torch.bfloat16
    }
}
```

---

## 📋 Requirements

### 🖥️ **System Requirements**

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| **OS** | Windows 10/Linux/macOS | Windows 11/Ubuntu 20.04+ |
| **RAM** | 8GB | 16GB+ |
| **GPU** | CUDA 11.0+ | RTX 3080/4080+ |
| **Storage** | 50GB free | 100GB+ SSD |

### 📦 **Python Dependencies**

```
torch>=2.0.0
transformers>=4.30.0
accelerate>=0.20.0
bitsandbytes>=0.39.0
customtkinter>=5.0.0
```

---

## 🔧 Troubleshooting

### ❓ **Common Issues**

**🚨 "Invalid model folder" error:**
- Use the Debug Path button to analyze the directory
- Ensure `config.json` and model weight files are present
- Check for symlink or permission issues

**💾 "Out of memory" error:**
- Close other GPU-intensive applications
- Try 8-bit quantization first for large models
- Ensure sufficient system RAM

**⚡ "CUDA not available" warning:**
- Install CUDA-compatible PyTorch version
- Verify GPU drivers are up to date
- CPU quantization is supported but slower

### 📞 **Getting Help**

1. 🔍 Check the debug output using the Debug Path button
2. 📝 Review the `quantizer.log` file for detailed error information
3. 🐛 Open an issue with system specs and error logs
4. 💬 Join our community discussions

---

## 🤝 Contributing

We welcome contributions! Here's how you can help:

### 🎯 **Ways to Contribute**
- 🐛 **Bug Reports**: Submit detailed issue reports
- 💡 **Feature Requests**: Suggest new functionality
- 🔧 **Code Contributions**: Submit pull requests
- 📚 **Documentation**: Improve guides and examples

### 📝 **Coding Standards**
- Follow PEP 8 style guidelines
- Include type hints for new functions
- Add comprehensive docstrings
- Write unit tests for new features

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

```
MIT License

Copyright (c) 2024 ModelQuants Contributors

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.
```

---

## 🌟 Acknowledgments

- 🤗 **HuggingFace Team** for the transformers ecosystem
- 🔧 **BitsAndBytesConfig** for quantization algorithms
- 🎨 **CustomTkinter** for the modern GUI framework
- 🚀 **PyTorch Team** for the underlying ML framework
- 👥 **Open Source Community** for continuous inspiration

---

## 📊 Project Stats

![GitHub stars](https://img.shields.io/github/stars/LMLK-seal/modelquants?style=social)
![GitHub forks](https://img.shields.io/github/forks/LMLK-seal/modelquants?style=social)
![GitHub issues](https://img.shields.io/github/issues/LMLK-seal/modelquants)
![GitHub pull requests](https://img.shields.io/github/issues-pr/LMLK-seal/modelquants)

---

<div align="center">

**⭐ Star this repository if ModelQuants helped you optimize your models! ⭐**

[🐛 Report Bug](https://github.com/LMLK-seal/modelquants/issues) • [💡 Request Feature](https://github.com/LMLK-seal/modelquants/issues) • [💬 Discussions](https://github.com/LMLK-seal/modelquants/discussions)

</div>
