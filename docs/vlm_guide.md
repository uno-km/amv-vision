# VLM (Visual Language Model) Inference Guide

This guide explains how to perform multimodal visual question answering directly on mobile Android Termux using Termux-Vision.

## 1. Loading Lightweight Vision-Language Models
Termux-Vision supports quantized GGUF VLM weights:

```python
import termux_vision as vision

vlm = vision.VLMEngine(model="fastvlm-1.5b-q4")
response = vlm.ask(image="photo.jpg", prompt="Describe the objects in this room.")
print(response)
```
