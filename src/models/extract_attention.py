import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import Optional

def load_model_and_tokenizer(model_name: str, quantization: str = "nf4", device: str = "cuda"):
    """
    Loads a model and tokenizer with the specified quantization configuration.
    Sets output_attentions=True and attn_implementation="eager" to correctly return attention weights.
    """
    import os
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Quantization setup for VRAM reduction
    model_kwargs = {
        "output_attentions": True,
        "attn_implementation": "eager" # Required for the model to correctly return weights (FA2 optimizations unsupported)
    }
    
    if quantization == "nf4":
        from transformers import BitsAndBytesConfig
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )
        model_kwargs["quantization_config"] = quantization_config
        model_kwargs["device_map"] = device
    elif quantization == "bf16":
        model_kwargs["torch_dtype"] = torch.bfloat16
        model_kwargs["device_map"] = device
    else:
        model_kwargs["device_map"] = device
        
    model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)
    return model, tokenizer

def run_inference_and_extract_attention(model, tokenizer, prompt: str, device: str = "cuda"):
    """
    Processes the prompt, performs a forward pass, and returns the raw attention tensors.
    """
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    
    with torch.no_grad():
        outputs = model(**inputs, output_attentions=True)
        
    # outputs.attentions is a tuple of length (num_layers), where each element is:
    # a tensor of shape (batch_size, num_heads, sequence_length, sequence_length)
    return outputs.attentions, inputs['input_ids']

