import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


def load_model_and_tokenizer(model_name: str, quantization: str = "nf4", device: str = "cuda"):
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    model_kwargs = {
        "output_attentions": True,
        "attn_implementation": "eager",
    }

    if quantization == "nf4":
        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )
        model_kwargs["device_map"] = device
    elif quantization == "bf16":
        model_kwargs["torch_dtype"] = torch.bfloat16
        model_kwargs["device_map"] = device
    else:
        model_kwargs["device_map"] = device

    model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)
    return model, tokenizer


def run_inference_and_extract_attention(model, tokenizer, prompt: str, device: str = "cuda"):
    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = model(**inputs, output_attentions=True)

    return outputs.attentions, inputs["input_ids"]
