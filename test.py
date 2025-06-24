from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel
import torch
import sys

# Model ve LoRA ağırlıkları yolları
base_model_id = "tiiuae/falcon-rw-1b"  # Kendi kullandığın base model
lora_weights_path = "./atllama-az-lora"               # Eğitilen LoRA klasörü

# 8-bit quantization konfigürasyonu
bnb_config = BitsAndBytesConfig(load_in_8bit=True)

# Tokenizer yükle
tokenizer = AutoTokenizer.from_pretrained(base_model_id)

# Base modeli quantized olarak yükle
base_model = AutoModelForCausalLM.from_pretrained(
    base_model_id,
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True
)

# LoRA ağırlıklarını yükle
model = PeftModel.from_pretrained(base_model, lora_weights_path)
model.eval()

# Eğer pad_token yoksa eos_token ile ata
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

def generate_response(prompt):
    # İnference için prompt — eğitimde kullandığın formata uygun!
    formatted_prompt = f"### Sual:\n{prompt}\n\n### Cavab:\n"

    inputs = tokenizer(formatted_prompt, return_tensors="pt").to(model.device)

    # Üretim (generation)
    outputs = model.generate(
        **inputs,
        max_new_tokens=100,
        do_sample=True,        # Sampling aktif, çıktıyı çeşitlendirir
        temperature=0.7,       # Sıcaklık, daha yaratıcı cevap için 0.7 (0-1 arası)
        top_p=0.9,             # nucleus sampling, top p %90
        pad_token_id=tokenizer.eos_token_id,
    )

    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# Komut satırından prompt al
if len(sys.argv) < 2:
    print("Lütfen bir soru girin. Örnek: python test.py 'Sorunuz buraya'")
    sys.exit(1)

prompt = " ".join(sys.argv[1:])
response = generate_response(prompt)
print(response)