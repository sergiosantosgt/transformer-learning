"""
Teste com técnicas avançadas de amostragem para evitar colapso.

Vamos testar:
1. Top-k sampling
2. Top-p (nucleus) sampling
3. Diferentes temperaturas
4. Validação em dados reais
"""

import torch
import torch.nn.functional as F
from model import GPTMini, CharacterTokenizer
import numpy as np

# ===== Carregar modelo e tokenizador =====
print("Carregando modelo...")
device = 'cpu'

checkpoint = torch.load("model/gpt_mini_best.pt", map_location=device)
tokenizer = CharacterTokenizer.load("model/tokenizer.pkl")

model = GPTMini(
    vocab_size=checkpoint['config']['vocab_size'],
    max_seq_len=checkpoint['config']['max_seq_len'],
    d_model=checkpoint['config']['d_model'],
    num_heads=checkpoint['config']['num_heads'],
    num_layers=checkpoint['config']['num_layers'],
    d_ff=checkpoint['config']['d_ff'],
).to(device)

model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

print(f"✅ Modelo carregado\n")

# ===== Teste A: Temperatura Diferente =====
print("=" * 70)
print("TESTE A: Diferentes Temperaturas")
print("=" * 70)

prompts = ["To be or", "The king", "O Romeo"]

for temp in [0.3, 0.7, 1.0, 1.5, 2.0]:
    print(f"\n🌡️  Temperatura {temp}:")
    for prompt in prompts:
        prompt_ids = tokenizer.encode(prompt)
        generated_ids = model.generate(
            prompt_ids,
            max_length=80,
            temperature=temp,
            device=device
        )
        generated_text = tokenizer.decode(generated_ids)
        print(f"  '{prompt}' → '{generated_text[:60]}...'")

# ===== Teste B: Top-K Sampling =====
print("\n" + "=" * 70)
print("TESTE B: Top-K Sampling (com K=10, temp=1.0)")
print("=" * 70)

for prompt in prompts:
    print(f"\n📝 Prompt: '{prompt}'")
    prompt_ids = tokenizer.encode(prompt)
    
    # Sem top-k (modo normal)
    generated_ids = model.generate(
        prompt_ids,
        max_length=80,
        temperature=1.0,
        top_k=None,
        device=device
    )
    text_normal = tokenizer.decode(generated_ids)
    
    # Com top-k=10
    generated_ids_topk = model.generate(
        prompt_ids,
        max_length=80,
        temperature=1.0,
        top_k=10,
        device=device
    )
    text_topk = tokenizer.decode(generated_ids_topk)
    
    print(f"  Normal: '{text_normal[:60]}...'")
    print(f"  Top-K:  '{text_topk[:60]}...'")

# ===== Teste C: Top-P (Nucleus) Sampling =====
print("\n" + "=" * 70)
print("TESTE C: Top-P (Nucleus) Sampling (p=0.9, temp=1.0)")
print("=" * 70)

for prompt in prompts:
    print(f"\n📝 Prompt: '{prompt}'")
    prompt_ids = tokenizer.encode(prompt)
    
    # Sem top-p (modo normal)
    generated_ids = model.generate(
        prompt_ids,
        max_length=80,
        temperature=1.0,
        top_p=None,
        device=device
    )
    text_normal = tokenizer.decode(generated_ids)
    
    # Com top-p=0.9
    generated_ids_topp = model.generate(
        prompt_ids,
        max_length=80,
        temperature=1.0,
        top_p=0.9,
        device=device
    )
    text_topp = tokenizer.decode(generated_ids_topp)
    
    print(f"  Normal:  '{text_normal[:60]}...'")
    print(f"  Top-P:   '{text_topp[:60]}...'")

# ===== Teste D: Combinação Otimizada =====
print("\n" + "=" * 70)
print("TESTE D: Combinação Otimizada (temp=0.8, top-p=0.95, top-k=40)")
print("=" * 70)

print("\n🎯 Gerando com parâmetros otimizados:\n")

for prompt in prompts:
    print(f"Prompt: '{prompt}'")
    prompt_ids = tokenizer.encode(prompt)
    
    generated_ids = model.generate(
        prompt_ids,
        max_length=100,
        temperature=0.8,
        top_k=40,
        top_p=0.95,
        device=device
    )
    generated_text = tokenizer.decode(generated_ids)
    print(f"Resultado: '{generated_text}'")
    print()

# ===== Teste E: Analisar o padrão repetitivo =====
print("=" * 70)
print("TESTE E: Análise do Padrão Repetitivo")
print("=" * 70)

def analyze_repetition(text, window_size=5):
    """Detectar padrões repetitivos."""
    if len(text) < window_size:
        return None
    
    last_window = text[-window_size:]
    count = 1
    
    for i in range(len(text) - window_size - 1, -1, -1):
        if i >= 0 and text[i:i+window_size] == last_window:
            count += 1
        else:
            break
    
    return count, last_window

prompt = "my name is Sergio"
prompt_ids = tokenizer.encode(prompt)

print(f"\nPrompt: '{prompt}'")

for temp in [1.0, 0.8]:
    print(f"\nTemperatura {temp}:")
    
    generated_ids = model.generate(
        prompt_ids,
        max_length=100,
        temperature=temp,
        device=device
    )
    text = tokenizer.decode(generated_ids)
    
    # Contar caracteres únicos no final
    unique_chars = len(set(text[-30:]))
    print(f"  Texto: '{text}'")
    print(f"  Últimos 30 chars têm {unique_chars} char(es) único(s)")
    
    # Ver se há padrão
    if unique_chars == 1:
        char = text[-1]
        count = len(text) - len(text.rstrip(char))
        print(f"  ⚠️  Colapso detectado: {count} repetições de '{char}'")
