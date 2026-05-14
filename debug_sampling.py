"""
Script para debugar o problema de geração repetitiva.

Vamos verificar:
1. Os logits que o modelo está gerando
2. Se há NaNs ou Infs
3. Se a função multinomial está funcionando
4. O comportamento passo-a-passo da geração
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

print(f"✅ Modelo carregado")
print(f"   Vocab size: {tokenizer.vocab_size}")
print(f"   Device: {device}")

# ===== Teste 1: Forward pass simples =====
print("\n" + "=" * 60)
print("TESTE 1: Forward Pass Simples")
print("=" * 60)

prompt = "To be or"
prompt_ids = tokenizer.encode(prompt)
print(f"Prompt: '{prompt}'")
print(f"Prompt IDs: {prompt_ids}")

# Converter para tensor
input_tensor = torch.tensor([prompt_ids], dtype=torch.long).to(device)
print(f"Input shape: {input_tensor.shape}")

# Forward pass
with torch.no_grad():
    logits = model(input_tensor)
    
print(f"Output logits shape: {logits.shape}")
print(f"Last token logits shape: {logits[0, -1, :].shape}")

last_logits = logits[0, -1, :]
print(f"\nÚltimos 10 valores de logits: {last_logits[-10:].tolist()}")
print(f"Min logit: {last_logits.min():.4f}")
print(f"Max logit: {last_logits.max():.4f}")
print(f"Mean logit: {last_logits.mean():.4f}")

# ===== Teste 2: Verificar NaN/Inf =====
print("\n" + "=" * 60)
print("TESTE 2: Verificar NaN/Inf nos Logits")
print("=" * 60)

has_nan = torch.isnan(logits).any()
has_inf = torch.isinf(logits).any()

print(f"Tem NaN: {has_nan}")
print(f"Tem Inf: {has_inf}")

if has_nan:
    print("⚠️  PROBLEMA: Tem NaN nos logits!")
if has_inf:
    print("⚠️  PROBLEMA: Tem Inf nos logits!")

# ===== Teste 3: Softmax com temperatura =====
print("\n" + "=" * 60)
print("TESTE 3: Softmax com Diferentes Temperaturas")
print("=" * 60)

for temp in [0.5, 1.0, 2.0]:
    with torch.no_grad():
        logits = model(input_tensor)
        next_token_logits = logits[0, -1, :]
        
        # Aplicar temperatura
        probs = F.softmax(next_token_logits / temp, dim=-1)
        
        # Top 5 tokens mais prováveis
        top_probs, top_indices = torch.topk(probs, k=5, dim=-1)
        
        print(f"\nTemperatura {temp}:")
        for i, (prob, idx) in enumerate(zip(top_probs, top_indices)):
            char = tokenizer.decode([idx.item()])
            print(f"  {i+1}. '{char}' (ID {idx.item():3d}): {prob:.4f}")
        
        # Verificar se tem NaN/Inf
        has_nan_probs = torch.isnan(probs).any()
        has_inf_probs = torch.isinf(probs).any()
        prob_sum = probs.sum()
        
        print(f"     NaN: {has_nan_probs}, Inf: {has_inf_probs}, Sum: {prob_sum:.4f}")

# ===== Teste 4: Sampling passo-a-passo =====
print("\n" + "=" * 60)
print("TESTE 4: Geração Passo-a-Passo (5 passos)")
print("=" * 60)

prompt = "my name is Sergio"
prompt_ids = tokenizer.encode(prompt)
print(f"Prompt inicial: '{prompt}'")
print(f"Prompt IDs: {prompt_ids}\n")

generated = torch.tensor([prompt_ids], dtype=torch.long).to(device)

with torch.no_grad():
    for step in range(5):
        print(f"Passo {step + 1}:")
        
        # Forward pass
        logits = model(generated[:, -128:])
        next_token_logits = logits[0, -1, :]
        
        # Temperatura = 1.0
        probs = F.softmax(next_token_logits / 1.0, dim=-1)
        
        # Amostrar
        next_token_id = torch.multinomial(probs, num_samples=1).item()
        next_char = tokenizer.decode([next_token_id])
        
        print(f"  Próximo token ID: {next_token_id}")
        print(f"  Próximo char: '{next_char}'")
        print(f"  Prob do char: {probs[next_token_id]:.6f}")
        
        # Adicionar à sequência
        generated = torch.cat([generated, torch.tensor([[next_token_id]], dtype=torch.long).to(device)], dim=1)
        
        # Mostrar sequência gerada até agora
        current_text = tokenizer.decode(generated[0].tolist())
        print(f"  Texto gerado: '{current_text}'")
        print()

# ===== Teste 5: Usar a função generate() original =====
print("\n" + "=" * 60)
print("TESTE 5: Usar Função generate() Original")
print("=" * 60)

prompt = "To be or"
prompt_ids = tokenizer.encode(prompt)

generated_ids = model.generate(
    prompt_ids,
    max_length=50,
    temperature=1.0,
    device=device
)

generated_text = tokenizer.decode(generated_ids)
print(f"Prompt: '{prompt}'")
print(f"Gerado: '{generated_text}'")
print(f"Comprimento: {len(generated_ids)} tokens")

# Verificar se está com padrão repetitivo
last_20 = generated_text[-20:]
print(f"Últimos 20 chars: '{last_20}'")

# Contar frequência de caracteres
from collections import Counter
char_freq = Counter(last_20)
print(f"Frequência dos últimos 20: {dict(char_freq)}")
