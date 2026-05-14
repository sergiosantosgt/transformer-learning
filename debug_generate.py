"""
debug_generate.py - Script para debugar a geração de texto

Este script testa o modelo e mostra:
1. Verificação do checkpoint
2. Teste de forward pass
3. Distribuição de probabilidades
4. Geração passo-a-passo
"""

import torch
import pickle
import os
from model import GPTMini, CharacterTokenizer

# ===== Setup =====
device = 'mps' if torch.backends.mps.is_available() else 'cpu'
print(f"Device: {device}")

checkpoint_path = "model/gpt_mini_best.pt"
tokenizer_path = "model/tokenizer.pkl"

# ===== Carregar tokenizador =====
print("\n1. Carregando tokenizador...")
tokenizer = CharacterTokenizer.load(tokenizer_path)
print(f"   Vocab size: {tokenizer.vocab_size}")
print(f"   Chars: {tokenizer.chars[:20]}...")

# ===== Carregar modelo =====
print("\n2. Carregando modelo...")
checkpoint = torch.load(checkpoint_path, map_location=device)

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

print(f"   Parâmetros: {model.get_num_parameters():,}")
print(f"   Tamanho: {model.get_model_size_mb():.2f} MB")
print(f"   Epoch: {checkpoint['epoch']}")
print(f"   Val loss: {checkpoint['val_loss']:.6f}")

# ===== Teste de forward pass =====
print("\n3. Testando forward pass...")
prompt = "To be or not to"
prompt_ids = tokenizer.encode(prompt)
print(f"   Prompt: '{prompt}'")
print(f"   IDs: {prompt_ids}")

prompt_tensor = torch.tensor([prompt_ids], dtype=torch.long).to(device)
with torch.no_grad():
    logits = model(prompt_tensor)
    print(f"   Logits shape: {logits.shape}")
    
    # Analisar o último logit (predição para próximo token)
    last_logits = logits[0, -1, :]
    print(f"   Last token logits shape: {last_logits.shape}")
    
    # Top 10 probabilidades
    probs = torch.softmax(last_logits, dim=-1)
    top_probs, top_indices = torch.topk(probs, k=10)
    
    print("\n   Top 10 próximos tokens mais prováveis:")
    for i, (prob, idx) in enumerate(zip(top_probs, top_indices)):
        char = tokenizer.id_to_char.get(idx.item(), '?')
        print(f"      {i+1}. '{char}' (ID {idx.item():3d}): {prob.item():.4f}")

# ===== Geração manual passo-a-passo =====
print("\n4. Gerando passo-a-passo (primeiros 20 tokens)...")
generated = prompt_ids.copy()
print(f"   Start: {tokenizer.decode(generated)}")

for step in range(20):
    # Converter para tensor
    input_tensor = torch.tensor([generated[-128:]], dtype=torch.long).to(device)
    
    with torch.no_grad():
        logits = model(input_tensor)
        next_logits = logits[0, -1, :]
        
        # Temperatura
        temperature = 1.0
        probs = torch.softmax(next_logits / temperature, dim=-1)
        
        # Amostrar
        next_token = torch.multinomial(probs, num_samples=1).item()
        generated.append(next_token)
        
        # Top probabilidade
        top_prob = probs[next_token].item()
        char = tokenizer.id_to_char.get(next_token, '?')
        
        print(f"   Step {step+1:2d}: '{char}' (prob: {top_prob:.4f})")

print(f"\n   Final: {tokenizer.decode(generated)}")

# ===== Teste com geração do modelo =====
print("\n5. Teste com função generate() do modelo...")
with torch.no_grad():
    generated_ids = model.generate(
        prompt_ids,
        max_length=50,
        temperature=1.0,
        device=device
    )

generated_text = tokenizer.decode(generated_ids)
print(f"   Prompt: '{prompt}'")
print(f"   Gerado: '{generated_text}'")
print(f"   Comprimento: {len(generated_ids)} tokens")

print("\n✅ Debug completo!")
