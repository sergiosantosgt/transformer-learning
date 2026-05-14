"""
Teste de validação: Avaliar performance real do modelo em dados conhecidos.

Se o modelo tem val_loss baixo (0.021) mas não gera bem, há desconexão entre:
- Loss de treinamento (predição de próximo token em dados conhecidos)
- Qualidade de geração (sequência longa de predições em sequência)

Vamos:
1. Pegar sequências reais do dataset de validação
2. Verificar as predições do modelo
3. Ver se as probabilidades fazem sentido
"""

import torch
import torch.nn.functional as F
from model import GPTMini, CharacterTokenizer, create_data_loaders
import numpy as np

print("Carregando modelo e dados...")
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

# Carregar dados
train_loader, val_loader, _ = create_data_loaders(
    "data/shakespeare.txt",
    seq_len=128,
    batch_size=32,
    train_split=0.95,
    num_workers=0
)

print("✅ Carregado\n")

# ===== Teste 1: Validar em dados reais =====
print("=" * 70)
print("TESTE 1: Predição em Dados Reais (Val Set)")
print("=" * 70)

model.eval()
total_loss = 0
total_batches = 0
correct_tokens = 0
total_tokens = 0

with torch.no_grad():
    for batch_idx, (input_ids, target_ids) in enumerate(val_loader):
        if batch_idx >= 5:  # Apenas 5 primeiros batches
            break
        
        input_ids = input_ids.to(device)
        target_ids = target_ids.to(device)
        
        logits = model(input_ids)
        
        # Loss
        loss = F.cross_entropy(
            logits.view(-1, model.vocab_size),
            target_ids.view(-1)
        )
        
        total_loss += loss.item()
        total_batches += 1
        
        # Acurácia: quantos tokens foram preditos corretamente?
        predictions = logits.argmax(dim=-1)
        correct = (predictions == target_ids).sum().item()
        correct_tokens += correct
        total_tokens += target_ids.numel()
        
        if batch_idx < 2:  # Mostrar apenas primeiros 2 batches
            print(f"\nBatch {batch_idx + 1}:")
            print(f"  Loss: {loss.item():.6f}")
            print(f"  Acurácia: {correct / target_ids.numel() * 100:.1f}%")
            
            # Mostrar uma sequência completa
            seq_idx = 0
            input_seq = input_ids[seq_idx].cpu().tolist()
            target_seq = target_ids[seq_idx].cpu().tolist()
            predicted_seq = logits[seq_idx].argmax(dim=-1).cpu().tolist()
            
            input_text = tokenizer.decode(input_seq[:32])
            target_text = tokenizer.decode(target_seq[:32])
            predicted_text = tokenizer.decode(predicted_seq[:32])
            
            print(f"  Input:     '{input_text}'")
            print(f"  Esperado:  '{target_text}'")
            print(f"  Predito:   '{predicted_text}'")

print(f"\nMédia Loss (primeiros 5 batches): {total_loss / total_batches:.6f}")
print(f"Acurácia geral: {correct_tokens / total_tokens * 100:.1f}%")

# ===== Teste 2: Greedy Decoding em Sequências Reais =====
print("\n" + "=" * 70)
print("TESTE 2: Greedy Decoding vs Ground Truth")
print("=" * 70)

val_iter = iter(val_loader)
input_ids, target_ids = next(val_iter)
input_ids = input_ids.to(device)
target_ids = target_ids.to(device)

# Pegar primeira sequência
seq_idx = 0
input_seq = input_ids[seq_idx].cpu().tolist()
target_seq = target_ids[seq_idx].cpu().tolist()

print(f"\nSequência real do dataset:")
input_text = tokenizer.decode(input_seq)
target_text = tokenizer.decode(target_seq)

print(f"Input (primeiros 64 chars):")
print(f"  '{input_text[:64]}'")
print(f"Target (primeiros 64 chars):")
print(f"  '{target_text[:64]}'")

# Fazer greedy decoding: começar do input e prever cada token
print(f"\nGreedy Decoding (começando do input):")
generated = input_ids[seq_idx:seq_idx+1].clone()

with torch.no_grad():
    for step in range(128):  # Gerar 128 tokens
        input_for_pred = generated[:, -128:]
        logits = model(input_for_pred)
        next_token = logits[0, -1].argmax().item()
        generated = torch.cat([
            generated,
            torch.tensor([[next_token]], dtype=torch.long).to(device)
        ], dim=1)

generated_text = tokenizer.decode(generated[0].cpu().tolist())
print(f"Gerado (primeiros 64 chars): '{generated_text[len(input_text):len(input_text)+64]}'")
print(f"Gerado (últimos 64 chars): '{generated_text[-64:]}'")

# Comparar com esperado
print(f"\nComparação:")
print(f"  Esperado:  '{target_text[:64]}'")
print(f"  Gerado:    '{generated_text[len(input_text):len(input_text)+64]}'")

# ===== Teste 3: Probabilidades do modelo =====
print("\n" + "=" * 70)
print("TESTE 3: Distribuição de Probabilidades")
print("=" * 70)

# Pegar um ponto no meio da sequência
input_seq = input_ids[0].cpu().tolist()
target_seq = target_ids[0].cpu().tolist()

# Testar posições específicas
test_positions = [50, 75, 100, 127]

with torch.no_grad():
    input_tensor = input_ids[0:1]  # [1, seq_len]
    logits = model(input_tensor)  # [1, seq_len, vocab_size]

print(f"Sequência: '{tokenizer.decode(input_seq)[:80]}...'")

for pos in test_positions:
    if pos < len(input_seq):
        curr_char = tokenizer.decode([input_seq[pos]])
        next_expected = tokenizer.decode([target_seq[pos]])
        
        # Logits nesta posição
        pos_logits = logits[0, pos, :]
        probs = F.softmax(pos_logits, dim=-1)
        
        # Top 5 predições
        top_probs, top_indices = torch.topk(probs, k=5)
        
        print(f"\nPosição {pos} (atual: '{curr_char}', esperado próximo: '{next_expected}'):")
        for i, (prob, idx) in enumerate(zip(top_probs, top_indices)):
            char = tokenizer.decode([idx.item()])
            is_correct = "✓" if idx.item() == target_seq[pos] else " "
            print(f"  {i+1}. '{char}' ({prob:.4f}) {is_correct}")

# ===== Teste 4: Ver o que Shakespeare realmente tem =====
print("\n" + "=" * 70)
print("TESTE 4: Análise do Dataset")
print("=" * 70)

with open("data/shakespeare.txt", "r") as f:
    text = f.read()

# Ver sequências que começam com nossos prompts
prompts = ["To be", "The king", "my name"]

for prompt in prompts:
    if prompt in text:
        idx = text.find(prompt)
        snippet = text[idx:idx+80]
        print(f"\nPrompt '{prompt}' encontrado no dataset:")
        print(f"  '{snippet}'")
    else:
        print(f"\nPrompt '{prompt}' NÃO encontrado no dataset")
