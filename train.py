"""
train.py - Script para treinar o modelo GPT Mini

Este script:
1. Carregar dataset e criar DataLoaders
2. Inicializar modelo
3. Loop de treinamento com validação
4. Salvar checkpoint
5. Plotar loss durante treinamento

Uso:
    python3 train.py
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import json

from model import GPTMini, create_data_loaders


def train_epoch(model, train_loader, optimizer, criterion, device, epoch):
    """
    Executar uma época de treinamento.
    
    Args:
        model: Modelo GPTMini
        train_loader: DataLoader com dados de treino
        optimizer: Otimizador (Adam, SGD, etc)
        criterion: Função de loss (CrossEntropyLoss)
        device: 'cpu' ou 'cuda'
        epoch: Número da época
    
    Returns:
        float: Loss médio da época
    """
    model.train()  # Modo de treinamento
    total_loss = 0
    num_batches = 0
    
    pbar = tqdm(train_loader, desc=f"Época {epoch+1}", leave=False)
    
    for input_ids, target_ids in pbar:
        input_ids = input_ids.to(device)
        target_ids = target_ids.to(device)
        
        # ===== Forward pass =====
        # input_ids: [batch_size, seq_len]
        # logits: [batch_size, seq_len, vocab_size]
        logits = model(input_ids)
        
        # ===== Calcular loss =====
        # Reshape para calcular loss
        # Cross-entropy espera [batch_size*seq_len, vocab_size] e [batch_size*seq_len]
        loss = criterion(
            logits.view(-1, model.vocab_size),
            target_ids.view(-1)
        )
        # Explicação:
        # Para cada posição em cada batch, queremos que o modelo prediga o token correto
        # Loss = -log(prob do token correto)
        # Média sobre todas as posições e batches
        
        # ===== Backward pass =====
        # Calcular gradientes
        optimizer.zero_grad()  # Limpar gradientes anteriores
        loss.backward()         # Calcular novos gradientes
        
        # ===== Clip gradients (opcional) =====
        # Prevenir "exploding gradients"
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        # ===== Update pesos =====
        optimizer.step()  # Atualizar pesos baseado em gradientes
        
        # Acumular loss
        total_loss += loss.item()
        num_batches += 1
        
        # Update barra de progresso
        pbar.set_postfix({'loss': loss.item():.4f})
    
    avg_loss = total_loss / num_batches
    return avg_loss


def validate(model, val_loader, criterion, device):
    """
    Avaliar modelo em dataset de validação.
    
    Args:
        model: Modelo GPTMini
        val_loader: DataLoader com dados de validação
        criterion: Função de loss
        device: 'cpu' ou 'cuda'
    
    Returns:
        float: Loss médio na validação
    """
    model.eval()  # Modo de avaliação (sem dropout, etc)
    total_loss = 0
    num_batches = 0
    
    with torch.no_grad():  # Não calcular gradientes (mais rápido)
        for input_ids, target_ids in val_loader:
            input_ids = input_ids.to(device)
            target_ids = target_ids.to(device)
            
            # Forward pass
            logits = model(input_ids)
            
            # Calcular loss
            loss = criterion(
                logits.view(-1, model.vocab_size),
                target_ids.view(-1)
            )
            
            total_loss += loss.item()
            num_batches += 1
    
    avg_loss = total_loss / num_batches
    return avg_loss


def train_model(
    model_name="gpt_mini",
    dataset_path="data/shakespeare.txt",
    checkpoint_dir="model",
    seq_len=128,
    batch_size=32,
    num_epochs=5,
    learning_rate=1e-3,
    device=None,
    save_every=1
):
    """
    Executar loop completo de treinamento.
    
    Args:
        model_name: Nome do modelo (para salvar)
        dataset_path: Caminho do dataset
        checkpoint_dir: Diretório para salvar checkpoints
        seq_len: Comprimento da sequência
        batch_size: Tamanho do batch
        num_epochs: Número de épocas
        learning_rate: Taxa de aprendizado
        device: 'cpu', 'cuda', ou None (auto-detect)
        save_every: Salvar checkpoint a cada N épocas
    """
    
    # ===== Setup =====
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Usando device: {device}")
    
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # ===== Criar DataLoaders =====
    print("\n" + "=" * 60)
    print("CARREGANDO DATASET")
    print("=" * 60)
    
    train_loader, val_loader, tokenizer = create_data_loaders(
        dataset_path,
        seq_len=seq_len,
        batch_size=batch_size,
        train_split=0.95,
        num_workers=0
    )
    
    # Salvar tokenizador
    tokenizer.save(os.path.join(checkpoint_dir, "tokenizer.pkl"))
    
    # ===== Criar modelo =====
    print("\n" + "=" * 60)
    print("CRIANDO MODELO")
    print("=" * 60)
    
    model = GPTMini(
        vocab_size=tokenizer.vocab_size,
        max_seq_len=seq_len,
        d_model=512,
        num_heads=8,
        num_layers=4,
        d_ff=2048,
        dropout=0.1
    ).to(device)
    
    num_params = model.get_num_parameters()
    size_mb = model.get_model_size_mb()
    
    print(f"Parâmetros: {num_params:,}")
    print(f"Tamanho: {size_mb:.2f} MB")
    print(f"Device: {device}")
    
    # ===== Setup de otimização =====
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.5)
    
    # ===== Loop de treinamento =====
    print("\n" + "=" * 60)
    print("TREINANDO")
    print("=" * 60)
    
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    
    for epoch in range(num_epochs):
        print(f"\nÉpoca {epoch + 1}/{num_epochs}")
        
        # Treino
        train_loss = train_epoch(
            model, train_loader, optimizer, criterion, device, epoch
        )
        train_losses.append(train_loss)
        print(f"  Train Loss: {train_loss:.4f}")
        
        # Validação
        val_loss = validate(model, val_loader, criterion, device)
        val_losses.append(val_loss)
        print(f"  Val Loss:   {val_loss:.4f}")
        
        # Scheduler step
        scheduler.step()
        
        # Salvar checkpoint
        if (epoch + 1) % save_every == 0:
            checkpoint_path = os.path.join(
                checkpoint_dir,
                f"{model_name}_epoch_{epoch+1}.pt"
            )
            
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
                'config': {
                    'vocab_size': tokenizer.vocab_size,
                    'max_seq_len': seq_len,
                    'd_model': 512,
                    'num_heads': 8,
                    'num_layers': 4,
                    'd_ff': 2048,
                }
            }, checkpoint_path)
            
            print(f"  Checkpoint salvo: {checkpoint_path}")
        
        # Salvar melhor modelo
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_checkpoint_path = os.path.join(checkpoint_dir, f"{model_name}_best.pt")
            
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
                'config': {
                    'vocab_size': tokenizer.vocab_size,
                    'max_seq_len': seq_len,
                    'd_model': 512,
                    'num_heads': 8,
                    'num_layers': 4,
                    'd_ff': 2048,
                }
            }, best_checkpoint_path)
            
            print(f"  Novo melhor modelo salvo!")
    
    # ===== Salvar histórico =====
    print("\n" + "=" * 60)
    print("FINALIZANDO")
    print("=" * 60)
    
    history = {
        'train_losses': train_losses,
        'val_losses': val_losses,
        'epochs': num_epochs
    }
    
    history_path = os.path.join(checkpoint_dir, f"{model_name}_history.json")
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2)
    
    # ===== Plotar loss =====
    print("\nPlotando loss...")
    
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Train Loss', marker='o')
    plt.plot(val_losses, label='Val Loss', marker='s')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Treinamento: GPT Mini com Shakespeare')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    plot_path = os.path.join(checkpoint_dir, f"{model_name}_loss.png")
    plt.savefig(plot_path, dpi=100)
    print(f"Gráfico salvo: {plot_path}")
    plt.show()
    
    # ===== Resumo final =====
    print("\n" + "=" * 60)
    print("RESUMO FINAL")
    print("=" * 60)
    print(f"Épocas completadas: {num_epochs}")
    print(f"Loss final de treino: {train_losses[-1]:.4f}")
    print(f"Loss final de validação: {val_losses[-1]:.4f}")
    print(f"Melhor loss de validação: {best_val_loss:.4f}")
    print(f"Parâmetros: {num_params:,}")
    print(f"\nArquivos salvos em: {checkpoint_dir}")
    print("  - gpt_mini_best.pt (melhor modelo)")
    print("  - gpt_mini_loss.png (gráfico)")
    print("  - tokenizer.pkl (tokenizador)")
    print("  - gpt_mini_history.json (histórico de loss)")
    print("=" * 60)


# ===== Teste de geração =====

def generate_text(
    checkpoint_path="model/gpt_mini_best.pt",
    tokenizer_path="model/tokenizer.pkl",
    prompt="To be",
    max_length=100,
    temperature=1.0
):
    """
    Usar modelo treinado para gerar texto.
    
    Args:
        checkpoint_path: Caminho do modelo salvo
        tokenizer_path: Caminho do tokenizador
        prompt: Texto inicial
        max_length: Máximo de tokens para gerar
        temperature: Criatividade (0.1 = determinístico, 2.0 = criativo)
    """
    from model import CharacterTokenizer, GPTMini
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Carregar
    tokenizer = CharacterTokenizer.load(tokenizer_path)
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
    
    # Gerar
    print(f"Prompt: {prompt}")
    print("-" * 60)
    
    prompt_ids = tokenizer.encode(prompt)
    generated_ids = model.generate(
        prompt_ids,
        max_length=max_length,
        temperature=temperature,
        device=device
    )
    
    generated_text = tokenizer.decode(generated_ids)
    print(generated_text)
    print("-" * 60)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Treinar GPT Mini")
    parser.add_argument('--epochs', type=int, default=5, help='Número de épocas')
    parser.add_argument('--batch-size', type=int, default=32, help='Tamanho do batch')
    parser.add_argument('--lr', type=float, default=1e-3, help='Taxa de aprendizado')
    parser.add_argument('--seq-len', type=int, default=128, help='Comprimento da sequência')
    parser.add_argument('--generate', action='store_true', help='Apenas gerar texto (não treinar)')
    
    args = parser.parse_args()
    
    if args.generate:
        print("Gerando texto com modelo treinado...")
        generate_text(
            prompt="To be or",
            max_length=100,
            temperature=0.8
        )
    else:
        print(f"Iniciando treinamento...")
        print(f"  Epochs: {args.epochs}")
        print(f"  Batch size: {args.batch_size}")
        print(f"  Learning rate: {args.lr}")
        print(f"  Seq length: {args.seq_len}")
        
        train_model(
            num_epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.lr,
            seq_len=args.seq_len
        )
