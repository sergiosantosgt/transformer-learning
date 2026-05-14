"""
train_improved.py - Treinamento melhorado com early stopping

Mudanças:
1. Early stopping quando val loss não melhora
2. Dropout maior (0.5)
3. Modelo um pouco menor para reduzir overfitting
4. Weight decay (L2 regularização)
5. Melhor seleção de checkpoint
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
    """Executar uma época de treinamento."""
    model.train()
    total_loss = 0
    num_batches = 0
    
    pbar = tqdm(train_loader, desc=f"Época {epoch+1}", leave=False)
    
    for input_ids, target_ids in pbar:
        input_ids = input_ids.to(device)
        target_ids = target_ids.to(device)
        
        logits = model(input_ids)
        loss = criterion(
            logits.view(-1, model.vocab_size),
            target_ids.view(-1)
        )
        
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        total_loss += loss.item()
        num_batches += 1
        
        pbar.set_postfix({'loss': f'{loss.item():.4f}'})
    
    avg_loss = total_loss / num_batches
    return avg_loss


def validate(model, val_loader, criterion, device):
    """Avaliar modelo em validação."""
    model.eval()
    total_loss = 0
    num_batches = 0
    
    with torch.no_grad():
        for input_ids, target_ids in val_loader:
            input_ids = input_ids.to(device)
            target_ids = target_ids.to(device)
            
            logits = model(input_ids)
            loss = criterion(
                logits.view(-1, model.vocab_size),
                target_ids.view(-1)
            )
            
            total_loss += loss.item()
            num_batches += 1
    
    avg_loss = total_loss / num_batches
    return avg_loss


def train_model_improved(
    model_name="gpt_mini",
    dataset_path="data/shakespeare.txt",
    checkpoint_dir="model",
    seq_len=128,
    batch_size=32,
    num_epochs=50,
    learning_rate=3e-4,
    device=None,
    early_stopping_patience=5
):
    """
    Treinamento com early stopping e regularização melhorada.
    
    Args:
        early_stopping_patience: Número de épocas sem melhora antes de parar
    """
    
    # ===== Setup =====
    if device is None:
        device = 'mps' if torch.backends.mps.is_available() else 'cpu'
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
    
    tokenizer.save(os.path.join(checkpoint_dir, "tokenizer.pkl"))
    
    # ===== Criar modelo (MENOR e com MAIS DROPOUT) =====
    print("\n" + "=" * 60)
    print("CRIANDO MODELO")
    print("=" * 60)
    
    model = GPTMini(
        vocab_size=tokenizer.vocab_size,
        max_seq_len=seq_len,
        d_model=256,  # Reduzido de 512
        num_heads=8,
        num_layers=3,  # Reduzido de 4
        d_ff=1024,     # Reduzido de 2048
        dropout=0.5    # Aumentado para 0.5
    ).to(device)
    
    num_params = model.get_num_parameters()
    size_mb = model.get_model_size_mb()
    
    print(f"Parâmetros: {num_params:,}")
    print(f"Tamanho: {size_mb:.2f} MB")
    print(f"Device: {device}")
    
    # ===== Setup de otimização =====
    criterion = nn.CrossEntropyLoss()
    # Weight decay = L2 regularização (reduz pesos grandes)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, 
        T_max=num_epochs,
        eta_min=1e-6
    )
    
    # ===== Loop de treinamento com Early Stopping =====
    print("\n" + "=" * 60)
    print("TREINANDO (com Early Stopping)")
    print("=" * 60)
    
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    patience_counter = 0
    best_epoch = 0
    
    for epoch in range(num_epochs):
        print(f"\nÉpoca {epoch + 1}/{num_epochs}")
        
        # Treino
        train_loss = train_epoch(
            model, train_loader, optimizer, criterion, device, epoch
        )
        
        # Validação
        val_loss = validate(model, val_loader, criterion, device)
        
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        
        print(f"  Train Loss: {train_loss:.4f}")
        print(f"  Val Loss:   {val_loss:.4f}")
        
        # Scheduler step
        scheduler.step()
        
        # ===== Early Stopping =====
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch + 1
            patience_counter = 0
            
            # Salvar melhor modelo
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
                    'd_model': 256,
                    'num_heads': 8,
                    'num_layers': 3,
                    'd_ff': 1024,
                }
            }, best_checkpoint_path)
            print(f"  ✅ Novo melhor modelo! (epoch {best_epoch})")
        else:
            patience_counter += 1
            print(f"  ⏸️  Sem melhora ({patience_counter}/{early_stopping_patience})")
        
        # Parar se não melhorou
        if patience_counter >= early_stopping_patience:
            print(f"\n⛔ Early stopping na época {epoch + 1}")
            print(f"   Melhor modelo foi na época {best_epoch} com val_loss={best_val_loss:.4f}")
            break
    
    # ===== Salvar histórico =====
    print("\n" + "=" * 60)
    print("FINALIZANDO")
    print("=" * 60)
    
    history_path = os.path.join(checkpoint_dir, f"{model_name}_history.json")
    with open(history_path, 'w') as f:
        json.dump({
            'train_losses': train_losses,
            'val_losses': val_losses,
            'epochs': len(train_losses),
            'best_epoch': best_epoch,
            'best_val_loss': float(best_val_loss),
        }, f, indent=2)
    
    # ===== Plot =====
    plot_path = os.path.join(checkpoint_dir, f"{model_name}_loss.png")
    
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(train_losses) + 1), train_losses, label='Train Loss', marker='o')
    plt.plot(range(1, len(val_losses) + 1), val_losses, label='Val Loss', marker='s')
    plt.axvline(x=best_epoch, color='green', linestyle='--', label=f'Best Model (Epoch {best_epoch})')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Treinamento do GPT Mini (com Early Stopping)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(plot_path, dpi=100)
    print(f"Gráfico salvo: {plot_path}")
    
    print(f"\nArquivos salvos em: {checkpoint_dir}")
    print(f"✅ Treinamento concluído!")
    print(f"   Melhor época: {best_epoch}")
    print(f"   Melhor val loss: {best_val_loss:.6f}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Treinar GPT Mini (Melhorado)")
    parser.add_argument('--epochs', type=int, default=50, help='Máximo de épocas')
    parser.add_argument('--batch-size', type=int, default=32, help='Tamanho do batch')
    parser.add_argument('--lr', type=float, default=3e-4, help='Taxa de aprendizado')
    parser.add_argument('--seq-len', type=int, default=128, help='Comprimento da sequência')
    parser.add_argument('--patience', type=int, default=5, help='Early stopping patience')
    
    args = parser.parse_args()
    
    print(f"Iniciando treinamento melhorado...")
    print(f"  Max Epochs: {args.epochs}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Learning rate: {args.lr}")
    print(f"  Seq length: {args.seq_len}")
    print(f"  Early stopping patience: {args.patience}")
    
    train_model_improved(
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        seq_len=args.seq_len,
        early_stopping_patience=args.patience
    )
