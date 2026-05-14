"""
train.py - Treinar o modelo GPT Mini

Uso:
    python train.py                  # Treinar com defaults
    python train.py --epochs 50      # Ajustar épocas
    python train.py --generate       # Gerar texto com modelo salvo
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from tqdm import tqdm
import json

from model import GPTMini, create_data_loaders, CharacterTokenizer


def _detect_device():
    if torch.cuda.is_available():
        return 'cuda'
    if torch.backends.mps.is_available():
        return 'mps'
    return 'cpu'


def train_epoch(model, train_loader, optimizer, criterion, device, epoch):
    model.train()
    total_loss = 0
    num_batches = 0

    pbar = tqdm(train_loader, desc=f"Época {epoch+1}", leave=False)

    for input_ids, target_ids in pbar:
        input_ids = input_ids.to(device)
        target_ids = target_ids.to(device)

        logits = model(input_ids)
        loss = criterion(logits.view(-1, model.vocab_size), target_ids.view(-1))

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item()
        num_batches += 1
        pbar.set_postfix({'loss': f'{loss.item():.4f}'})

    return total_loss / num_batches


def validate(model, val_loader, criterion, device):
    model.eval()
    total_loss = 0
    num_batches = 0

    with torch.no_grad():
        for input_ids, target_ids in val_loader:
            input_ids = input_ids.to(device)
            target_ids = target_ids.to(device)

            logits = model(input_ids)
            loss = criterion(logits.view(-1, model.vocab_size), target_ids.view(-1))

            total_loss += loss.item()
            num_batches += 1

    return total_loss / num_batches


def train_model(
    model_name="gpt_mini",
    dataset_path="data/shakespeare.txt",
    checkpoint_dir="model",
    seq_len=128,
    batch_size=32,
    num_epochs=50,
    learning_rate=3e-4,
    device=None,
    early_stopping_patience=5,
    d_model=256,
    num_heads=8,
    num_layers=3,
    d_ff=1024,
    dropout=0.5,
):
    if device is None:
        device = _detect_device()
    print(f"Usando device: {device}")

    os.makedirs(checkpoint_dir, exist_ok=True)

    print("\n" + "=" * 60)
    print("CARREGANDO DATASET")
    print("=" * 60)

    train_loader, val_loader, tokenizer = create_data_loaders(
        dataset_path,
        seq_len=seq_len,
        batch_size=batch_size,
        train_split=0.95,
        num_workers=0,
    )
    tokenizer.save(os.path.join(checkpoint_dir, "tokenizer.pkl"))

    print("\n" + "=" * 60)
    print("CRIANDO MODELO")
    print("=" * 60)

    model = GPTMini(
        vocab_size=tokenizer.vocab_size,
        max_seq_len=seq_len,
        d_model=d_model,
        num_heads=num_heads,
        num_layers=num_layers,
        d_ff=d_ff,
        dropout=dropout,
    ).to(device)

    print(f"Parâmetros: {model.get_num_parameters():,}")
    print(f"Tamanho:    {model.get_model_size_mb():.2f} MB")
    print(f"Device:     {device}")

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=num_epochs, eta_min=1e-6
    )

    print("\n" + "=" * 60)
    print("TREINANDO (com Early Stopping)")
    print("=" * 60)

    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    patience_counter = 0
    best_epoch = 0

    config = {
        'vocab_size': tokenizer.vocab_size,
        'max_seq_len': seq_len,
        'd_model': d_model,
        'num_heads': num_heads,
        'num_layers': num_layers,
        'd_ff': d_ff,
        'dropout': dropout,
    }

    for epoch in range(num_epochs):
        print(f"\nÉpoca {epoch + 1}/{num_epochs}")

        train_loss = train_epoch(model, train_loader, optimizer, criterion, device, epoch)
        val_loss = validate(model, val_loader, criterion, device)

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        print(f"  Train Loss: {train_loss:.4f}")
        print(f"  Val Loss:   {val_loss:.4f}")

        scheduler.step()

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch + 1
            patience_counter = 0

            best_path = os.path.join(checkpoint_dir, f"{model_name}_best.pt")
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
                'config': config,
            }, best_path)
            print(f"  Novo melhor modelo! (val_loss={best_val_loss:.4f})")
        else:
            patience_counter += 1
            print(f"  Sem melhora ({patience_counter}/{early_stopping_patience})")

        if patience_counter >= early_stopping_patience:
            print(f"\nEarly stopping na época {epoch + 1}")
            print(f"Melhor foi época {best_epoch} com val_loss={best_val_loss:.4f}")
            break

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

    plot_path = os.path.join(checkpoint_dir, f"{model_name}_loss.png")
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(train_losses) + 1), train_losses, label='Train Loss', marker='o')
    plt.plot(range(1, len(val_losses) + 1), val_losses, label='Val Loss', marker='s')
    plt.axvline(x=best_epoch, color='green', linestyle='--', label=f'Best (Epoch {best_epoch})')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Treinamento GPT Mini')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(plot_path, dpi=100)
    print(f"Gráfico salvo: {plot_path}")

    print(f"\nMelhor época:     {best_epoch}")
    print(f"Melhor val loss:  {best_val_loss:.6f}")
    print(f"Arquivos em:      {checkpoint_dir}/")


def generate_text(
    checkpoint_path="model/gpt_mini_best.pt",
    tokenizer_path="model/tokenizer.pkl",
    prompt="To be",
    max_length=200,
    temperature=0.8,
    top_p=0.95,
    repetition_penalty=1.2,
):
    device = _detect_device()

    tokenizer = CharacterTokenizer.load(tokenizer_path)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    cfg = checkpoint['config']

    model = GPTMini(
        vocab_size=cfg['vocab_size'],
        max_seq_len=cfg['max_seq_len'],
        d_model=cfg['d_model'],
        num_heads=cfg['num_heads'],
        num_layers=cfg['num_layers'],
        d_ff=cfg['d_ff'],
        dropout=cfg.get('dropout', 0.1),
    ).to(device)

    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    print(f"Prompt: {prompt}")
    print("-" * 60)

    prompt_ids = tokenizer.encode(prompt)
    generated_ids = model.generate(
        prompt_ids,
        max_length=max_length,
        temperature=temperature,
        top_p=top_p,
        repetition_penalty=repetition_penalty,
        device=device,
    )

    print(tokenizer.decode(generated_ids))
    print("-" * 60)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Treinar/usar GPT Mini")
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--seq-len', type=int, default=128)
    parser.add_argument('--patience', type=int, default=5)
    parser.add_argument('--generate', action='store_true', help='Gerar texto com modelo salvo')
    parser.add_argument('--prompt', type=str, default='To be or', help='Prompt para geração')

    args = parser.parse_args()

    if args.generate:
        generate_text(prompt=args.prompt)
    else:
        print(f"Iniciando treinamento...")
        train_model(
            num_epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.lr,
            seq_len=args.seq_len,
            early_stopping_patience=args.patience,
        )
