"""
utils.py - Utilidades para tokenização e carregamento de dados

Este arquivo contém:
1. Tokenizador (converter texto em IDs e vice-versa)
2. Dataset loader (ler arquivo Shakespeare e criar batches)
3. Vocabulário construtor
"""

import os
import pickle
import torch
from torch.utils.data import Dataset, DataLoader


class CharacterTokenizer:
    """
    Tokenizador de nível de caractere.
    
    Converte:
    - Texto → IDs de caracteres (encoding)
    - IDs de caracteres → Texto (decoding)
    
    Por que character-level?
    - Simples e didático
    - Vocabulário pequeno (256 caracteres ASCII)
    - Tudo é conhecido (sem "out of vocabulary")
    
    Alternativas:
    - Word-level: Vocabulário grande, tokens raros desconhecidos
    - BPE (Byte Pair Encoding): Mais complexo, usado em GPT real
    
    Args:
        vocab_file (str): Caminho para salvar/carregar vocabulário
    """
    
    def __init__(self, vocab_file=None):
        # Criar mapeamento de caracteres
        # Usar todos caracteres ASCII imprimíveis + caracteres especiais
        self.chars = sorted(list(set(
            chr(i) for i in range(32, 127)  # ASCII imprimível
        )))
        self.vocab_size = len(self.chars)
        
        # Mapeamentos
        self.char_to_id = {ch: i for i, ch in enumerate(self.chars)}
        self.id_to_char = {i: ch for i, ch in enumerate(self.chars)}
        
        self.vocab_file = vocab_file
    
    def encode(self, text):
        """
        Converter texto em IDs de caracteres.
        
        Args:
            text (str): Texto para codificar
        
        Returns:
            list: IDs dos caracteres
        
        Exemplo:
            tokenizer = CharacterTokenizer()
            ids = tokenizer.encode("Hello")
            # ids = [72, 101, 108, 108, 111]
        """
        return [self.char_to_id.get(ch, 0) for ch in text]
    
    def decode(self, ids):
        """
        Converter IDs de caracteres em texto.
        
        Args:
            ids (list ou torch.Tensor): IDs para decodificar
        
        Returns:
            str: Texto decodificado
        
        Exemplo:
            ids = [72, 101, 108, 108, 111]
            text = tokenizer.decode(ids)
            # text = "Hello"
        """
        if isinstance(ids, torch.Tensor):
            ids = ids.tolist()
        
        # Se é 2D (batch), converter primeiro elemento
        if isinstance(ids, list) and len(ids) > 0 and isinstance(ids[0], list):
            ids = ids[0]
        
        return ''.join(self.id_to_char.get(id, '?') for id in ids)
    
    def save(self, filepath):
        """Salvar tokenizador em arquivo."""
        data = {
            'chars': self.chars,
            'char_to_id': self.char_to_id,
            'id_to_char': self.id_to_char
        }
        with open(filepath, 'wb') as f:
            pickle.dump(data, f)
    
    @staticmethod
    def load(filepath):
        """Carregar tokenizador de arquivo."""
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        
        tokenizer = CharacterTokenizer()
        tokenizer.chars = data['chars']
        tokenizer.char_to_id = data['char_to_id']
        tokenizer.id_to_char = data['id_to_char']
        tokenizer.vocab_size = len(tokenizer.chars)
        
        return tokenizer


class ShakespeareDataset(Dataset):
    """
    Dataset para Shakespeare.
    
    Funcionamento:
    1. Carregar arquivo de texto completo
    2. Tokenizar (converter em IDs)
    3. Criar "janelas" de comprimento fixed
    
    Exemplo:
    Texto: "abcdefgh"
    seq_len = 3
    
    Amostras:
    - Input: [a, b, c] → Target: d
    - Input: [b, c, d] → Target: e
    - Input: [c, d, e] → Target: f
    - ...
    
    Args:
        filepath (str): Caminho do arquivo de texto
        seq_len (int): Comprimento da sequência de entrada
        tokenizer (CharacterTokenizer): Tokenizador para usar
    """
    
    def __init__(self, filepath, seq_len=128, tokenizer=None, stride=None):
        # Carregar arquivo
        print(f"Carregando {filepath}...")
        with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
            text = f.read()
        
        # Criar tokenizador se não fornecido
        if tokenizer is None:
            tokenizer = CharacterTokenizer()
        
        self.tokenizer = tokenizer
        self.seq_len = seq_len
        self.stride = stride if stride is not None else seq_len  # Non-overlapping por padrão
        self.vocab_size = tokenizer.vocab_size
        
        # Tokenizar
        print("Tokenizando...")
        self.token_ids = torch.tensor(tokenizer.encode(text), dtype=torch.long)
        self.num_tokens = len(self.token_ids)
        
        print(f"  Total de tokens: {self.num_tokens:,}")
        print(f"  Tamanho do vocabulário: {self.vocab_size}")
        print(f"  Comprimento da sequência: {seq_len}")
        print(f"  Número de amostras: {len(self):,}")
    
    def __len__(self):
        """
        Número de amostras possíveis.
        
        Com stride=seq_len (padrão), cria chunks não-sobrepostos.
        Reduz amostras de 5.3M (stride=1) para ~42k (stride=seq_len).
        
        Returns:
            int: Número de amostras
        """
        # Calcular quantas "janelas" cabem com o stride especificado
        return max(0, (self.num_tokens - self.seq_len) // self.stride + 1)
    
    def __getitem__(self, idx):
        """
        Obter uma amostra.
        
        Args:
            idx (int): Índice da amostra
        
        Returns:
            tuple: (input_ids, target_ids)
                - input_ids: [seq_len]
                - target_ids: [seq_len] (shifted by 1)
        
        Exemplo:
            dataset = ShakespeareDataset("data.txt", seq_len=10)
            input_ids, target_ids = dataset[0]
            # input_ids = tokens[0:10]
            # target_ids = tokens[1:11]
        """
        # Converter índice do dataset para posição no token array usando stride
        start_idx = idx * self.stride
        input_ids = self.token_ids[start_idx:start_idx + self.seq_len]
        target_ids = self.token_ids[start_idx + 1:start_idx + self.seq_len + 1]
        
        return input_ids, target_ids


def create_data_loaders(
    filepath,
    seq_len=128,
    batch_size=32,
    train_split=0.9,
    num_workers=0,
    tokenizer=None
):
    """
    Criar DataLoaders para treino e validação.
    
    Args:
        filepath (str): Caminho do arquivo de texto
        seq_len (int): Comprimento das sequências
        batch_size (int): Tamanho do batch
        train_split (float): Proporção para treino (resto é validação)
        num_workers (int): Número de workers (0 = single-threaded)
        tokenizer (CharacterTokenizer): Tokenizador (None = criar novo)
    
    Returns:
        tuple: (train_loader, val_loader, tokenizer)
    
    Exemplo:
        train_loader, val_loader, tokenizer = create_data_loaders(
            "data/shakespeare.txt",
            seq_len=128,
            batch_size=32
        )
        
        for input_ids, target_ids in train_loader:
            # Treinar...
    """
    
    # Criar tokenizador
    if tokenizer is None:
        tokenizer = CharacterTokenizer()
    
    # Criar dataset completo
    dataset = ShakespeareDataset(filepath, seq_len, tokenizer)
    
    # Dividir em treino e validação
    train_size = int(len(dataset) * train_split)
    val_size = len(dataset) - train_size
    
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset,
        [train_size, val_size]
    )
    
    print(f"\nDataset dividido:")
    print(f"  Treino: {train_size:,} amostras")
    print(f"  Validação: {val_size:,} amostras")
    
    # Criar loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=False
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=False
    )
    
    print(f"\nDataLoaders:")
    print(f"  Batches treino por época: {len(train_loader)}")
    print(f"  Batches validação: {len(val_loader)}")
    print(f"  Batch size: {batch_size}")
    
    return train_loader, val_loader, tokenizer


# ===== Testes =====

if __name__ == "__main__":
    """
    Testes para tokenizador e dataset.
    """
    
    print("=" * 60)
    print("Testando Tokenizador e Dataset")
    print("=" * 60)
    
    # ===== Teste 1: CharacterTokenizer =====
    print("\n✓ Testando CharacterTokenizer...")
    tokenizer = CharacterTokenizer()
    
    # Encode
    text = "To be or not to be"
    token_ids = tokenizer.encode(text)
    print(f"  Texto: '{text}'")
    print(f"  IDs: {token_ids[:10]}...")
    print(f"  Vocab size: {tokenizer.vocab_size}")
    
    # Decode
    decoded = tokenizer.decode(token_ids)
    print(f"  Decodificado: '{decoded}'")
    assert text == decoded, "Encode/decode falhou!"
    print("  ✅ CharacterTokenizer OK")
    
    # ===== Teste 2: ShakespeareDataset =====
    print("\n✓ Testando ShakespeareDataset...")
    
    # Verificar se arquivo existe
    if os.path.exists("data/shakespeare.txt"):
        dataset = ShakespeareDataset(
            "data/shakespeare.txt",
            seq_len=128,
            tokenizer=tokenizer
        )
        
        print(f"  Dataset size: {len(dataset):,}")
        
        # Obter uma amostra
        input_ids, target_ids = dataset[0]
        print(f"  Input shape: {input_ids.shape}")
        print(f"  Target shape: {target_ids.shape}")
        print(f"  Input sample: {input_ids[:10].tolist()}...")
        
        assert input_ids.shape[0] == 128, "Input shape incorreto!"
        assert target_ids.shape[0] == 128, "Target shape incorreto!"
        print("  ✅ ShakespeareDataset OK")
        
        # ===== Teste 3: DataLoaders =====
        print("\n✓ Testando DataLoaders...")
        train_loader, val_loader, tok = create_data_loaders(
            "data/shakespeare.txt",
            seq_len=64,
            batch_size=16,
            train_split=0.9,
            tokenizer=tokenizer
        )
        
        # Obter um batch
        for input_ids_batch, target_ids_batch in train_loader:
            print(f"\n  Batch shapes:")
            print(f"    Input: {input_ids_batch.shape}")
            print(f"    Target: {target_ids_batch.shape}")
            print(f"    Expected: (16, 64)")
            
            assert input_ids_batch.shape == (16, 64), "Batch shape incorreto!"
            assert target_ids_batch.shape == (16, 64), "Batch shape incorreto!"
            break
        
        print("  ✅ DataLoaders OK")
    else:
        print("  ⚠️  Arquivo data/shakespeare.txt não encontrado")
        print("     (teste pulado)")
    
    print("\n" + "=" * 60)
    print("✅ TODOS OS TESTES PASSARAM!")
    print("=" * 60)
