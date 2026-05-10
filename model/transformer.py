"""
transformer.py - Blocos fundamentais do Transformer

Este arquivo contém os componentes básicos para construir um modelo Transformer:
1. PositionalEncoding - Adiciona informação de posição aos tokens
2. MultiHeadAttention - Mecanismo de atenção com múltiplas "cabeças"
3. TransformerBlock - Bloco completo (Attention + FFN)

Referência: "Attention is All You Need" (Vaswani et al., 2017)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class PositionalEncoding(nn.Module):
    """
    Positional Encoding usando sin/cos.
    
    Problema: Transformers processam dados em paralelo, não sequencial.
    Como sabe qual token vem primeiro?
    
    Solução: Adicionar informação de posição a cada token.
    
    Fórmula:
    PE(pos, 2i) = sin(pos / 10000^(2i/d_model))
    PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
    
    Args:
        d_model (int): Dimensão do embedding (ex: 512)
        max_seq_len (int): Comprimento máximo da sequência (ex: 1000)
        dropout (float): Taxa de dropout (padrão: 0.1)
    """
    
    def __init__(self, d_model, max_seq_len=1000, dropout=0.1):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        # Criar matriz PE de tamanho [max_seq_len, d_model]
        pe = torch.zeros(max_seq_len, d_model)
        position = torch.arange(0, max_seq_len, dtype=torch.float).unsqueeze(1)
        # position: [max_seq_len, 1] - índice de cada posição
        
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * 
            -(math.log(10000.0) / d_model)
        )
        # div_term: [d_model/2] - denominator para sin/cos
        
        # Preencher posições pares com sin
        pe[:, 0::2] = torch.sin(position * div_term)
        
        # Preencher posições ímpares com cos
        pe[:, 1::2] = torch.cos(position * div_term)
        
        # Registrar como buffer (não é parâmetro, mas persiste)
        self.register_buffer('pe', pe.unsqueeze(0))  # [1, max_seq_len, d_model]
    
    def forward(self, x):
        """
        Adicionar positional encoding ao embedding.
        
        Args:
            x (torch.Tensor): Embeddings de entrada [batch_size, seq_len, d_model]
        
        Returns:
            torch.Tensor: Embeddings + posição [batch_size, seq_len, d_model]
        """
        seq_len = x.size(1)
        # x + PE (PE é expandido para batch_size automaticamente)
        x = x + self.pe[:, :seq_len, :]
        return self.dropout(x)


class MultiHeadAttention(nn.Module):
    """
    Multi-Head Attention Mechanism.
    
    Por que múltiplas cabeças?
    Uma única atenção é limitada. Com 8 cabeças, o modelo aprende DIFERENTES aspectos:
    - Head 1: Relação entre palavras adjacentes
    - Head 2: Relação entre verbo e sujeito
    - Head 3: Encontrar objetos no contexto
    - ... (5 mais)
    
    Matemática (Single Head):
    Attention(Q, K, V) = softmax((Q·K^T) / √d_k) · V
    
    Multi-Head (parallelo):
    head_i = Attention(Q·W_q_i, K·W_k_i, V·W_v_i)
    concat_heads = [head_1, head_2, ..., head_h]
    output = concat_heads · W_o
    
    Args:
        d_model (int): Dimensão do modelo (ex: 512)
        num_heads (int): Número de cabeças de atenção (ex: 8)
        dropout (float): Taxa de dropout
    """
    
    def __init__(self, d_model, num_heads, dropout=0.1):
        super(MultiHeadAttention, self).__init__()
        
        # Garantir que d_model é divisível por num_heads
        assert d_model % num_heads == 0, f"{d_model} não divisível por {num_heads}"
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads  # Dimensão por cabeça (ex: 512/8=64)
        
        # Projeções lineares para Q, K, V
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        
        # Projeção de saída (concatenar cabeças)
        self.W_o = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, Q, K, V, mask=None):
        """
        Calcular Multi-Head Attention.
        
        Args:
            Q (torch.Tensor): Query [batch_size, seq_len_q, d_model]
            K (torch.Tensor): Key [batch_size, seq_len_k, d_model]
            V (torch.Tensor): Value [batch_size, seq_len_v, d_model]
            mask (torch.Tensor): Máscara para padding/causalidade [opcional]
        
        Returns:
            torch.Tensor: Saída [batch_size, seq_len_q, d_model]
        """
        batch_size = Q.size(0)
        
        # ===== Passo 1: Projetar e dividir em múltiplas cabeças =====
        # Pré-projeção: [batch_size, seq_len, d_model]
        Q = self.W_q(Q)
        K = self.W_k(K)
        V = self.W_v(V)
        
        # Reshape: [batch_size, seq_len, num_heads, d_k]
        Q = Q.view(batch_size, -1, self.num_heads, self.d_k)
        K = K.view(batch_size, -1, self.num_heads, self.d_k)
        V = V.view(batch_size, -1, self.num_heads, self.d_k)
        
        # Transpor: [batch_size, num_heads, seq_len, d_k]
        # (para calcular atenção em paralelo para cada cabeça)
        Q = Q.transpose(1, 2)
        K = K.transpose(1, 2)
        V = V.transpose(1, 2)
        
        # ===== Passo 2: Calcular atenção =====
        # Scores: [batch_size, num_heads, seq_len_q, seq_len_k]
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        # Explicação:
        # - Q @ K^T: [batch, heads, seq_q, d_k] @ [batch, heads, d_k, seq_k]
        #          = [batch, heads, seq_q, seq_k]
        # - Dividir por √d_k: Estabilidade numérica
        
        # Aplicar máscara (se fornecida)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        
        # Softmax: [batch_size, num_heads, seq_len_q, seq_len_k]
        # (converte scores em probabilidades que somam 1)
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        # ===== Passo 3: Aplicar atenção aos Values =====
        # Context: [batch_size, num_heads, seq_len_q, d_k]
        context = torch.matmul(attention_weights, V)
        # Explicação:
        # [batch, heads, seq_q, seq_k] @ [batch, heads, seq_k, d_k]
        # = [batch, heads, seq_q, d_k]
        # Resultado: cada posição tem "informação ponderada" de todos outros
        
        # ===== Passo 4: Concatenar cabeças =====
        # Transpor: [batch_size, seq_len_q, num_heads, d_k]
        context = context.transpose(1, 2).contiguous()
        
        # Reshape: [batch_size, seq_len_q, d_model]
        context = context.view(batch_size, -1, self.d_model)
        
        # ===== Passo 5: Projeção final =====
        # Output: [batch_size, seq_len_q, d_model]
        output = self.W_o(context)
        
        return output


class FeedForwardNetwork(nn.Module):
    """
    Feed-Forward Network (FFN).
    
    Componente importante do Transformer Block.
    
    Estrutura:
    Dense(d_model → 4*d_model) → ReLU → Dense(4*d_model → d_model)
    
    Por que expande e depois comprime?
    - Expandir: Permite o modelo encontrar padrões complexos
    - Comprir: Volta ao tamanho original para próxima camada
    
    Args:
        d_model (int): Dimensão de entrada/saída
        d_ff (int): Dimensão intermitente (geralmente 4*d_model)
        dropout (float): Taxa de dropout
    """
    
    def __init__(self, d_model, d_ff=None, dropout=0.1):
        super(FeedForwardNetwork, self).__init__()
        
        if d_ff is None:
            d_ff = 4 * d_model  # Padrão: 4x d_model
        
        # Duas transformações lineares com ReLU no meio
        self.linear1 = nn.Linear(d_model, d_ff)      # Expande
        self.linear2 = nn.Linear(d_ff, d_model)      # Comprime
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.ReLU()
    
    def forward(self, x):
        """
        Forward pass: x → 4x → ReLU → x
        
        Args:
            x (torch.Tensor): [batch_size, seq_len, d_model]
        
        Returns:
            torch.Tensor: [batch_size, seq_len, d_model]
        """
        return self.linear2(self.dropout(self.activation(self.linear1(x))))


class TransformerBlock(nn.Module):
    """
    Um bloco Transformer completo.
    
    Estrutura:
    ┌─────────────────────────────┐
    │ Input x                     │
    ├─────────────────────────────┤
    │ LayerNorm                   │
    │ ↓                           │
    │ MultiHeadAttention          │ ← "O quê presto atenção"
    │ ↓                           │
    │ + x (skip connection)       │ ← "Mantém informação original"
    ├─────────────────────────────┤
    │ LayerNorm                   │
    │ ↓                           │
    │ FeedForward Network         │ ← "Processa com não-linearidade"
    │ ↓                           │
    │ + x (skip connection)       │
    ├─────────────────────────────┤
    │ Output                      │
    └─────────────────────────────┘
    
    Skip connections (x + output) são cruciais:
    - Sem elas: Gradientes desaparecem em modelos profundos
    - Com elas: Informação flui melhor através de muitas camadas
    
    Args:
        d_model (int): Dimensão do modelo
        num_heads (int): Número de cabeças de atenção
        d_ff (int): Dimensão do FFN (padrão: 4*d_model)
        dropout (float): Taxa de dropout
    """
    
    def __init__(self, d_model, num_heads, d_ff=None, dropout=0.1):
        super(TransformerBlock, self).__init__()
        
        # Layer Normalization
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        # Multi-Head Attention
        self.attention = MultiHeadAttention(d_model, num_heads, dropout)
        
        # Feed-Forward Network
        self.ffn = FeedForwardNetwork(d_model, d_ff, dropout)
        
        # Dropout para regularização
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, mask=None):
        """
        Forward pass com skip connections.
        
        Args:
            x (torch.Tensor): [batch_size, seq_len, d_model]
            mask (torch.Tensor): Máscara opcional [opcional]
        
        Returns:
            torch.Tensor: [batch_size, seq_len, d_model]
        """
        # ===== Sub-layer 1: Multi-Head Attention =====
        # x_normalized = LayerNorm(x)
        x_normalized = self.norm1(x)
        
        # attention_output = MultiHeadAttention(x_normalized, x_normalized, x_normalized)
        # (Query, Key, Value são todos x - "self-attention")
        attention_output = self.attention(x_normalized, x_normalized, x_normalized, mask)
        
        # x = x + attention_output (skip connection!)
        x = x + self.dropout(attention_output)
        
        # ===== Sub-layer 2: Feed-Forward Network =====
        # x_normalized = LayerNorm(x)
        x_normalized = self.norm2(x)
        
        # ffn_output = FeedForward(x_normalized)
        ffn_output = self.ffn(x_normalized)
        
        # x = x + ffn_output (skip connection!)
        x = x + self.dropout(ffn_output)
        
        return x


# ===== Testes Unitários =====

if __name__ == "__main__":
    """
    Testes para validar que cada componente funciona.
    """
    
    print("=" * 60)
    print("Testando Componentes do Transformer")
    print("=" * 60)
    
    # Parâmetros
    batch_size = 2
    seq_len = 10
    d_model = 512
    num_heads = 8
    d_ff = 2048
    
    # ===== Teste 1: PositionalEncoding =====
    print("\n✓ Testando PositionalEncoding...")
    pos_enc = PositionalEncoding(d_model=d_model, max_seq_len=100)
    x = torch.randn(batch_size, seq_len, d_model)
    x_with_pos = pos_enc(x)
    assert x_with_pos.shape == (batch_size, seq_len, d_model), "Shape incorreto!"
    print(f"  Input shape: {x.shape}")
    print(f"  Output shape: {x_with_pos.shape}")
    print("  ✅ PositionalEncoding OK")
    
    # ===== Teste 2: MultiHeadAttention =====
    print("\n✓ Testando MultiHeadAttention...")
    mha = MultiHeadAttention(d_model=d_model, num_heads=num_heads)
    Q = torch.randn(batch_size, seq_len, d_model)
    K = torch.randn(batch_size, seq_len, d_model)
    V = torch.randn(batch_size, seq_len, d_model)
    attention_output = mha(Q, K, V)
    assert attention_output.shape == (batch_size, seq_len, d_model), "Shape incorreto!"
    print(f"  Input Q shape: {Q.shape}")
    print(f"  Output shape: {attention_output.shape}")
    print(f"  Número de heads: {num_heads}")
    print(f"  Dimensão por head: {d_model // num_heads}")
    print("  ✅ MultiHeadAttention OK")
    
    # ===== Teste 3: FeedForwardNetwork =====
    print("\n✓ Testando FeedForwardNetwork...")
    ffn = FeedForwardNetwork(d_model=d_model, d_ff=d_ff)
    x_ffn = torch.randn(batch_size, seq_len, d_model)
    ffn_output = ffn(x_ffn)
    assert ffn_output.shape == (batch_size, seq_len, d_model), "Shape incorreto!"
    print(f"  Input shape: {x_ffn.shape}")
    print(f"  Dimensão intermédio: {d_ff} (4x d_model)")
    print(f"  Output shape: {ffn_output.shape}")
    print("  ✅ FeedForwardNetwork OK")
    
    # ===== Teste 4: TransformerBlock =====
    print("\n✓ Testando TransformerBlock...")
    transformer_block = TransformerBlock(
        d_model=d_model,
        num_heads=num_heads,
        d_ff=d_ff
    )
    x_block = torch.randn(batch_size, seq_len, d_model)
    block_output = transformer_block(x_block)
    assert block_output.shape == (batch_size, seq_len, d_model), "Shape incorreto!"
    print(f"  Input shape: {x_block.shape}")
    print(f"  Output shape: {block_output.shape}")
    print(f"  d_model: {d_model}")
    print(f"  num_heads: {num_heads}")
    print(f"  d_ff: {d_ff}")
    print("  ✅ TransformerBlock OK")
    
    # ===== Teste 5: Stack de múltiplos blocos =====
    print("\n✓ Testando Stack de Transformer Blocks...")
    num_layers = 4
    x_stack = torch.randn(batch_size, seq_len, d_model)
    
    for i in range(num_layers):
        block = TransformerBlock(d_model, num_heads, d_ff)
        x_stack = block(x_stack)
    
    assert x_stack.shape == (batch_size, seq_len, d_model), "Shape incorreto!"
    print(f"  Input shape: {torch.randn(batch_size, seq_len, d_model).shape}")
    print(f"  Output shape após {num_layers} blocos: {x_stack.shape}")
    print(f"  ✅ Stack de {num_layers} blocos OK")
    
    print("\n" + "=" * 60)
    print("✅ TODOS OS TESTES PASSARAM!")
    print("=" * 60)
