"""
gpt_mini.py - Modelo GPT Mini completo

Este arquivo contém o modelo Transformer para geração de texto.

Arquitetura:
Input tokens → Embedding + Positional Encoding → N blocos Transformer → Linear projection → Softmax → Output logits

O modelo aprender a prever o próximo token dada uma sequência anterior.
"""

import torch
import torch.nn as nn
from model.transformer import PositionalEncoding, TransformerBlock


class GPTMini(nn.Module):
    """
    GPT Mini - Modelo Transformer simplificado para geração de texto.
    
    Diferenças em relação a GPT real:
    - GPT: 96 camadas, 175B parâmetros (GPT-3)
    - GPT Mini: 4 camadas, ~2M parâmetros (PoC educacional)
    
    Ainda assim, mantém todos os conceitos principais:
    ✓ Embeddings de token
    ✓ Positional encoding
    ✓ Multi-head attention
    ✓ Feed-forward networks
    ✓ Geração autoregressiva (próximo token por vez)
    
    Args:
        vocab_size (int): Tamanho do vocabulário (ex: 256 para caracteres)
        max_seq_len (int): Comprimento máximo da sequência
        d_model (int): Dimensão dos embeddings e saídas dos blocos
        num_heads (int): Número de cabeças de atenção
        num_layers (int): Número de blocos Transformer
        d_ff (int): Dimensão do feed-forward network (padrão: 4*d_model)
        dropout (float): Taxa de dropout
    """
    
    def __init__(
        self,
        vocab_size,
        max_seq_len=512,
        d_model=512,
        num_heads=8,
        num_layers=4,
        d_ff=None,
        dropout=0.1
    ):
        super(GPTMini, self).__init__()
        
        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len
        self.d_model = d_model
        
        if d_ff is None:
            d_ff = 4 * d_model
        
        # ===== Embeddings =====
        # Converter IDs de tokens em vetores d_model-dimensionais
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        
        # Adicionar informação de posição (necessário para Transformers!)
        self.positional_encoding = PositionalEncoding(d_model, max_seq_len, dropout)
        
        # ===== Stack de Transformer Blocks =====
        # Cada bloco tem: MultiHeadAttention + FeedForward + Skip connections
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])
        
        # ===== Layer Normalization final =====
        # Normalizar antes da projeção final
        self.final_norm = nn.LayerNorm(d_model)
        
        # ===== Projeção de saída =====
        # Converter d_model dimensões para logits de vocab_size
        # (probabilidades sobre qual é o próximo token)
        self.output_projection = nn.Linear(d_model, vocab_size)
        
        # Inicializar pesos
        self._init_weights()
    
    def _init_weights(self):
        """
        Inicializar pesos com valores razoáveis.
        Inicialização ruim pode fazer o modelo nunca treinar.
        """
        for name, param in self.named_parameters():
            if param.dim() > 1:
                # Matrizes: usar Xavier uniform
                nn.init.xavier_uniform_(param)
            else:
                # Vieses: inicializar com zero
                nn.init.constant_(param, 0)
    
    def forward(self, input_ids, attention_mask=None):
        """
        Forward pass do modelo.
        
        Args:
            input_ids (torch.LongTensor): IDs de tokens [batch_size, seq_len]
            attention_mask (torch.Tensor): Máscara de atenção (opcional) [batch_size, seq_len]
        
        Returns:
            torch.Tensor: Logits [batch_size, seq_len, vocab_size]
                         (probabilidades não-normalizadas do próximo token)
        
        Exemplo:
            input_ids = torch.tensor([[1, 2, 3]])  # batch_size=1, seq_len=3
            logits = model(input_ids)
            # logits.shape = [1, 3, vocab_size]
            
            # Pegar predição para último token
            next_token_logits = logits[0, -1, :]  # [vocab_size]
            next_token_id = next_token_logits.argmax()
        """
        batch_size, seq_len = input_ids.size()
        
        # ===== Passo 1: Token Embedding =====
        # input_ids: [batch_size, seq_len]
        # embedding: [batch_size, seq_len, d_model]
        x = self.token_embedding(input_ids)
        
        # Escalar embeddings por sqrt(d_model)
        # (técnica de normalização que ajuda o treinamento)
        x = x * (self.d_model ** 0.5)
        
        # ===== Passo 2: Adicionar Positional Encoding =====
        # x: [batch_size, seq_len, d_model]
        x = self.positional_encoding(x)
        
        # ===== Passo 3: Passar por Transformer Blocks =====
        # Causal mask: posição i só pode ver posições 0..i (triângulo inferior)
        # Essencial para GPT - sem isso o modelo "trapaceia" vendo tokens futuros
        causal_mask = torch.tril(
            torch.ones(seq_len, seq_len, device=input_ids.device)
        ).unsqueeze(0).unsqueeze(0)  # [1, 1, seq_len, seq_len]

        for transformer_block in self.transformer_blocks:
            x = transformer_block(x, mask=causal_mask)
        
        # ===== Passo 4: Layer Normalization final =====
        x = self.final_norm(x)
        
        # ===== Passo 5: Projeção para vocabulário =====
        # Converter [batch_size, seq_len, d_model] → [batch_size, seq_len, vocab_size]
        logits = self.output_projection(x)
        
        # logits: [batch_size, seq_len, vocab_size]
        # Interpretação: Para cada posição seq_len, temos vocab_size "pontuações"
        # indicando quão provável é cada token ser o próximo
        
        return logits
    
    def generate(
        self,
        prompt_ids,
        max_length=100,
        temperature=1.0,
        top_k=None,
        top_p=None,
        repetition_penalty=1.2,
        device='cpu'
    ):
        """
        Gerar sequência de tokens começando de um prompt.
        
        Funcionamento:
        1. Input: ['T', 'o', ' ', 'b', 'e'] (primeiros tokens)
        2. Processar com modelo → prever próximo token
        3. Amostrar próximo token com base em probabilidades
        4. Adicionar à sequência
        5. Repetir até max_length ou token especial de fim
        
        Args:
            prompt_ids (list ou torch.Tensor): IDs de tokens iniciais
            max_length (int): Máximo de tokens para gerar
            temperature (float): Controla criatividade
                - T < 1.0: Mais determinístico (repete padrões)
                - T = 1.0: Balanceado
                - T > 1.0: Mais aleatório (mais criativo mas com erros)
            top_k (int): Se fornecido, considerar apenas top-k tokens mais prováveis
            top_p (float): Se fornecido, usar nucleus sampling (tokens acumulando p% probabilidade)
            repetition_penalty (float): Penaliza tokens que já aparecem na sequência
                - 1.0: Sem penalidade (padrão, mas permite repetição)
                - 1.2: Penalidade moderada (recomendado, evita loops)
                - 2.0+: Penalidade forte (pode limitare geração natural)
            device (str): 'cpu' ou 'cuda'
        
        Returns:
            list: IDs dos tokens gerados
        
        Exemplo:
            prompt = [1, 2, 3]  # "To "
            generated = model.generate(
                prompt, 
                max_length=50, 
                temperature=0.8,
                repetition_penalty=1.2  # ← Evita loops!
            )
            # generated = [1, 2, 3, 45, 12, 8, ...]
        """
        
        # Converter para tensor se necessário
        if isinstance(prompt_ids, list):
            prompt_ids = torch.tensor([prompt_ids], dtype=torch.long).to(device)
        elif isinstance(prompt_ids, torch.Tensor):
            if prompt_ids.dim() == 1:
                prompt_ids = prompt_ids.unsqueeze(0).to(device)
        
        generated = prompt_ids.clone()
        
        # Gerar tokens um por um
        with torch.no_grad():  # Não calcular gradientes (mais rápido)
            for step in range(max_length):
                # Garantir que sequência não exceda max_seq_len
                input_ids = generated[:, -self.max_seq_len:]
                
                # Forward pass: obter logits para próximo token
                logits = self.forward(input_ids)  # [batch, seq_len, vocab_size]
                
                # Pegar logits do último token (o que queremos prever)
                next_token_logits = logits[:, -1, :]  # [batch, vocab_size]
                
                # ===== NOVO: Aplicar penalização de repetição =====
                if repetition_penalty != 1.0:
                    for batch_idx in range(generated.shape[0]):
                        # Iterar apenas nos tokens únicos presentes na sequência
                        unique_tokens, counts = generated[batch_idx].unique(return_counts=True)
                        for token_id, token_count in zip(unique_tokens.tolist(), counts.tolist()):
                            next_token_logits[batch_idx, token_id] /= repetition_penalty ** token_count
                
                # ===== Converter logits em probabilidades (com temperatura) =====
                probs = torch.softmax(next_token_logits / temperature, dim=-1)
                
                # ===== Aplicar filtros (opcional) =====
                if top_k is not None:
                    # Top-k sampling: manter apenas os k tokens mais prováveis
                    top_k_probs, top_k_indices = torch.topk(
                        probs, 
                        k=min(top_k, self.vocab_size),
                        dim=-1
                    )
                    
                    # Criar tensor zerado e preencher com top-k
                    probs_filtered = torch.zeros_like(probs)
                    probs_filtered.scatter_(-1, top_k_indices, top_k_probs)
                    
                    # Re-normalizar para garantir que soma = 1
                    probs = probs_filtered / (probs_filtered.sum(dim=-1, keepdim=True) + 1e-10)
                
                if top_p is not None:
                    # Nucleus (top-p) sampling: manter tokens que acumulam p% da probabilidade
                    sorted_probs, sorted_indices = torch.sort(probs, descending=True, dim=-1)
                    cumsum_probs = torch.cumsum(sorted_probs, dim=-1)

                    # Remover tokens onde o cumsum ANTES deste token já excedeu top_p
                    # (shift de 1 para incluir o token que causa o cumsum ultrapassar p)
                    sorted_indices_to_remove = cumsum_probs - sorted_probs > top_p
                    # Garantir que pelo menos 1 token é mantido
                    sorted_indices_to_remove[..., 0] = False
                    
                    # Zerar probabilidades dos tokens marcados
                    sorted_probs[sorted_indices_to_remove] = 0
                    
                    # Reverter para índices originais
                    probs_filtered = torch.zeros_like(probs)
                    probs_filtered.scatter_(-1, sorted_indices, sorted_probs)
                    
                    # Re-normalizar
                    probs = probs_filtered / (probs_filtered.sum(dim=-1, keepdim=True) + 1e-10)
                
                # ===== Amostrar próximo token =====
                # Selecionar aleatoriamente baseado em probabilidades
                next_token = torch.multinomial(probs, num_samples=1)  # [batch, 1]
                
                # Adicionar à sequência gerada
                generated = torch.cat([generated, next_token], dim=1)
        
        # Retornar como lista (remover batch dimension)
        return generated[0].cpu().tolist()
    
    def get_num_parameters(self):
        """
        Contar número de parâmetros treináveis.
        
        Returns:
            int: Número de parâmetros
        
        Exemplo:
            params = model.get_num_parameters()
            print(f"Modelo tem {params:,} parâmetros")
        """
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def get_model_size_mb(self):
        """
        Estimar tamanho do modelo em MB.
        
        Returns:
            float: Tamanho em MB
        """
        num_params = self.get_num_parameters()
        # Assumir float32 = 4 bytes por parâmetro
        size_bytes = num_params * 4
        size_mb = size_bytes / (1024 * 1024)
        return size_mb


# ===== Testes =====

if __name__ == "__main__":
    """
    Testes para validar o modelo GPT Mini.
    """
    
    print("=" * 60)
    print("Testando GPT Mini")
    print("=" * 60)
    
    # Configuração
    vocab_size = 256  # Caracteres ASCII
    batch_size = 2
    seq_len = 10
    
    # Criar modelo
    model = GPTMini(
        vocab_size=vocab_size,
        max_seq_len=512,
        d_model=256,
        num_heads=8,
        num_layers=4,
        d_ff=1024,
        dropout=0.1
    )
    
    print("\n✓ Testando forward pass...")
    input_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
    logits = model(input_ids)
    
    print(f"  Input shape: {input_ids.shape}")
    print(f"  Output logits shape: {logits.shape}")
    print(f"  Esperado: ({batch_size}, {seq_len}, {vocab_size})")
    assert logits.shape == (batch_size, seq_len, vocab_size), "Shape incorreto!"
    print("  ✅ Forward pass OK")
    
    print("\n✓ Testando parâmetros do modelo...")
    num_params = model.get_num_parameters()
    size_mb = model.get_model_size_mb()
    print(f"  Número de parâmetros: {num_params:,}")
    print(f"  Tamanho estimado: {size_mb:.2f} MB")
    print("  ✅ Parâmetros OK")
    
    print("\n✓ Testando geração de texto...")
    prompt = [ord('T'), ord('o'), ord(' ')]  # "To "
    generated = model.generate(
        prompt,
        max_length=20,
        temperature=1.0
    )
    
    print(f"  Prompt IDs: {prompt}")
    print(f"  Comprimento gerado: {len(generated)}")
    print(f"  Generated IDs: {generated[:15]}...")  # Mostrar primeiros 15
    
    # Converter para caracteres (se válido)
    try:
        generated_text = ''.join(chr(c) if 32 <= c < 127 else '?' for c in generated)
        print(f"  Generated text: {generated_text[:50]}...")
    except:
        pass
    
    print("  ✅ Geração OK")
    
    print("\n" + "=" * 60)
    print("✅ TODOS OS TESTES PASSARAM!")
    print("=" * 60)
