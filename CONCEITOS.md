# 🧠 Guia Didático: Transformers e LLMs

*Um guia completo para entender como Transformers funcionam e por que são a base dos LLMs modernos como ChatGPT*

---

## Índice

1. [Redes Neurais 101](#redes-neurais-101)
2. [Por Que RNNs Não Servem](#por-que-rnns-não-servem)
3. [Attention Mechanism](#attention-mechanism)
4. [Multi-Head Attention](#multi-head-attention)
5. [Positional Encoding](#positional-encoding)
6. [Transformer Block](#transformer-block)
7. [Arquitetura Completa do Transformer](#arquitetura-completa-do-transformer)
8. [Geração de Texto (Autoregressivo)](#geração-de-texto-autoregressivo)
9. [Treinamento e Loss](#treinamento-e-loss)
10. [Troubleshooting & FAQ](#troubleshooting--faq)

---

## Redes Neurais 101

### O que é uma Rede Neural?

Uma rede neural é inspirada no cérebro humano. Consiste em camadas de "neurônios" que processam informações.

```
Input → [Layer 1] → [Layer 2] → [Layer 3] → Output
```

### Exemplo Simples: Perceptron

```python
# Um neurônio básico
import numpy as np

def perceptron(input_vector, weights, bias):
    # Produto escalar: sum(input * weights)
    z = np.dot(input_vector, weights) + bias
    
    # Ativação: ReLU (Rectified Linear Unit)
    output = max(0, z)  # Se z < 0, output = 0; senão output = z
    
    return output

# Exemplo
input_vec = np.array([1.0, 2.0, 3.0])
weights = np.array([0.5, -0.3, 0.2])
bias = 0.1

result = perceptron(input_vec, weights, bias)
print(f"Output: {result}")  # (~0.9)
```

### Backpropagation (Treinamento)

O processo para "ensinar" a rede:

1. **Forward pass**: Dados entram, rede prediz
2. **Calcular erro**: Comparar predição com valor real
3. **Backward pass**: Atualizar pesos para reduzir erro (gradiente descendente)

```
Dado um exemplo: input=[1,2], target=5, output=3 (errado!)
Erro = (5 - 3)² = 4
Ajustar pesos: w_novo = w_velho - learning_rate × ∂Erro/∂w
```

**Chave**: Para cada iteração, reduzir gradualmente o erro.

---

## Por Que RNNs Não Servem

### RNNs (Recurrent Neural Networks) - Primeira Tentativa

RNNs foram populares antes dos Transformers. Eles processam sequências **uma palavra por vez**, mantendo uma "memória":

```
Input:  "O gato subiu"
        ↓    ↓      ↓
Step 1: "O" + hidden_state → output_1, new_hidden_state
Step 2: "gato" + new_hidden_state → output_2, new_hidden_state
Step 3: "subiu" + new_hidden_state → output_3
```

### Problema: Vanishing Gradient (Gradiente Que Desaparece)

Quando treinamos RNNs com sequências longas, o gradiente fica **exponencialmente pequeno**:

```
Sequência: [palavra_1, palavra_2, ..., palavra_100]

Gradiente para palavra_1 ao final da sequência:
gradient = ∂loss/∂w ≈ 0.0000000001...
(fica praticamente zero!)

Resultado: Pesos da palavra_1 não são atualizados → modelo não aprende dependências de longo alcance
```

**Analogia**: Como falar no telefone num corredor longo. A mensagem passa por muitos pontos (gradientes) e vai perdendo força (desaparecendo).

### Por Que Transformers Resolvem Isso?

Transformers usam **Attention**, que permite a rede:
- Olhar para **qualquer palavra** da sequência em qualquer momento
- **Não sequencial**: Processa tudo em paralelo
- Gradientes viajam mais curtos → não desaparecem

---

## Attention Mechanism

### Intuição: O Que é Attention?

Imagine que você está lendo uma frase:
```
"O gato comeu o rato que fugiu do telhado"
         ↑ (queremos prever a próxima palavra)
```

Seu cérebro "presta atenção" em diferentes palavras com diferentes pesos:
- "rato" → atenção alta (50%)
- "gato" → atenção média (30%)
- "comeu" → atenção média (15%)
- "o" (artigo) → atenção baixa (5%)

**Attention faz exatamente isso**: pesa quais partes da entrada são relevantes.

### Matemática: Scaled Dot-Product Attention

```
Attention(Q, K, V) = softmax(Q·K^T / √d_k) · V
```

Partes:
- **Q (Query)**: "O que estou procurando?" (palavra atual)
- **K (Key)**: "Identificadores das outras palavras"
- **V (Value)**: "Informação que quero extrair"
- **softmax**: Converte números em probabilidades (sum=1)
- **√d_k**: Escala para estabilidade numérica

### Exemplo Numérico (Simplificado)

```python
import numpy as np
from scipy.special import softmax

# Simulando atenção para prever token após "gato"
# Seqência: "O gato comeu"
d_k = 2  # dimensão

# Representações (embeddings)
Query = np.array([[1.0, 0.5]])      # "gato" - O que procuro
Keys = np.array([
    [0.9, 0.4],                     # "O"
    [1.0, 0.5],                     # "gato"
    [0.7, 0.8]                      # "comeu"
])
Values = np.array([
    [0.2, 0.1],
    [0.5, 0.6],
    [0.8, 0.7]
])

# Passo 1: Calcular scores (similaridade)
scores = Query @ Keys.T / np.sqrt(d_k)
# scores = [[0.96], [1.0], [0.85]]

# Passo 2: Aplicar softmax (converter em probabilidades)
attention_weights = softmax(scores)
# attention_weights ≈ [[0.33], [0.38], [0.29]]

# Passo 3: Aplicar ao Values
output = attention_weights.T @ Values
# output = contexto ponderado

print(f"Attention weights: {attention_weights.flatten()}")
# Resultado: [0.33, 0.38, 0.29]
# Modelo presta mais atenção a "gato" (ele mesmo) e "comeu"
```

**Interpretação**: O modelo olha para:
- 33% para "O"
- 38% para "gato"
- 29% para "comeu"

---

## Multi-Head Attention

### Por Que Múltiplas Cabeças?

Uma única atenção é limitada. Com **múltiplas cabeças**, o modelo aprende **diferentes aspectos** simultaneamente:

```
Head 1: "Qual token precede este?"
        ↓ (presta atenção em sequência)

Head 2: "Qual verbo está associado?"
        ↓ (presta atenção em verbos)

Head 3: "Qual é o sujeito?"
        ↓ (presta atenção em nomes)
```

### Arquitetura Multi-Head

```
Input x
  ↓
[W_Q] [W_K] [W_V]        (Linear projections para cabeça 1)
  ↓    ↓    ↓
Attention(Q₁, K₁, V₁) → output₁

[W_Q] [W_K] [W_V]        (Linear projections para cabeça 2)
  ↓    ↓    ↓
Attention(Q₂, K₂, V₂) → output₂

...

[W_Q] [W_K] [W_V]        (Linear projections para cabeça h)
  ↓    ↓    ↓
Attention(Qₕ, Kₕ, Vₕ) → outputₕ

Concatenate [output₁, output₂, ..., outputₕ]
  ↓
[W_O] (Linear layer)
  ↓
Final output
```

### Código (Simplificado)

```python
import torch
import torch.nn as nn

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model=512, num_heads=8):
        super().__init__()
        self.num_heads = num_heads
        self.d_k = d_model // num_heads  # Dimensão por cabeça
        
        # Projeções lineares
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
    
    def forward(self, Q, K, V):
        batch_size = Q.shape[0]
        
        # Projetar e dividir em múltiplas cabeças
        Q = self.W_q(Q).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        K = self.W_k(K).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        V = self.W_v(V).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        
        # Calcular atenção para cada cabeça
        scores = torch.matmul(Q, K.transpose(-2, -1)) / torch.sqrt(torch.tensor(self.d_k).float())
        attention_weights = torch.softmax(scores, dim=-1)
        context = torch.matmul(attention_weights, V)
        
        # Concatenar cabeças
        context = context.transpose(1, 2).contiguous()
        context = context.view(batch_size, -1, self.d_model)
        
        # Projeção final
        output = self.W_o(context)
        return output
```

**Benefício**: Cada cabeça aprende padrões diferentes. Com 8 heads, temos 8 "lentes" diferentes para ver os dados.

---

## Positional Encoding

### O Problema: Transformers são Não-Sequenciais

RNNs processam token por token: naturalmente entendem ordem.

Transformers processam **todos tokens simultaneamente** em paralelo:

```
RNN (sequencial):
"O" → "O gato" → "O gato comeu"

Transformer (paralelo):
["O", "gato", "comeu"] (tudo ao mesmo tempo!)
```

**Problema**: Como o modelo sabe a ordem das palavras?

```
Frase 1: "O gato comeu o rato"
Frase 2: "O rato comeu o gato"
         ↑ mesmos tokens, ordem diferente!
```

### Solução: Positional Encoding

Adicionar **informação de posição** a cada token:

```
token_1 = embedding("O") + pos_encoding(0)
token_2 = embedding("gato") + pos_encoding(1)
token_3 = embedding("comeu") + pos_encoding(2)
```

### Fórmula (Sine/Cosine)

```
PE(pos, 2i) = sin(pos / 10000^(2i/d_model))
PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))

Onde:
- pos = posição na sequência (0, 1, 2, ...)
- i = dimensão
- d_model = tamanho do embedding
```

### Visualização

```python
import numpy as np
import matplotlib.pyplot as plt

d_model = 512
seq_len = 100

pos = np.arange(seq_len)[:, np.newaxis]
i = np.arange(d_model)[np.newaxis, :]

pe = np.zeros((seq_len, d_model))
pe[:, 0::2] = np.sin(pos / 10000 ** (2*i[0, 0::2] / d_model))
pe[:, 1::2] = np.cos(pos / 10000 ** (2*i[0, 1::2] / d_model))

# Visualizar (heatmap)
plt.imshow(pe, cmap='viridis', aspect='auto')
plt.title('Positional Encoding')
plt.colorbar()
plt.show()
```

**Resultado**: Cada posição tem assinatura única, mas com padrão regular que permite o modelo aprender generalizar.

---

## Transformer Block

### Componentes Principais

Um bloco Transformer contém:

```
Input x
  ↓
┌─────────────────────────────────┐
│ Layer Normalization             │
│ MultiHeadAttention              │
└─────────────────────────────────┘
  ↓ (Add & Norm - skip connection)
  ↓
┌─────────────────────────────────┐
│ Layer Normalization             │
│ Feed-Forward Network (FFN)      │
│ [Dense(d_model → 4*d_model)] →  │
│ [ReLU] →                        │
│ [Dense(4*d_model → d_model)]    │
└─────────────────────────────────┘
  ↓ (Add & Norm - skip connection)
  ↓
Output y
```

### Por Que Skip Connections?

```python
# Sem skip connection:
x1 = attention(x0)
x2 = ffn(x1)
x3 = attention(x2)
...
# Problema: Gradientes podem desaparecer/explodir

# Com skip connection (residual):
x1 = x0 + attention(x0)  # ← "x0 sobrevive"
x2 = x1 + ffn(x1)        # ← "x1 sobrevive"
x3 = x2 + attention(x2)  # ← "x2 sobrevive"
# Benefício: Gradientes fluem melhor
```

### Layer Normalization

```python
def layer_norm(x, gamma, beta, eps=1e-5):
    mean = x.mean()
    std = x.std()
    x_normalized = (x - mean) / (std + eps)
    return gamma * x_normalized + beta
```

**Objetivo**: Estabilizar treinamento, fazendo entradas ter média 0 e variância 1.

### Feed-Forward Network

```python
# FFN simples
ffn = Sequential([
    Linear(d_model, 4 * d_model),  # Expande
    ReLU(),                         # Não-linear
    Linear(4 * d_model, d_model)    # Comprime
])

# Exemplo: d_model=512
# Camada 1: 512 → 2048 (expande)
# ReLU: 2048 → 2048 (não-linear)
# Camada 2: 2048 → 512 (volta)
```

**Intuição**: Expande para capturar padrões complexos, depois comprime.

---

## Arquitetura Completa do Transformer

### Diagrama Geral

```
Input tokens: ["O", "gato", "comeu"]
       ↓
Embedding + Positional Encoding
       ↓
┌──────────────────────────┐
│ Transformer Block 1      │ ← Attention + FFN
│ - 8 heads de atenção     │
│ - 512-dim vectors        │
└──────────────────────────┘
       ↓
┌──────────────────────────┐
│ Transformer Block 2      │ ← Repetir padrão
└──────────────────────────┘
       ↓
     ... (mais blocos)
       ↓
┌──────────────────────────┐
│ Transformer Block N      │
└──────────────────────────┘
       ↓
Linear layer (proj para vocab size)
       ↓
Softmax → probabilidades
       ↓
Output: [p("o"), p("rato"), p("que"), ...]
                  ↑ (maior probabilidade)
```

### Código Estrutura

```python
import torch
import torch.nn as nn

class GPTMini(nn.Module):
    def __init__(self, vocab_size, d_model=512, num_heads=8, num_layers=4):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = PositionalEncoding(d_model, max_seq_len=1000)
        
        # Stack de N blocos transformer
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(d_model, num_heads) 
            for _ in range(num_layers)
        ])
        
        # Camada final para predição
        self.output_layer = nn.Linear(d_model, vocab_size)
    
    def forward(self, token_ids):
        # token_ids: shape [batch_size, seq_len]
        
        # Embedding
        x = self.embedding(token_ids)  # [batch_size, seq_len, d_model]
        
        # Adicionar posição
        x = self.pos_encoding(x)
        
        # Passar por cada bloco transformer
        for transformer_block in self.transformer_blocks:
            x = transformer_block(x)
        
        # Projetar para vocabulário
        logits = self.output_layer(x)  # [batch_size, seq_len, vocab_size]
        
        return logits
```

---

## Geração de Texto (Autoregressivo)

### Como o Modelo Gera Texto

```
Input: "O gato"
       ↓
Modelo processa: ["O", "gato"]
       ↓
Output: [p("o"), p("comeu"), p("subiu"), ...]
                     ↑ (maior probabilidade)
       ↓
Próximo token = "comeu"
       ↓

Repetir com: ["O", "gato", "comeu"]
       ↓
Gera próximo token...
```

### Decodificação: Greedy vs Sampling

**Greedy** (determinístico):
```python
next_token = argmax(probabilities)  # Sempre a maior probabilidade
# Resultado: Sempre a mesma saída

Exemplo:
Input: "Uma vez"
Output: "Uma vez havia um príncipe" (sempre igual)
```

**Sampling** (aleatório):
```python
next_token = sample(probabilities, temperature=1.0)
# Resultado: Variado, às vezes com erros

Temperatura:
- T = 0.1: Muito determinístico (próximo de greedy)
- T = 1.0: Balanceado
- T = 2.0: Muito aleatório e criativo
```

### Código

```python
def generate(model, prompt_tokens, max_length=100, temperature=1.0):
    generated = prompt_tokens.copy()
    
    for _ in range(max_length):
        # Processar sequência atual
        logits = model(torch.tensor([generated]))
        # [1, seq_len, vocab_size]
        
        # Pegar último token (próxima predição)
        next_logits = logits[0, -1, :]  # [vocab_size]
        
        # Aplicar temperatura
        scaled_logits = next_logits / temperature
        
        # Obter probabilidades
        probs = torch.softmax(scaled_logits, dim=-1)
        
        # Amostrar próximo token
        next_token = torch.multinomial(probs, num_samples=1).item()
        
        generated.append(next_token)
        
        # Parar se token de fim (EOS)
        if next_token == EOS_TOKEN:
            break
    
    return generated
```

---

## Treinamento e Loss

### Função de Loss: Cross-Entropy

O modelo treina predizendo o **próximo token** dada uma sequência:

```
Sequência: ["O", "gato", "comeu", "o", "rato"]
            ↓    ↓      ↓      ↓     ↓
Alvo:     ["gato", "comeu", "o", "rato", EOS]

Loss = 0

Para cada posição:
1. Modelo prediz: "gato" com prob 0.8 (correto!) → loss += -log(0.8) = 0.22
2. Modelo prediz: "comeu" com prob 0.6 → loss += -log(0.6) = 0.51
3. Modelo prediz: "o" com prob 0.9 → loss += -log(0.9) = 0.11
4. Modelo prediz: "rato" com prob 0.7 → loss += -log(0.7) = 0.36
5. Modelo prediz: "EOS" com prob 0.5 → loss += -log(0.5) = 0.69

Total loss = 0.22 + 0.51 + 0.11 + 0.36 + 0.69 = 1.89
```

### Loop de Treinamento

```python
def train_epoch(model, dataloader, optimizer, device):
    model.train()
    total_loss = 0
    
    for batch_idx, (input_ids, target_ids) in enumerate(dataloader):
        input_ids = input_ids.to(device)
        target_ids = target_ids.to(device)
        
        # Forward pass
        logits = model(input_ids)  # [batch, seq_len, vocab]
        
        # Calcular loss
        loss = nn.CrossEntropyLoss()(
            logits.view(-1, vocab_size),
            target_ids.view(-1)
        )
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    return total_loss / len(dataloader)
```

### Métricas de Progresso

```
Época 1: Loss = 5.23 (alto, modelo não entende)
Época 5: Loss = 3.12 (melhora)
Época 10: Loss = 2.01 (converge)
Época 20: Loss = 1.85 (praticamente estável)
```

**Regra**: Se loss não decresce → learning rate muito alto/baixo, problema no código, ou modelo muito pequeno.

---

## Troubleshooting & FAQ

### P: Por que meu modelo não converge?

**R**: Possíveis causas:

1. **Learning rate** errado
   - Muito alto: Loss explode
   - Muito baixo: Loss não muda
   - Solução: Testar valores (1e-4, 5e-4, 1e-3)

2. **Modelo muito pequeno**
   - Não consegue aprender padrões complexos
   - Solução: Aumentar num_layers, d_model, num_heads

3. **Dataset muito pequeno**
   - Modelo memoriza, não generaliza
   - Solução: Mais dados ou mais simples

### P: Qual é o erro de "exploding gradients"?

**R**: Gradientes ficam muito grandes durante backprop:

```
gradient = 100000 (muito grande!)
weight_novo = weight_velho - learning_rate * gradient
           = 1.0 - 0.001 * 100000
           = 1.0 - 100 = -99 (completamente errado!)
```

Solução:
```python
# Gradient clipping
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
```

### P: Qual é a diferença entre training e validation loss?

**R**:

```
Training loss:   Loss no dataset que modelo vê
Validation loss: Loss no dataset que modelo NÃO viu

Se validation >> training: Overfitting (memorizou)
Se ambos decrescem: Normal, modelo aprendendo
Se validation aumenta: Overfitting, parar treinamento
```

### P: Como sou se modelo está funcionando bem?

**R**: Checklist:

- ✅ Loss decresce ao longo das épocas
- ✅ Modelo gera texto coerente (não random)
- ✅ Aumentar temperatura → texto mais criativo
- ✅ Reduzir temperatura → texto mais repetitivo
- ✅ Heatmap de atenção mostra padrões razoáveis

### P: Qual é a diferença entre "teacher forcing" e free running?

**R**:

```
Teacher forcing (treinamento):
Input: ["O", "gato", "comeu"]
Target: ["gato", "comeu", "o"]
Modelo usa TARGET real para próxima predição

Free running (geração):
Input: "O"
Gera: "gato"
Usa "gato" predito (pode estar errado!) para próximo
```

**Problema**: Distribuição shift (modelo nunca treina com seus próprios erros).

---

## Resumo Visual

```
Transformer Block:
┌─ Atende para qual palavra/contexto é relevante? (Attention)
│
├─ Processa com Feed-Forward Network
│
└─ Repetir N vezes (4-96 blocos) para GPT gigante

Multi-Head Attention:
├─ 8 "câmeras" diferentes olhando os dados
├─ Head 1: Sintaxe
├─ Head 2: Semântica
└─ Heads 3-8: Outros padrões

Position Encoding:
├─ "Palavra 1 está no início"
├─ "Palavra 2 está no meio"
└─ "Palavra 3 está perto do fim"

Treinar:
├─ Prever próximo token
├─ Comparar com alvo real
├─ Atualizar pesos para reduzir erro
└─ Repetir bilhões de vezes!
```

---

## Recursos Adicionais

- [Attention is All You Need](https://arxiv.org/abs/1706.03762) - Paper original (legível!)
- [3Blue1Brown - Attention](https://www.youtube.com/watch?v=eMlx5aFJsrQ) - Visualização excelente
- [Andrej Karpathy - makemore](https://github.com/karpathy/makemore) - Implementação educacional

---

**Última atualização**: Maio 2026 | Versão: 1.0
