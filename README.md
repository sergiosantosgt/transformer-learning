# 🧠 GPT Mini: Aprendendo Transformers com Shakespeare

Um modelo Transformer simplificado para entender como LLMs (Large Language Models) funcionam internamente.

```
Input:  "O gato comeu"
         ↓
    [Transformer]
         ↓
Output: "O gato comeu o rato"
```

## 🚀 Instalação

### 1. Criar Virtual Environment

```bash
# Criar venv
python3 -m venv venv

# Ativar (Linux/Mac)
source venv/bin/activate

# Ativar (Windows)
venv\\Scripts\\activate
```

### 2. Instalar Dependências

```bash
pip install -r requirements.txt
```

### 3. Verificar Setup

```bash
# Verificar dependências
python -c "import torch, pandas, streamlit; print('✅ Tudo instalado!')"

# Verificar dataset
ls -lh data/shakespeare.txt
# Deve mostrar: ~5.2 MB
```

## ⚡ Quick Start

### 1. Treinar Modelo

```bash
python train.py --epochs 5 --batch-size 16 --seq-len 128
```

Será:
- ✅ Carregar dataset (reduzido com stride otimizado)
- ✅ Treinar 5 épocas
- ✅ Salvar modelo em `model/gpt_mini_best.pt`
- ✅ Plotar loss em `model/gpt_mini_loss.png`

Tempo esperado: **~40-50 minutos** (M1 CPU com otimização de stride)

### 2. Usar a Interface

```bash
streamlit run app.py
```

Abrirá em `http://localhost:8501` com:
- 📝 Input: Digite um prompt (ex: "To be")
- 🎲 Slider: Controlar temperatura (criatividade)
- 📊 Visualizações: Attention heatmap, probabilidades dos tokens

## 📁 Estrutura do Projeto

```
my_model/
├── data/
│   └── shakespeare.txt          # Dataset (5.17 MB)
├── model/
│   ├── transformer.py           # Blocos Transformer
│   ├── gpt_mini.py              # Modelo GPT Mini
│   ├── utils.py                 # Tokenização, dataset
│   └── checkpoint.pt            # Modelo treinado (gerado)
├── train.py                     # Script de treinamento
├── app.py                       # Interface Streamlit
├── CONCEITOS.md                 # Documentação didática (9k linhas!)
├── claude.md                    # Histórico de implementação
├── README.md                    # Este arquivo
└── requirements.txt             # Dependências Python
```

## 🎓 Aprendizado

### Entender Transformers

1. **Leia primeiro**: [CONCEITOS.md](CONCEITOS.md)
   - 10 seções com 50+ exemplos de código
   - Explica por que Transformers funcionam
   - Não precisa de background em ML

2. **Estude o código**:
   ```bash
   # Arquitetura do modelo
   cat model/gpt_mini.py
   
   # Forward pass e geração
   cat model/transformer.py
   ```

3. **Experiimente**:
   - Treinar com diferentes hiperparâmetros
   - Modificar temperatura na interface
   - Ver heatmap de atenção mudar

### Experimentos Sugeridos

```python
# Experimento 1: Aumentar número de layers
# Em train.py, mudar: num_layers=4 → num_layers=8
# Ver: Loss converge mais rápido? Qualidade melhora?

# Experimento 2: Aumentar número de heads
# Em train.py, mudar: num_heads=8 → num_heads=16
# Ver: Treinamento mais lento? Melhor qualidade?

# Experimento 3: Temperatura na interface
# Temperature = 0.1 → Repetitivo
# Temperature = 1.0 → Balanceado
# Temperature = 2.0 → Muito aleatório
```

## 🔧 Configuração

### Hiperparâmetros (em `train.py`)

```python
# Tamanho do modelo
d_model = 512           # Dimensão dos embeddings
num_heads = 8           # Número de heads de atenção
num_layers = 4          # Número de blocos Transformer
vocab_size = 256        # Caracteres únicos

# Treinamento
batch_size = 32
learning_rate = 1e-3
num_epochs = 10
seq_len = 128           # Comprimento das sequências

# Geração
max_length = 100        # Máximo de tokens gerados
temperature = 1.0       # Criatividade (0.1-2.0)
```

### Ajustar para Seu Hardware

**CPU (lento, mas funciona)**:
```python
batch_size = 16         # Reduzir
d_model = 256           # Reduzir
num_layers = 2          # Reduzir
num_epochs = 5          # Menos épocas
```

**GPU (rápido)**:
```python
# Mantém configuração padrão
# Pode aumentar batch_size → 64, 128
```

## 📊 Métricas

### Loss esperado durante treinamento

```
Épocas 1-3:    Loss: 4.0 → 2.5 (aprendendo rápido)
Épocas 4-7:    Loss: 2.5 → 1.5 (aprendendo normal)
Épocas 8-10:   Loss: 1.5 → 1.2 (convergindo)
```

Se loss não diminui → ver [CONCEITOS.md Troubleshooting](#troubleshooting--faq)

### Qualidade de Geração

Escala subjetiva:

```
1/5: Completamente random
2/5: Alguns padrões, muito erros
3/5: Texto coerente às vezes ✅
4/5: Muito bom, parece Shakespeare
5/5: Impossível distinguir de Shakespeare real
```

Esperado neste projeto: **2.5-3.5/5** (PoC)

## 🎯 Próximas Melhorias

- [ ] Fine-tuning em modelos pré-treinados
- [ ] Decoder stack (ler embeddings preditos)
- [ ] Beam search (além de greedy/sampling)
- [ ] Análise de token frequency
- [ ] Multi-GPU training
- [ ] Exportar para ONNX (deploy)

## 🐛 Troubleshooting

### Erro: "torch não encontrado"

```bash
# Verificar se venv está ativado
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows

# Reinstalar
pip install --upgrade torch numpy
```

### Erro: "module not found: model"

```bash
# Executar do diretório raiz do projeto
cd transformer-learning
python train.py
```

### Treinamento muito lento

```bash
# Reduzir tamanho do batch ou sequência
python train.py --batch-size 8 --seq-len 64 --epochs 1
```

Se ainda lento, reduzir ainda mais ou usar GPU (se disponível).

### "Dataset não encontrado"

```bash
# Verificar se existe
ls -la data/shakespeare.txt

# Se não existir, baixar
python3 << 'EOF'
import urllib.request
import ssl
ssl._create_default_https_context = ssl._create_unverified_context
urllib.request.urlretrieve("https://www.gutenberg.org/files/100/100-0.txt", "data/shakespeare.txt")
EOF
```

### "Streamlit command not found"

```bash
# Usar com python
python3 -m streamlit run app.py
```

### "Memory error / Out of memory"

```bash
# Reduzir batch size
batch_size = 8  # em train.py

# Reduzir modelo
d_model = 256
num_layers = 2
```

### "Loss não decresce"

1. Learning rate muito alto/baixo (testar 1e-4, 5e-4, 1e-3)
2. Modelo muito pequeno (aumentar d_model, num_layers)
3. Dataset muito pequeno (adicionar mais dados)

Ver [CONCEITOS.md Troubleshooting](#troubleshooting--faq) para soluções detalhadas.

## 📚 Referências

- **Origem**: "Attention is All You Need" (Vaswani et al., 2017)
- **PyTorch**: https://pytorch.org/tutorials/
- **Streamlit**: https://docs.streamlit.io/
- **Dataset**: Project Gutenberg Complete Works of Shakespeare

## 💡 Dicas para Aprender

1. **Não memorize fórmulas** - Entenda o "por quê"
2. **Rode o código** - Não só leia, execute!
3. **Modifique** - Mude hiperparâmetros, veja impacto
4. **Visualize** - Use heatmap de atenção para intuição
5. **Conecte** - Veja como cada peça do Transformer se encaixa

## 📝 Documentação

- [CONCEITOS.md](CONCEITOS.md) - Guia completo dos conceitos (9k linhas!)
- [claude.md](claude.md) - Histórico de cada passo executado
- Código comentado em `model/` e `train.py`

## 👨‍💻 Estrutura de Código

```python
# Arquitetura: model/transformer.py + model/gpt_mini.py
# ┌─────────────────────────────────┐
# │ GPTMini (Model)                 │
# ├─────────────────────────────────┤
# │ - Embedding + Positional Enc    │
# │ - 4 TransformerBlocks (cada)    │
# │   ├─ 8-Head Attention           │
# │   └─ Feed-Forward Network       │
# │ - Output Linear Projection      │
# └─────────────────────────────────┘

# Uso: model/utils.py
# ┌─────────────────────────────────┐
# │ Tokenização (character-level)   │
# │ Dataset Loading                 │
# │ Vocab Builder                   │
# └─────────────────────────────────┘

# Treinamento: train.py
# ┌─────────────────────────────────┐
# │ 1. Carregar dataset             │
# │ 2. Loop por épocas              │
# │ 3. Calcular loss                │
# │ 4. Backprop                     │
# │ 5. Atualizar pesos              │
# │ 6. Salvar checkpoint            │
# └─────────────────────────────────┘

# Interface: app.py
# ┌─────────────────────────────────┐
# │ Streamlit UI                    │
# │ - Input: prompt                 │
# │ - Slider: temperatura           │
# │ - Output: texto gerado          │
# │ - Vis: attention, probabilities │
# └─────────────────────────────────┘
```

## ⚙️ Desenvolvimento

### Agregar novo componente

1. Implementar em `model/`
2. Testar em Jupyter notebook (opcional)
3. Integrar em `train.py`
4. Documentar em `CONCEITOS.md`

### Debuggar modelo

```python
# Em train.py
print(f"Input shape: {input_ids.shape}")
print(f"Embedding output shape: {embeddings.shape}")
print(f"Attention output shape: {attention_out.shape}")
print(f"Logits shape: {logits.shape}")
print(f"Loss: {loss.item()}")
```

## 📞 Suporte

Para dúvidas sobre:
- **Conceitos**: Leia [CONCEITOS.md](CONCEITOS.md) (tem FAQ!)
- **Código**: Ver comentários nos arquivos `.py`
- **Implementação**: Ver [claude.md](claude.md) para decisões técnicas

## 📄 Licença

Educacional. Use, modifique, aprenda!

---

**Versão**: 1.0  
**Data**: Maio 2026  
**Status**: ✅ Pronto para Treinar

Comece com: `python3 train.py` 🚀
