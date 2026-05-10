# 🚀 Guia: Subir para GitHub

Este documento explica como subir seu projeto para GitHub.

---

## 📝 Nome do Repositório Recomendado

### **Opção Mais Recomendada:**
```
gpt-mini-shakespeare
```

**Por quê:**
- ✅ Claro e descritivo
- ✅ Combina modelo + dataset + educacional
- ✅ Fácil de pronunciar e lembrar
- ✅ SEO amigável (buscas no GitHub)

**Outras opções válidas:**
- `transformer-learning` (foco educacional)
- `llm-from-scratch` (aprendizado prático)
- `transformers-101` (referência educacional)

---

## 📊 O que Subir vs O que Não Subir

### ✅ **SUBIR para GitHub:**

```
my-model/
├── model/
│   ├── transformer.py       ✅ Código
│   ├── gpt_mini.py          ✅ Código
│   ├── utils.py             ✅ Código
│   └── __init__.py          ✅ Código
├── train.py                 ✅ Script de treino
├── app.py                   ✅ Interface
├── CONCEITOS.md             ✅ Documentação
├── README.md                ✅ Documentação
├── GETTING_STARTED.md       ✅ Documentação
├── claude.md                ✅ Documentação
├── .gitignore               ✅ Este arquivo!
├── requirements.txt         ✅ Dependências
├── LICENSE                  ✅ (adicione depois)
└── .github/
    └── workflows/           ✅ (CI/CD, opcional)
```

### ❌ **NÃO SUBIR (será ignorado automaticamente):**

```
data/
├── shakespeare.txt          ❌ Dataset (5.17 MB)
│   → Será baixado automaticamente

model/
├── gpt_mini_best.pt         ❌ Modelo treinado (12.5 MB)
├── gpt_mini_epoch_*.pt      ❌ Checkpoints
├── tokenizer.pkl            ❌ Tokenizador
├── gpt_mini_history.json    ❌ Histórico
└── gpt_mini_loss.png        ❌ Gráficos

Outros:
├── venv/                    ❌ Virtual environment
├── __pycache__/             ❌ Cache Python
├── .DS_Store                ❌ Arquivo macOS
├── ._*                      ❌ Arquivos macOS
└── *.pyc                    ❌ Bytecode Python
```

---

## 🔧 Passo a Passo: Subir para GitHub

### **1. Criar Repositório no GitHub**

```bash
# Acesse: https://github.com/new
# Preencha:
# - Repository name: gpt-mini-shakespeare
# - Description: Modelo Transformer simplificado para aprender LLMs
# - Public (para que todos vejam)
# - NÃO inicialize com README (já temos!)
```

### **2. Preparar Repositório Local**

```bash
cd /Volumes/Extreme\ SSD/IA/repos/my_model

# Inicializar git
git init

# Adicionar todos os arquivos (exceto os do .gitignore)
git add .

# Verificar o que será commitado
git status
# Deve listar: CONCEITOS.md, README.md, train.py, etc.
# NÃO deve listar: data/shakespeare.txt, model/*.pt, venv/
```

### **3. Fazer Primeiro Commit**

```bash
git config user.name "Seu Nome"
git config user.email "seu.email@example.com"

git commit -m "Initial commit: GPT Mini modelo Transformer com Shakespeare"
```

### **4. Conectar ao GitHub**

```bash
# Substituir USER e REPO pelos seus valores
git remote add origin https://github.com/USER/gpt-mini-shakespeare.git

# Renomear branch para main (padrão GitHub)
git branch -M main

# Fazer push (enviar para GitHub)
git push -u origin main
```

**Exemplo completo:**
```bash
git remote add origin https://github.com/sergiosantos/gpt-mini-shakespeare.git
git branch -M main
git push -u origin main
```

---

## 📋 Checklist Antes de Fazer Push

```
☐ .gitignore criado corretamente
☐ Nenhum arquivo ._* será enviado
☐ data/shakespeare.txt NÃO será enviado
☐ model/*.pt NÃO será enviado
☐ venv/ NÃO será enviado
☐ __pycache__/ NÃO será enviado

Verificar com:
git status --ignored
```

---

## 📖 README.md para GitHub

Seu README.md já está excelente! Ele deve ter:

```markdown
# GPT Mini: Transformers com Shakespeare

- Descrição clara ✅
- Quick Start ✅
- Estrutura do projeto ✅
- Como usar ✅
- Troubleshooting ✅
```

---

## 🔒 Adicionar LICENSE (Recomendado)

```bash
# Criar arquivo LICENSE
touch LICENSE
```

**Conteúdo simples (MIT License):**
```
MIT License

Copyright (c) 2026 [Seu Nome]

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

[continua...]
```

---

## 📊 O que vai aparecer no GitHub

Após fazer push, seu repositório terá:

```
📦 gpt-mini-shakespeare

📄 README.md              ← Página inicial
📄 CONCEITOS.md           ← Documentação didática (link no README)
📄 GETTING_STARTED.md     ← Guia de ação
📄 claude.md              ← Histórico técnico

📁 model/                 ← Código-fonte
   ├── transformer.py
   ├── gpt_mini.py
   ├── utils.py
   └── __init__.py

📄 train.py               ← Script de treinamento
📄 app.py                 ← Interface Streamlit
📄 requirements.txt       ← Dependências

📄 .gitignore             ← Configuração
📄 LICENSE                ← Licença (adicione!)
```

---

## 💻 Atualizações Futuras

Depois que subir, para fazer atualizações:

```bash
# Fazer mudanças no código...

# Adicionar mudanças
git add .

# Commit
git commit -m "Descrição das mudanças"

# Push para GitHub
git push
```

**Exemplo:**
```bash
git add model/transformer.py
git commit -m "Melhorar documentação de TransformerBlock"
git push
```

---

## 🎯 Como Compartilhar

Após subir para GitHub:

1. **Copiar URL do repositório:**
   ```
   https://github.com/SEU_USER/gpt-mini-shakespeare
   ```

2. **Compartilhar com:**
   - Colegas
   - Comunidades (Reddit, Discord, etc)
   - LinkedIn
   - Portfolio pessoal

3. **Pessoas podem:**
   - Ver o código
   - Clonar e usar localmente
   - Dar estrelas ⭐
   - Fazer fork e contribuir

---

## 📌 Comandos Git Úteis

```bash
# Ver status
git status

# Ver commits
git log --oneline

# Ver diferenças
git diff

# Desfazer último commit (local apenas)
git reset --soft HEAD~1

# Clonar repositório (para outros)
git clone https://github.com/SEU_USER/gpt-mini-shakespeare.git
```

---

## ✨ Recursos Adicionais (Opcional)

### **.github/workflows/ (CI/CD)**
Automatizar testes quando faz push:
```yaml
# Arquivo: .github/workflows/test.yml
name: Tests
on: [push]
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - name: Test transformer
        run: python3 -m model.transformer
```

### **CHANGELOG.md**
Rastrear mudanças versão por versão.

### **CONTRIBUTING.md**
Guiar pessoas que querem contribuir.

---

## 🚀 Próximos Passos

1. ✅ Criar `.gitignore` (já fiz!)
2. ⏳ Escolher nome: `gpt-mini-shakespeare`
3. ⏳ Criar repositório no GitHub
4. ⏳ Fazer `git init` e `git push`
5. ⏳ Compartilhar link!

---

## 📞 Dúvidas Comuns

**P: Preciso deletar data/shakespeare.txt?**  
R: Não! Deixa lá. O `.gitignore` faz git ignorar. Seu script ainda baixa automaticamente.

**P: Posso subir o modelo treinado?**  
R: Pode, mas não é recomendado (GitHub cobrava por >1GB). Melhor deixar script de treino.

**P: E se eu quiser compartilhar o modelo treinado?**  
R: Use Hugging Face Model Hub ou Google Drive (gratuito, sem limite).

**P: Preciso do arquivo LICENSE?**  
R: Não é obrigatório, mas recomendado (deixa claro as permissões).

---

**Última atualização:** 10 de Maio de 2026
