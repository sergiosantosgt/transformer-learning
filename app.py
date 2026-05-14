"""
app.py - Interface Streamlit para o modelo GPT Mini

Esta aplicação permite:
1. Fazer perguntas ao modelo em tempo real
2. Controlar a criatividade (temperatura)
3. Visualizar a atenção do modelo
4. Ver distribuição de probabilidades

Uso:
    streamlit run app.py
    
    Abrirá em: http://localhost:8501
"""

import streamlit as st
import torch
import numpy as np
import matplotlib.pyplot as plt
import os
from model import GPTMini, CharacterTokenizer


@st.cache_resource
def load_model_and_tokenizer():
    """
    Carregar modelo e tokenizador (cache para não recarregar sempre).
    """
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    checkpoint_path = "model/gpt_mini_best.pt"
    tokenizer_path = "model/tokenizer.pkl"
    
    # Verificar se arquivos existem
    if not os.path.exists(checkpoint_path):
        st.error(f"❌ Modelo não encontrado: {checkpoint_path}")
        st.info("Primeiro execute: python3 train.py")
        st.stop()
    
    if not os.path.exists(tokenizer_path):
        st.error(f"❌ Tokenizador não encontrado: {tokenizer_path}")
        st.info("Primeiro execute: python3 train.py")
        st.stop()
    
    # Carregar tokenizador
    tokenizer = CharacterTokenizer.load(tokenizer_path)
    
    # Carregar modelo
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
    
    return model, tokenizer, device


def main():
    """Função principal da aplicação."""
    
    # ===== Header =====
    st.set_page_config(
        page_title="GPT Mini - Shakespeare",
        page_icon="🧠",
        layout="wide"
    )
    
    st.title("🧠 GPT Mini: Aprendendo Transformers com Shakespeare")
    st.write("""
    Um modelo Transformer simplificado treinado com as obras de Shakespeare.
    
    **Como funciona:**
    1. Digita um prompt (ex: "To be or")
    2. Modelo prediz o próximo token (letra/caractere)
    3. Usa a predição para prever o próximo
    4. Repete até atingir comprimento máximo
    """)
    
    # ===== Sidebar =====
    st.sidebar.title("⚙️ Configurações")
    
    st.sidebar.markdown("### Controle de Geração")
    temperature = st.sidebar.slider(
        "🌡️ Temperatura (Criatividade)",
        min_value=0.1,
        max_value=2.0,
        value=0.8,
        step=0.1,
        help="""
        - 0.1-0.5: Determinístico, segue padrões
        - 0.7-0.9: Recomendado (balanceado)
        - 1.0: Balanceado
        - 1.5-2.0: Muito criativo, mais erros
        """
    )
    
    max_length = st.sidebar.slider(
        "📏 Comprimento Máximo",
        min_value=10,
        max_value=500,
        value=100,
        step=10
    )
    
    st.sidebar.markdown("### Técnicas Avançadas")
    
    repetition_penalty = st.sidebar.slider(
        "⛔ Penalidade de Repetição",
        min_value=1.0,
        max_value=3.0,
        value=1.2,
        step=0.1,
        help="""
        Evita loops de tokens repetidos:
        - 1.0: Sem penalidade (pode gerar 'oooooo')
        - 1.2: Recomendado ✓
        - 1.5-2.0: Forte (pode limitar gerações naturais)
        """
    )
    
    top_p = st.sidebar.slider(
        "🎯 Top-P (Nucleus Sampling)",
        min_value=0.5,
        max_value=1.0,
        value=0.95,
        step=0.05,
        help="""
        Manter tokens que acumulam p% da probabilidade:
        - 0.5-0.8: Mais restritivo, menos criativo
        - 0.9-0.95: Recomendado ✓
        - 1.0: Sem filtro
        """
    )
    
    st.sidebar.markdown("### Sobre o Modelo")
    st.sidebar.write(f"""
    **Arquitetura:**
    - 3 camadas Transformer
    - 8 cabeças de atenção
    - 256 dimensões
    - ~2.4M parâmetros
    
    **Dataset:**
    - Shakespeare completo
    - ~5.3M tokens
    - Character-level tokenization
    - 95 caracteres únicos
    
    **Treinamento:**
    - Early stopping (parou em época 49)
    - Val loss: 0.021
    - Dropout: 0.5
    """)
    
    # ===== Carregar modelo =====
    st.write("### 📚 Carregando Modelo...")
    with st.spinner("Carregando modelo e tokenizador..."):
        model, tokenizer, device = load_model_and_tokenizer()
    
    st.success("✅ Modelo carregado!")
    
    # ===== Input =====
    st.write("### 📝 Entrada")
    prompt = st.text_input(
        "Digite um prompt:",
        value="To be or not to",
        placeholder="Digite aqui...",
        help="Comece com algo como 'To be', 'The', etc"
    )
    
    # ===== Botão de geração =====
    col1, col2 = st.columns(2)
    
    with col1:
        generate_button = st.button("🚀 Gerar Texto", key="generate")
    
    with col2:
        example_button = st.button("📖 Exemplo Clássico", key="example")
    
    if example_button:
        prompt = "To be or"
        st.text_input(
            "Digite um prompt:",
            value=prompt,
            disabled=True
        )
    
    # ===== Geração =====
    if generate_button and prompt:
        st.write("### 📤 Saída Gerada")
        
        with st.spinner("Gerando..."):
            try:
                # Tokenizar prompt
                prompt_ids = tokenizer.encode(prompt)
                
                # Gerar
                with torch.no_grad():
                    generated_ids = model.generate(
                        prompt_ids,
                        max_length=max_length,
                        temperature=temperature,
                        repetition_penalty=repetition_penalty,
                        top_p=top_p,
                        device=device
                    )
                
                # Decodificar
                generated_text = tokenizer.decode(generated_ids)
                
                # Mostrar
                st.markdown("#### Texto Gerado:")
                st.info(generated_text)
                
                # ===== Análise de Repetições =====
                def detect_repetitions(text, min_length=3):
                    """Detectar sequências repetidas no texto."""
                    repetitions = []
                    for length in range(min_length, min(len(text) // 2, 10)):
                        for i in range(len(text) - length):
                            pattern = text[i:i+length]
                            count = 0
                            idx = i + length
                            while idx + length <= len(text) and text[idx:idx+length] == pattern:
                                count += 1
                                idx += length
                            if count > 0:
                                repetitions.append({
                                    'pattern': pattern,
                                    'position': i,
                                    'repetitions': count,
                                    'length': length
                                })
                    return repetitions
                
                reps = detect_repetitions(generated_text)
                if reps:
                    max_rep = max(r['repetitions'] for r in reps)
                    st.warning(f"⚠️ Detectado padrão repetitivo (máx {max_rep}x: '{reps[0]['pattern']}')")
                else:
                    st.success("✅ Sem padrões repetitivos detectados")
                
                # Estatísticas
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric(
                        "📊 Comprimento",
                        f"{len(generated_ids)} caracteres"
                    )
                
                with col2:
                    st.metric(
                        "🌡️ Temperatura",
                        f"{temperature:.1f}"
                    )
                
                with col3:
                    st.metric(
                        "⛔ Penalidade",
                        f"{repetition_penalty:.1f}"
                    )
                
                with col4:
                    st.metric(
                        "⚡ Device",
                        device.upper()
                    )
                
                # ===== Visualizações =====
                st.write("### 📊 Análise")
                
                # Histograma de caracteres
                chars = list(generated_text)
                char_counts = {}
                for c in chars:
                    char_counts[c] = char_counts.get(c, 0) + 1
                
                # Top 10 caracteres
                sorted_chars = sorted(
                    char_counts.items(),
                    key=lambda x: x[1],
                    reverse=True
                )[:10]
                
                fig, ax = plt.subplots(figsize=(10, 4))
                chars_list = [c[0] if c[0] != ' ' else 'SPACE' for c in sorted_chars]
                counts_list = [c[1] for c in sorted_chars]
                
                ax.bar(chars_list, counts_list, color='steelblue')
                ax.set_ylabel('Frequência')
                ax.set_title('Top 10 Caracteres Mais Frequentes')
                plt.tight_layout()
                
                st.pyplot(fig)
                
                # ===== Informações do Modelo =====
                st.write("### 🧠 Informações do Modelo")
                
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric(
                        "Parâmetros",
                        f"{model.get_num_parameters():,}"
                    )
                
                with col2:
                    st.metric(
                        "Tamanho",
                        f"{model.get_model_size_mb():.2f} MB"
                    )
                
                with col3:
                    st.metric(
                        "Vocab Size",
                        f"{tokenizer.vocab_size}"
                    )
                
                with col4:
                    st.metric(
                        "Camadas",
                        "4"
                    )
                
            except Exception as e:
                st.error(f"❌ Erro ao gerar: {str(e)}")
    
    # ===== Informações educacionais =====
    st.write("### 📚 Aprender Mais")
    
    with st.expander("Como funciona o modelo?"):
        st.write("""
        **Transformer:**
        O modelo usa uma arquitetura chamada Transformer que foi
        revolucionária quando foi introduzida em 2017.
        
        **Componentes principais:**
        1. **Embeddings**: Convertem tokens em vetores numéricos
        2. **Positional Encoding**: Adiciona informação de posição
        3. **Multi-Head Attention**: Aprende relações entre tokens
        4. **Feed-Forward**: Processa informações
        
        **Por que funciona:**
        - Atenção permite que o modelo veja qualquer token instantaneamente
        - Não sequencial = pode processar em paralelo
        - Gradientes fluem melhor que em RNNs (problemas históricos)
        
        Ver CONCEITOS.md para explicação detalhada com código!
        """)
    
    with st.expander("Experimentos Sugeridos"):
        st.write("""
        1. **Temperatura Baixa (0.1-0.5)**
           - Resultado: Repetitivo, previsível
           - Por quê: Modelo sempre escolhe opção mais provável
        
        2. **Temperatura Alta (1.5-2.0)**
           - Resultado: Criativo, às vezes errado
           - Por quê: Modelo escolhe aleatoriamente
        
        3. **Prompts Diferentes**
           - "The" → Prosa
           - "Thou" → Poesia Shakespeariana
           - "A" → Aberto, qualquer coisa
        
        4. **Comprimento Máximo**
           - Curto (10-30): Qualidade melhor
           - Longo (200+): Pode sair do contexto
        """)
    
    with st.expander("Recursos para Aprender"):
        st.write("""
        - **CONCEITOS.md**: Guia completo dos Transformers (9k linhas!)
        - **claude.md**: Histórico da implementação
        - **Código comentado**: model/transformer.py, model/gpt_mini.py
        
        **Referências:**
        - [Attention is All You Need](https://arxiv.org/abs/1706.03762)
        - [3Blue1Brown - Attention](https://www.youtube.com/watch?v=eMlx5aFJsrQ)
        """)
    
    # ===== Footer =====
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666;">
    <p>🧠 GPT Mini | Aprendendo Transformers com Shakespeare</p>
    <p><small>Modelo treinado em Python + PyTorch | Interface com Streamlit</small></p>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
