"""
Teste da penalização de repetição.
"""

import torch
from model import GPTMini, CharacterTokenizer

print("Carregando modelo...")
device = 'cpu'

checkpoint = torch.load("model/gpt_mini_best.pt", map_location=device)
tokenizer = CharacterTokenizer.load("model/tokenizer.pkl")

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

print("✅ Carregado\n")

# Testar com diferentes parâmetros
prompts = ["To be", "The king", "my name is Sergio"]

print("=" * 70)
print("TESTE: Penalização de Repetição")
print("=" * 70)

for prompt in prompts:
    print(f"\n📝 Prompt: '{prompt}'")
    prompt_ids = tokenizer.encode(prompt)
    
    # Sem penalização
    generated_no_penalty = model.generate(
        prompt_ids,
        max_length=100,
        temperature=1.0,
        repetition_penalty=1.0,
        device=device
    )
    text_no_penalty = tokenizer.decode(generated_no_penalty)
    
    # Com penalização 1.2
    generated_penalty12 = model.generate(
        prompt_ids,
        max_length=100,
        temperature=1.0,
        repetition_penalty=1.2,
        device=device
    )
    text_penalty12 = tokenizer.decode(generated_penalty12)
    
    # Com penalização 1.5
    generated_penalty15 = model.generate(
        prompt_ids,
        max_length=100,
        temperature=1.0,
        repetition_penalty=1.5,
        device=device
    )
    text_penalty15 = tokenizer.decode(generated_penalty15)
    
    # Com penalização + top_p
    generated_best = model.generate(
        prompt_ids,
        max_length=100,
        temperature=0.9,
        repetition_penalty=1.3,
        top_p=0.95,
        device=device
    )
    text_best = tokenizer.decode(generated_best)
    
    print(f"\n  Sem penalização:")
    print(f"    '{text_no_penalty}'")
    
    print(f"\n  Penalização 1.2:")
    print(f"    '{text_penalty12}'")
    
    print(f"\n  Penalização 1.5:")
    print(f"    '{text_penalty15}'")
    
    print(f"\n  Ótima (temp=0.9, pen=1.3, top_p=0.95):")
    print(f"    '{text_best}'")
    
    # Contar repetições
    def count_char_repetitions(text):
        """Contar sequências de caracteres idênticos."""
        if not text:
            return 0
        max_rep = 1
        current_rep = 1
        for i in range(1, len(text)):
            if text[i] == text[i-1]:
                current_rep += 1
                max_rep = max(max_rep, current_rep)
            else:
                current_rep = 1
        return max_rep
    
    print(f"\n  Estatísticas:")
    print(f"    Sem penalização - Seq. repetida max: {count_char_repetitions(text_no_penalty)}")
    print(f"    Com penalização 1.2 - Seq. repetida max: {count_char_repetitions(text_penalty12)}")
    print(f"    Com penalização 1.5 - Seq. repetida max: {count_char_repetitions(text_penalty15)}")
    print(f"    Ótima - Seq. repetida max: {count_char_repetitions(text_best)}")
