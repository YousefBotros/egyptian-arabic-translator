"""
Egyptian Arabic to English Translator - Inference Script
Loads saved model and translates text
"""

import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# Configuration
MODEL_PATH = "./saved_model"
MAX_LEN = 128
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load model and tokenizer
print(f"Loading model from {MODEL_PATH}...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_PATH)
model.to(DEVICE)
model.eval()
print("Model loaded successfully!")

def translate(egyptian_text, num_beams=4, temperature=0.7):
    """
    Translate Egyptian Arabic text to English
    
    Args:
        egyptian_text (str): Input text in Egyptian Arabic dialect
        num_beams (int): Beam search width (default: 4)
        temperature (float): Sampling temperature (default: 0.7)
    
    Returns:
        str: English translation
    """
    if not egyptian_text or not egyptian_text.strip():
        return "Please enter Egyptian Arabic text."
    
    # Tokenize input
    input_ids = tokenizer.encode(
        egyptian_text, 
        return_tensors='pt', 
        truncation=True, 
        max_length=MAX_LEN
    ).to(DEVICE)
    
    # Generate translation
    with torch.no_grad():
        output_ids = model.generate(
            input_ids,
            max_length=MAX_LEN,
            num_beams=num_beams,
            temperature=temperature,
            early_stopping=True,
            do_sample=False if num_beams > 1 else True
        )
    
    # Decode output
    translated = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    return translated

# CLI interface
if __name__ == "__main__":
    print("\n" + "="*50)
    print("Egyptian Arabic to English Translator")
    print("Type 'quit' to exit")
    print("="*50 + "\n")
    
    # Load BLEU score if exists
    try:
        with open("bleu_score.txt", "r") as f:
            bleu = f.read()
        print(f"Model BLEU Score: {bleu}\n")
    except:
        pass
    
    while True:
        text = input("Egyptian Arabic: ").strip()
        if text.lower() in ['quit', 'exit', 'q']:
            print("Goodbye!")
            break
        if text:
            result = translate(text)
            print(f"English: {result}\n")
