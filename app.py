"""
Egyptian Arabic to English Translator - Gradio Web App
Run this to launch an interactive web interface
"""

import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import gradio as gr

# Configuration
MODEL_PATH = "./saved_model"
MAX_LEN = 128
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load model and tokenizer
print("Loading model...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_PATH)
model.to(DEVICE)
model.eval()
print("Model loaded!")

# Load BLEU score
try:
    with open("bleu_score.txt", "r") as f:
        BLEU_SCORE = f.read().strip()
except:
    BLEU_SCORE = "Not available"

def translate_text(egyptian_text):
    """Translate function for Gradio"""
    if not egyptian_text or not egyptian_text.strip():
        return "⚠️ Please enter Egyptian Arabic text."
    
    # Tokenize
    input_ids = tokenizer.encode(
        egyptian_text,
        return_tensors='pt',
        truncation=True,
        max_length=MAX_LEN
    ).to(DEVICE)
    
    # Generate
    with torch.no_grad():
        output_ids = model.generate(
            input_ids,
            max_length=MAX_LEN,
            num_beams=4,
            temperature=0.7,
            early_stopping=True
        )
    
    # Decode
    translated = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    return translated

# Example sentences
examples = [
    ["ازيك عامل ايه؟"],
    ["انت رايح فين النهاردة؟"],
    ["انا عاوز اروح الجامعة"],
    ["الاكل ده كان تحفة"],
    ["بكام القميص ده؟"],
    ["انا تعبان شوية"],
    ["متزعلش كله هيكون تمام"],
]

# Custom CSS
custom_css = """
.gradio-container {
    max-width: 800px;
    margin: auto;
}
h1 {
    text-align: center;
    color: #2c3e50;
}
"""

# Create interface
with gr.Blocks(css=custom_css, title="Egyptian Arabic Translator") as demo:
    gr.Markdown(f"""
    # 🇪🇬 ➡️ 🇺🇸 Egyptian Arabic to English Translator
    
    This model translates **Egyptian Arabic dialect** (Egyptian Arabic) to English using a fine-tuned Transformer model.
    
    ### 📊 Model Performance
    - **BLEU Score**: `{BLEU_SCORE}`
    - **Architecture**: Helsinki-NLP/opus-mt-ar-en
    - **Training Data**: ~50,000 Egyptian Arabic-English pairs
    
    ### 💡 Try these examples below ⬇️
    """)
    
    with gr.Row():
        with gr.Column():
            input_text = gr.Textbox(
                label="Egyptian Arabic",
                placeholder="اكتب بالعامية المصرية هنا...",
                lines=4
            )
            translate_btn = gr.Button("Translate ➡️", variant="primary")
        
        with gr.Column():
            output_text = gr.Textbox(
                label="English Translation",
                placeholder="Translation will appear here...",
                lines=4
            )
    
    translate_btn.click(
        fn=translate_text,
        inputs=input_text,
        outputs=output_text
    )
    
    gr.Markdown("### 📝 Example Translations")
    gr.Examples(
        examples=examples,
        inputs=input_text,
        outputs=output_text,
        fn=translate_text,
        cache_examples=True
    )
    
    gr.Markdown("""
    ---
    ### 📚 About
    - **Dataset**: [Egyptian to English](https://huggingface.co/datasets/Abdalrahmankamel/Egyption_2_English)
    - **Base Model**: Helsinki-NLP OPUS-MT
    - **Code**: [GitHub Repository](https://github.com/YousefBotros/egyptian-arabic-translator)
    """)

if __name__ == "__main__":
    demo.launch(share=True)
