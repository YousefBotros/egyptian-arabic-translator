"""
Egyptian Arabic to English Translator - Training Script
Run this once to train and save the model
"""

import pandas as pd
import torch
from datasets import Dataset, load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    DataCollatorForSeq2Seq
)
from nltk.translate.bleu_score import corpus_bleu

# ============ CONFIGURATION ============
MODEL_CHECKPOINT = "Helsinki-NLP/opus-mt-ar-en"
BATCH_SIZE = 32
MAX_LEN = 128
NUM_EPOCHS = 3
LEARNING_RATE = 2e-5
SRC_COL = 'egyption_Text'
TGT_COL = 'english_Text'
OUTPUT_DIR = "./saved_model"

# ============ LOAD DATASET ============
print("Loading dataset...")
dataset_dict = load_dataset("Abdalrahmankamel/Egyption_2_English")
df_train = dataset_dict['train'].to_pandas()
df_test = dataset_dict['test'].to_pandas()

print(f"Train: {len(df_train)} samples, Test: {len(df_test)} samples")

# Convert to HuggingFace Dataset
train_dataset = Dataset.from_pandas(df_train[[SRC_COL, TGT_COL]])
test_dataset = Dataset.from_pandas(df_test[[SRC_COL, TGT_COL]])

# ============ TOKENIZER & MODEL ============
print("Loading tokenizer and model...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_CHECKPOINT)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_CHECKPOINT)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
print(f"Using device: {device}")

# ============ PREPROCESSING ============
def preprocess_function(examples):
    inputs = examples[SRC_COL]
    targets = examples[TGT_COL]
    model_inputs = tokenizer(inputs, max_length=MAX_LEN, truncation=True, padding=False)
    labels = tokenizer(targets, max_length=MAX_LEN, truncation=True, padding=False)
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

print("Tokenizing datasets...")
train_tokenized = train_dataset.map(
    preprocess_function, 
    batched=True, 
    remove_columns=train_dataset.column_names
)
test_tokenized = test_dataset.map(
    preprocess_function, 
    batched=True, 
    remove_columns=test_dataset.column_names
)

# ============ TRAINING ============
data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)

training_args = Seq2SeqTrainingArguments(
    output_dir="./checkpoints",
    eval_strategy="epoch",
    save_strategy="epoch",
    learning_rate=LEARNING_RATE,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    weight_decay=0.01,
    save_total_limit=2,
    num_train_epochs=NUM_EPOCHS,
    predict_with_generate=True,
    fp16=torch.cuda.is_available(),
    logging_dir="./logs",
    logging_steps=50,
    report_to="none",
)

trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=train_tokenized,
    eval_dataset=test_tokenized,
    tokenizer=tokenizer,
    data_collator=data_collator,
)

print("Starting training...")
trainer.train()

# ============ EVALUATION ============
print("Evaluating model...")
predictions = trainer.predict(test_tokenized)
decoded_preds = tokenizer.batch_decode(predictions.predictions, skip_special_tokens=True)
decoded_labels = tokenizer.batch_decode(predictions.label_ids, skip_special_tokens=True)

tokenized_preds = [pred.split() for pred in decoded_preds]
tokenized_labels = [[label.split()] for label in decoded_labels]
bleu_score = corpus_bleu(tokenized_labels, tokenized_preds)

print(f"\n🎯 BLEU Score: {bleu_score:.4f}")

# Save BLEU score
with open("bleu_score.txt", "w") as f:
    f.write(str(bleu_score))

# ============ SAVE MODEL ============
print(f"Saving model to {OUTPUT_DIR}...")
model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)
print("Done! Model saved.")

# Save as ONNX (optional - for deployment)
try:
    from optimum.onnxruntime import ORTModelForSeq2SeqLM
    from optimum.onnxruntime.configuration import DecoderOnnxConfig
    print("Exporting to ONNX...")
    ort_model = ORTModelForSeq2SeqLM.from_pretrained(OUTPUT_DIR, export=True)
    ort_model.save_pretrained("./onnx_model")
    print("ONNX model saved!")
except:
    print("ONNX export skipped (install optimum if needed)")
