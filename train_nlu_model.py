import pandas as pd
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer
)
import numpy as np
from sklearn.metrics import f1_score, accuracy_score
import torch
import os
import csv

DATASET_CSV_PATH = "nlu_dataset.csv"
MODEL_OUTPUT_DIRECTORY = "./nlu_model"

def prepare_dataset_from_csv(csv_path):
    print(f"Loading and preparing dataset from {csv_path}...")
    if not os.path.exists(csv_path):
        print(f"ERROR: Dataset file not found at '{csv_path}'.")
        return None, None, None
    
    instructions = []
    intents_str = []
    try:
        with open(csv_path, 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            header = next(reader) # Skip header
            for i, row in enumerate(reader):
                if not row: # Skip empty rows
                    continue
                if len(row) < 2:
                    print(f"Warning: Skipping malformed row {i+2}: {row}")
                    continue
                instructions.append(row[0])
                intents_str.append(','.join(row[1:]))
        df = pd.DataFrame({'instruction': instructions, 'intent': intents_str})
    except Exception as e:
        print(f"ERROR reading CSV: {e}")
        return None, None, None
        
    df['intent'] = df['intent'].astype(str).str.strip()

    # Create a sorted, unique list of all possible single intents
    all_intents = sorted(list(set(intent.strip() for intents in df['intent'] for intent in intents.split(','))))
    
    id2label = {i: label for i, label in enumerate(all_intents)}
    label2id = {label: i for i, label in enumerate(all_intents)}
    
    num_labels = len(all_intents)
    print(f"Found {len(df)} total examples.")
    print(f"Identified {num_labels} unique intents.")

    # Create the multi-hot encoded 'labels' column
    labels = np.zeros((len(df), num_labels), dtype=int)
    for i, row in df.iterrows():
        current_intents = [intent.strip() for intent in row['intent'].split(',')]
        for intent in current_intents:
            if intent in label2id:
                labels[i, label2id[intent]] = 1
    
    df['labels'] = list(labels.astype(float)) # Use float for PyTorch tensors
    
    return Dataset.from_pandas(df), id2label, label2id

def compute_metrics(p):
    preds = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
    sigmoid_preds = 1 / (1 + np.exp(-preds))
    binary_preds = (sigmoid_preds > 0.5).astype(int)
    labels = p.label_ids
    f1 = f1_score(y_true=labels, y_pred=binary_preds, average='weighted', zero_division=0) # Computes F1-score for multi-label predictions
    return {'f1': f1}

# Fine-tunes a DistilBERT model for multi-label intent classification
def train_nlu_model(dataset, id2label, label2id, model_output_dir):
    train_test_split_dataset = dataset.train_test_split(test_size=0.2, seed=42)
    train_dataset = train_test_split_dataset['train']
    test_dataset = train_test_split_dataset['test']
    
    MODEL_NAME = 'distilbert-base-uncased'
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    def tokenize(batch):
        return tokenizer(batch['instruction'], padding='max_length', truncation=True, max_length=128)

    train_dataset = train_dataset.map(tokenize, batched=True)
    test_dataset = test_dataset.map(tokenize, batched=True)

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=len(id2label),
        id2label=id2label,
        label2id=label2id,
        problem_type="multi_label_classification"
    )

    training_args = TrainingArguments(
        output_dir=os.path.join(model_output_dir, "training_checkpoints"),
        num_train_epochs=4,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        learning_rate=3e-5,
        weight_decay=0.01,
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        push_to_hub=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics
    )

    print("\nStarting Multi-Label NLU Model Training...")
    trainer.train()
    print("Training finished.")
    
    print(f"\nSaving best model to {model_output_dir}...")
    trainer.save_model(model_output_dir)
    print(f"Model saved successfully to: {model_output_dir}")


full_dataset, id2label, label2id = prepare_dataset_from_csv(DATASET_CSV_PATH)

if full_dataset:
    train_nlu_model(full_dataset, id2label, label2id, MODEL_OUTPUT_DIRECTORY)