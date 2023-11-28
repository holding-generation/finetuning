import torch
from torch.utils.data import DataLoader, TensorDataset
import torch.nn as nn
import torch.optim as optim
import transformers
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import pandas as pd
import sys

NUM_EPOCHS = 5

def finetune(csv_name):
    tokenizer = AutoTokenizer.from_pretrained('LLaMA2')  # Replace with actual model name

    df = pd.read_csv('test_set_for_jesse.csv')

    # Tokenize
    inputs = tokenizer(df['input'].tolist(), max_length=512, padding='max_length', truncation=True, return_tensors="pt")
    outputs = tokenizer(df['output'].tolist(), max_length=128, padding='max_length', truncation=True, return_tensors="pt")

    # Create a dataset
    dataset = TensorDataset(inputs['input_ids'], inputs['attention_mask'], outputs['input_ids'])

    # Create a DataLoader
    loader = DataLoader(dataset, batch_size=32, shuffle=True)

    model = AutoModelForSeq2SeqLM.from_pretrained('LLaMA-2-model-name')  # Replace with the appropriate model
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)

    model.train()

    for epoch in range(NUM_EPOCHS):
        for batch in loader:
            input_ids, attention_mask, labels = batch

            # Forward pass
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print(f"Epoch {epoch+1}, Loss: {loss.item()}")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        filename = sys.argv[1]
        finetune(filename)
    else:
        print("Please provide a filename for training data.")
