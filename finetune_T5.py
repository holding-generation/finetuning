import torch
from torch.utils.data import DataLoader, TensorDataset
import torch.nn as nn
import torch.optim as optim
import transformers
from transformers import AutoTokenizer, LongT5ForConditionalGeneration
import pandas as pd
import sys

NUM_EPOCHS = 5

def finetune(csv_name):
    model_name = "google/long-t5-local-base"
    print("Try loading model from HuggingFace")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = LongT5ForConditionalGeneration.from_pretrained(model_name)
    
    print("Reading the CSV")
    df = pd.read_csv(csv_name)

    # Tokenize
    inputs = tokenizer(df['input'].tolist(), max_length=512, padding='max_length', truncation=True, return_tensors="pt")
    outputs = tokenizer(df['output'].tolist(), max_length=128, padding='max_length', truncation=True, return_tensors="pt")

    # Create a dataset
    print("Creating the dataset")
    dataset = TensorDataset(inputs['input_ids'], inputs['attention_mask'], outputs['input_ids'])

    # Create a DataLoader
    loader = DataLoader(dataset, batch_size=32, shuffle=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)

    print("Training the model")
    model.train()

    for epoch in range(NUM_EPOCHS):
        print(f"Running epoch {epoch+1}")
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
    
    print("You did it, saving the finetuned model.")
    model.save_pretrained('T5_finetuned')
    tokenizer.save_pretrained('T5_finetuned')

if __name__ == "__main__":
    if len(sys.argv) > 1:
        print("Good job dummy, starting the finetuning script")
        filename = sys.argv[1]
        finetune(filename)
    else:
        print("Please provide a filename for training data.")
