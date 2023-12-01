import torch
from torch.utils.data import DataLoader, TensorDataset
import torch.nn as nn
import torch.optim as optim
import transformers
from transformers import LlamaForCausalLM, LlamaTokenizer
from transformers import AutoTokenizer, AutoModelForCausalLM
import pandas as pd
import sys

NUM_EPOCHS = 3

def finetune(csv_name):
    with open('hf_token.txt', 'r') as file:
        huggingface_token = file.read().strip()
    
    model_dir = "./Llama-2-7b-hf"
    # print(f"loading model from {model_dir}")
    # model = LlamaForCausalLM.from_pretrained(model_dir)
    # tokenizer = LlamaTokenizer.from_pretrained(model_dir)

    # This was how HF said to use the model but I also git cloned the weight directly into GCP so use above.
    # Above method seems to be broken.
    print("Try loading tokenizer from HuggingFace")
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf", token=huggingface_token)
    print("Try loading model from HuggingFace")
    model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf", token=huggingface_token)
    
    print("Reading the CSV")
    df = pd.read_csv(csv_name)

    # Tokenize
    print("Tokenize the inputs")
    inputs = tokenizer(df['input'].tolist(), max_length=512, padding='max_length', truncation=True, return_tensors="pt")
    print("Tokenize the outputs")
    outputs = tokenizer(df['output'].tolist(), max_length=128, padding='max_length', truncation=True, return_tensors="pt")

    # Create a dataset
    print("Creating the dataset")
    dataset = TensorDataset(inputs['input_ids'], inputs['attention_mask'], outputs['input_ids'])

    # Create a DataLoader
    print("Create data loader")
    loader = DataLoader(dataset, batch_size=32, shuffle=True)
    print("Create optimizer")
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)

    print("Training the model")
    model.train()
    print(f"Train for {NUM_EPOCHS} epochs")
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
    model.save_pretrained('Llama2_finetuned')
    tokenizer.save_pretrained('Llama2_finetuned')

if __name__ == "__main__":
    if len(sys.argv) > 1:
        print("Good job dummy, starting the finetuning script")
        filename = sys.argv[1]
        finetune(filename)
    else:
        print("Please provide a filename for training data.")
