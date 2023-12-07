import torch
from torch.utils.data import DataLoader, TensorDataset
from transformers import AutoTokenizer, LongT5ForConditionalGeneration
import pandas as pd
import sys

NUM_EPOCHS = 3
INPUT_MAX_LENGTH = 512
OUTPUT_MAX_LENGTH = 128
BATCH_SIZE = 16

def finetune(train_csv_name):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    model_name = "google/long-t5-local-base"
    print("Try loading Tokenizer from HuggingFace")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    print("Try loading model from HuggingFace")
    model = LongT5ForConditionalGeneration.from_pretrained(model_name)

    # parallelize
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs!")
        model = torch.nn.DataParallel(model)

    print(f"Moving model to {device}")
    model = model.to(device)
    print("Reading the CSV")
    df = pd.read_csv(train_csv_name)

    # Tokenize
    print("Tokenize Inputs")
    inputs = tokenizer(df['input'].tolist(), max_length=INPUT_MAX_LENGTH, padding='max_length', truncation=True, return_tensors="pt")
    print("Tokenize Outputs")
    outputs = tokenizer(df['output'].tolist(), max_length=OUTPUT_MAX_LENGTH, padding='max_length', truncation=True, return_tensors="pt")

    # Create a dataset
    print("Creating the dataset")
    dataset = TensorDataset(inputs['input_ids'], inputs['attention_mask'], outputs['input_ids'])

    # Create a DataLoader
    print("Create Data loader")
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    print("Create optimizer")
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)

    print("Training the model")
    model.train()
    print(f"Train for {NUM_EPOCHS} epochs")
    for epoch in range(NUM_EPOCHS):
        print(f"Running epoch {epoch+1}")
        for batch in loader:
            print("Assign values from the batch")
            input_ids, attention_mask, labels = batch
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            labels = labels.to(device)

            # Forward pass
            print("Do the forward pass")
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            if outputs.dim() == 0:
                outputs = torch.unsqueeze(outputs, 0)
            print("Record loss")
            loss = outputs.loss

            # Backward pass and optimization
            print("Do zero_grad")
            optimizer.zero_grad()
            print("Do the backwards pass")
            loss = loss.mean()
            loss.backward()
            print("Call step()")
            optimizer.step()

        print(f"Epoch {epoch+1}, Loss: {loss.item()}")
    
    print("You did it, saving the finetuned model.")
    model.save_pretrained('T5_finetuned')
    print("Save the tokenizer")
    tokenizer.save_pretrained('T5_finetuned')

    '''
    print("Now load and run the validation data and loop")
    val_df = pd.read_csv(val_csv_name)
    val_inputs = tokenizer(val_df['input'].tolist(), padding=True, truncation=True, max_length=INPUT_MAX_LENGTH, return_tensors="pt")
    val_outputs = tokenizer(val_df['output'].tolist(), padding=True, truncation=True, max_length=OUTPUT_MAX_LENGTH, return_tensors="pt")
    val_dataset = TensorDataset(val_inputs['input_ids'], val_inputs['attention_mask'], val_outputs['input_ids'])
    validation_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)

    model.eval()  # Set the model to evaluation mode
    val_total_loss = 0

    print("Running the validation loop")
    with torch.no_grad():
        for batch in validation_loader:
            input_ids, attention_mask, labels = batch
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            labels = labels.to(device)

            # Forward pass
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss

            val_total_loss += loss.item()

    val_avg_loss = val_total_loss / len(validation_loader)
    print(f'Average validation loss: {val_avg_loss}')

    print("Now load and run the TEST data and loop")
    test_df = pd.read_csv(test_csv_name)
    test_inputs = tokenizer(test_df['input'].tolist(), padding=True, truncation=True, max_length=INPUT_MAX_LENGTH, return_tensors="pt")
    test_outputs = tokenizer(test_df['output'].tolist(), padding=True, truncation=True, max_length=OUTPUT_MAX_LENGTH, return_tensors="pt")
    test_dataset = TensorDataset(test_inputs['input_ids'], test_inputs['attention_mask'], test_outputs['input_ids'])
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)

    test_total_loss = 0

    print("Running the test eval loop")
    with torch.no_grad():
        for batch in test_loader:
            input_ids, attention_mask, labels = batch
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            labels = labels.to(device)

            # Forward pass
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss

            test_total_loss += loss.item()

    test_avg_loss = test_total_loss / len(test_loader)
    print(f'Average test loss: {test_avg_loss}')
    '''

if __name__ == "__main__":
    if len(sys.argv) > 1:
        print("Good job dummy, starting the finetuning script")
        train_csv = sys.argv[1]
        '''
        val_csv = sys.argv[2]
        test_csv = sys.argv[3]
        finetune(train_csv, val_csv, test_csv)
        '''
        finetune(train_csv)
    else:
        print("Please provide a filename for training data.")
