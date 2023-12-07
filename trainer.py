import torch
from torch.utils.data import TensorDataset, Dataset
from transformers import AutoTokenizer, LongT5ForConditionalGeneration, Trainer, TrainingArguments
import pandas as pd
import numpy as np
import sys

NUM_EPOCHS = 1
INPUT_MAX_LENGTH = 4096 # use 2048 on a T4
OUTPUT_MAX_LENGTH = 512
BATCH_SIZE = 1

class CustomDataset(Dataset):
    def __init__(self, tokenizer, df, max_input_length, max_output_length):
        self.tokenizer = tokenizer
        self.inputs = df['input']
        self.outputs = df['output']
        self.max_input_length = max_input_length
        self.max_output_length = max_output_length

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        input_text = str(self.inputs[idx])
        output_text = str(self.outputs[idx])
        inputs = self.tokenizer.encode_plus(
            input_text, 
            max_length=self.max_input_length, 
            padding='max_length', 
            truncation=True,
            return_tensors="pt"
        )
        outputs = self.tokenizer.encode_plus(
            output_text, 
            max_length=self.max_output_length, 
            padding='max_length', 
            truncation=True,
            return_tensors="pt"
        )
        return {
            'input_ids': inputs['input_ids'].flatten(),
            'attention_mask': inputs['attention_mask'].flatten(),
            'labels': outputs['input_ids'].flatten()
        }

def finetune(train_csv_name, val_csv, test_csv):

    model_name = "google/long-t5-local-base"
    print("Try loading Tokenizer from HuggingFace")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    print("Try loading model from HuggingFace")
    model = LongT5ForConditionalGeneration.from_pretrained(model_name)

    print("Reading the CSVs")
    df = pd.read_csv(train_csv_name)
    val_df = pd.read_csv(val_csv)
    test_df = pd.read_csv(test_csv)
    print("Creating the datasets")
    train_dataset = CustomDataset(tokenizer, df, INPUT_MAX_LENGTH, OUTPUT_MAX_LENGTH)
    val_dataset = CustomDataset(tokenizer, val_df, INPUT_MAX_LENGTH, OUTPUT_MAX_LENGTH)
    test_dataset = CustomDataset(tokenizer, test_df, INPUT_MAX_LENGTH, OUTPUT_MAX_LENGTH)

    # Tokenizeand create dataset. Old, not using TensorDataset now
    print("Tokenize Inputs")
    inputs = tokenizer(df['input'].tolist(), max_length=INPUT_MAX_LENGTH, padding='max_length', truncation=True, return_tensors="pt")
    print("Tokenize Outputs")
    outputs = tokenizer(df['output'].tolist(), max_length=OUTPUT_MAX_LENGTH, padding='max_length', truncation=True, return_tensors="pt")

    # dataset = TensorDataset(inputs['input_ids'], inputs['attention_mask'], outputs['input_ids'])

    print("Training the model")
    # model.config.use_cache = False

    print(f"Train for {NUM_EPOCHS} epochs")
    training_args = TrainingArguments(
        output_dir='./results',
        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        # gradient_checkpointing=True,
        learning_rate=2.5e-5,
        warmup_steps=500,
        weight_decay=0.01,
        fp16=True,
        logging_dir='./logs',
        logging_steps=10,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset
    )
    
    trainer.train()

    print("You did it, saving the finetuned model.")
    model.save_pretrained('T5_finetuned')
    print("Save the tokenizer")
    tokenizer.save_pretrained('T5_finetuned')

    print("Performing validation")
    trainer.evaluate(eval_dataset=val_dataset)

    print("Testing the model")
    res = trainer.predict(test_dataset)
    print(res.metrics)

    print("Save the predictions")
    np.save("test_predictions.npy", res.predictions)

    print("Save the metrics")
    with open("test_metrics.txt", "w") as file:
        for key, value in res.metrics.items():
            file.write(f"{key}: {value}\n")

    print("View an arbitrary input and output pair")
    input_text = test_df.iloc[5]['input'] # magic arbitrary number for index.

    # Tokenize and generate output
    inputs = tokenizer(input_text, return_tensors="pt", padding=True, truncation=True, max_length=INPUT_MAX_LENGTH)
    input_ids = inputs["input_ids"].to(model.device)
    attention_mask = inputs["attention_mask"].to(model.device)

    # Generate output
    model.eval()
    with torch.no_grad():
        output_ids = model.generate(input_ids=input_ids, attention_mask=attention_mask, max_length=OUTPUT_MAX_LENGTH)[0]

    # Decode output
    generated_text = tokenizer.decode(output_ids, skip_special_tokens=True)
    print("Input:", input_text)
    print("Generated Output:", generated_text)

if __name__ == "__main__":
    if len(sys.argv) > 3: # Don't forget to change this if you change the inputs
        print("Good job dummy, starting the finetuning script")
        train_csv = sys.argv[1]
        
        val_csv = sys.argv[2]
        test_csv = sys.argv[3]
        finetune(train_csv, val_csv, test_csv)
        
        # finetune(train_csv)
    else:
        print("Please provide a filename for training data.")
