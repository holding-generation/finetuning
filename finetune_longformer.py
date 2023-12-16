import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import torch
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, LEDForConditionalGeneration, Trainer, TrainingArguments
from rouge import Rouge
import pandas as pd
import numpy as np
import sys
import csv

NUM_EPOCHS = 4
INPUT_MAX_LENGTH = 4096 # use 2048 on a T4
OUTPUT_MAX_LENGTH = 128
BATCH_SIZE = 1
MODEL_NAME = 'longformer'

# After debugging with chatGPT, I learned that I needed to use a custom dataset.
# Using chatGPT along with the instructions from this website, 
# https://towardsdatascience.com/fine-tuning-a-t5-transformer-for-any-summarization-task-82334c64c81
# I came up with the following data object.
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

    model_name = "allenai/led-base-16384"
    print("Try loading Tokenizer from HuggingFace")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    print("Try loading model from HuggingFace")
    model = LEDForConditionalGeneration.from_pretrained(model_name)

    print("Reading the CSVs")
    df = pd.read_csv(train_csv_name)
    val_df = pd.read_csv(val_csv)
    test_df = pd.read_csv(test_csv)
    print("Creating the datasets")
    train_dataset = CustomDataset(tokenizer, df, INPUT_MAX_LENGTH, OUTPUT_MAX_LENGTH)
    val_dataset = CustomDataset(tokenizer, val_df, INPUT_MAX_LENGTH, OUTPUT_MAX_LENGTH)
    test_dataset = CustomDataset(tokenizer, test_df, INPUT_MAX_LENGTH, OUTPUT_MAX_LENGTH)

    # Tokenizeand create dataset. Old, not using TensorDataset now
    '''
    print("Tokenize Inputs")
    inputs = tokenizer(df['input'].tolist(), max_length=INPUT_MAX_LENGTH, padding='max_length', truncation=True, return_tensors="pt")
    print("Tokenize Outputs")
    outputs = tokenizer(df['output'].tolist(), max_length=OUTPUT_MAX_LENGTH, padding='max_length', truncation=True, return_tensors="pt")

    # dataset = TensorDataset(inputs['input_ids'], inputs['attention_mask'], outputs['input_ids'])
    '''
    print("Training the model")
    # model.config.use_cache = False

    print(f"Train for {NUM_EPOCHS} epochs")
    training_args = TrainingArguments(
        output_dir=f'./{MODEL_NAME}_results',
        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        # gradient_checkpointing=True, # This was causing an error for some reason
        save_steps=500,
        save_total_limit=5,
        gradient_accumulation_steps=4,
        warmup_steps=500,
        weight_decay=0.01,
        # fp16=True, # Using this causes trainer to not display loss
        logging_dir=f'./{MODEL_NAME}_logs',
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
    model.save_pretrained(f'{MODEL_NAME}_finetuned')
    print("Save the tokenizer")
    tokenizer.save_pretrained(f'{MODEL_NAME}_finetuned')

    print("Performing validation")
    trainer.evaluate(eval_dataset=val_dataset)

    # Manual testing loop
    # For some reason trainer.predict was causing an OOM error, so I had to 
    # implement training manually. I had chatGPT get me started with the 
    # structure of the loop and the function calls and then I modified, for
    # example by setting batch size to 1 and setting it to calculate ROUGE scores.
    print("Test the model manually with pytorch because GPU memory")
    data_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)
    model.eval()
    loss_function = CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)
    total_loss = 0
    num = 0
    references = []
    hypotheses = []
    prediction_csv_name = f'{MODEL_NAME}_predictions.csv'

    print("Start predicting")
    # Write the results
    with open(prediction_csv_name, 'w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow(['Input', 'Prediction', 'Reference', 'Loss'])
    
        with torch.no_grad():
            for i, batch in enumerate(data_loader):
                print(f"predict on the input number {num+1}")
                num += 1
                input_ids = batch['input_ids'].to(model.device)
                attention_mask = batch['attention_mask'].to(model.device)
                labels = batch['labels'].to(model.device)

                # Generate outputs
                outputs = model.generate(input_ids=input_ids, attention_mask=attention_mask, max_length=OUTPUT_MAX_LENGTH)
                prediction_logits = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels).logits

                # Calculate loss
                loss = loss_function(prediction_logits.view(-1, model.config.vocab_size), labels.view(-1))
                total_loss += loss.item()

                # Find all values to store
                input_text = df['input'].iloc[i]
                reference_text = df['output'].iloc[i]
                generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

                # Store to lists for ROUGE calculation
                hypotheses.append(generated_text)
                references.append(reference_text)

                # Write to file
                writer.writerow([input_text, generated_text, reference_text, loss.item()])

    average_loss = total_loss / len(data_loader)
    print(f"Average Loss: {average_loss}")

    # Get and write the rouge scores
    print("Calculate Rouge Scores")
    rouge = Rouge()
    rouge_scores = rouge.get_scores(hypotheses, references, avg=True)
    rouge_file = f"{MODEL_NAME}_rouge_scores.txt"

    print(f"Store Rouge scores in {rouge_file}")
    with open(rouge_file, 'w') as file:
        file.write(f"Average Loss: {average_loss}\n")
        for key, value in rouge_scores.items():
            file.write(f"{key}: {value}\n")
    
    
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
