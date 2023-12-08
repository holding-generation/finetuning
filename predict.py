import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import torch
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, LongT5ForConditionalGeneration, Trainer, TrainingArguments
from rouge import Rouge
import pandas as pd
import numpy as np
import sys
import csv

INPUT_MAX_LENGTH = 4096 # use 2048 on a T4
OUTPUT_MAX_LENGTH = 64
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

def my_predict(predict_csv_name, model_path):
    # Load the tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = LongT5ForConditionalGeneration.from_pretrained(model_path)

    # Load the dataset
    df = pd.read_csv(predict_csv_name)
    dataset = CustomDataset(tokenizer, df, INPUT_MAX_LENGTH, OUTPUT_MAX_LENGTH)

    # Try the trainer.predict method
    '''
    trainer = Trainer(model=model)
    prediction = trainer.predict(dataset)

    print("Save the predictions")
    np.save("test_predictions.npy", prediction.predictions)

    print("Save the metrics")
    with open("test_metrics.txt", "w") as file:
        for key, value in prediction.metrics.items():
            file.write(f"{key}: {value}\n")

    print("View an arbitrary input and output pair")
    input_text = df.iloc[4]['input'] # magic arbitrary number for index.
    fifth_prediction_id = prediction.predictions[4]

    fifth_prediction_text = tokenizer.decode(fifth_prediction_id, skip_special_tokens=True)
    print("Input:", input_text)
    print("Generated text:", fifth_prediction_text)

    '''
    # Try the pytorch model.eval() method
    # DataLoader
    data_loader = DataLoader(dataset, batch_size=BATCH_SIZE)

    # Prediction
    
    model.eval()
    loss_function = CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)
    total_loss = 0
    num = 0
    references = []
    hypotheses = []
    prediction_csv_name = 'T5_predictions.csv'

    print("Start predicting")
    # Write the results
    with open(prediction_csv_name, 'w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow(['Input', 'Prediction', 'Reference', 'Loss'])
    
        with torch.no_grad():
            for i, batch in enumerate(data_loader):
                print(f"predict on the {num+1} input")
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
    rouge_file = "rouge_scores.txt"
    print(f"Store Rouge scores in {rouge_file}")
    with open(rouge_file, 'w') as file:
        file.write(f"Average Loss: {average_loss}\n")
        for key, value in rouge_scores.items():
            file.write(f"{key}: {value}\n")
    '''
    # Print the results
    model.eval()
    loss_function = CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)
    total_loss = 0
    print("Start predicting")
    with torch.no_grad():
        for i, batch in enumerate(data_loader):
            input_ids = batch['input_ids'].to(model.device)
            attention_mask = batch['attention_mask'].to(model.device)
            labels = batch['labels'].to(model.device)

            # Generate outputs
            outputs = model.generate(input_ids=input_ids, attention_mask=attention_mask, max_length=OUTPUT_MAX_LENGTH)
            prediction_logits = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels).logits

            # Calculate loss
            loss = loss_function(prediction_logits.view(-1, model.config.vocab_size), labels.view(-1))
            total_loss += loss.item()

            print(f"Input: {df['input'].iloc[i]}")
            print(f"Prediction: {tokenizer.decode(outputs[0], skip_special_tokens=True)}")
            print(f"Loss: {loss.item()}")
            print("------")

    average_loss = total_loss / len(data_loader)
    print(f"Average Loss: {average_loss}")
    '''
    
if __name__ == "__main__":
    if len(sys.argv) > 2: # Don't forget to change this if you change the inputs
        print("Good job dummy, starting the predicting script")
        predict_csv = sys.argv[1]
        model_path = sys.argv[2]
        my_predict(predict_csv, model_path)
    else:
        print("Please provide a filename for test data.")