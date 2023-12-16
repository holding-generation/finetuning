import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import torch
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, LongT5ForConditionalGeneration
from rouge import Rouge
import pandas as pd
import sys
import csv

INPUT_MAX_LENGTH = 4096 # use 2048 on a T4
OUTPUT_MAX_LENGTH = 64
BATCH_SIZE = 1
''' 
This is the same inference loop that is found in the finetuning scripts.
I separated it out so I could run inference if/when that script got interrupted.
'''
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

def my_predict(predict_csv_name, model_path):
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = LongT5ForConditionalGeneration.from_pretrained(model_path)

    df = pd.read_csv(predict_csv_name)
    dataset = CustomDataset(tokenizer, df, INPUT_MAX_LENGTH, OUTPUT_MAX_LENGTH)

    # Try the trainer.predict method
    # For some reason trainer.predict was causing an OOM error, so I had to 
    # implement training manually. I had chatGPT get me started with the 
    # structure of the loop and the function calls and then I modified, for
    # example by setting batch size to 1 and setting it to calculate ROUGE scores.
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

                outputs = model.generate(input_ids=input_ids, attention_mask=attention_mask, max_length=OUTPUT_MAX_LENGTH)
                prediction_logits = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels).logits

                loss = loss_function(prediction_logits.view(-1, model.config.vocab_size), labels.view(-1))
                total_loss += loss.item()

                input_text = df['input'].iloc[i]
                reference_text = df['output'].iloc[i]
                generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

                # Store to lists for ROUGE calculation
                hypotheses.append(generated_text)
                references.append(reference_text)

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
    
if __name__ == "__main__":
    if len(sys.argv) > 2: # Don't forget to change this if you change the inputs
        print("Good job dummy, starting the predicting script")
        predict_csv = sys.argv[1]
        model_path = sys.argv[2]
        my_predict(predict_csv, model_path)
    else:
        print("Please provide a filename for test data.")