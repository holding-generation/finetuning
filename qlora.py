import torch
from torch.utils.data import Dataset
from transformers import(
    AutoTokenizer, 
    LongT5ForConditionalGeneration, 
    Trainer, 
    TrainingArguments,
    AutoModelForCausalLM,
    BitsAndBytesConfig
)
import pandas as pd
import sys
from peft import prepare_model_for_kbit_training, LoraConfig, get_peft_model

NUM_EPOCHS = 3
INPUT_MAX_LENGTH = 512
OUTPUT_MAX_LENGTH = 128
BATCH_SIZE = 2

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

def finetune(train_csv_name):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    model_name = "google/long-t5-local-base"
    print("Try loading Tokenizer from HuggingFace")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    print("Try loading model from HuggingFace")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )
    
    model = AutoModelForCausalLM.from_pretrained(model_name, quantization_config=bnb_config)
    # model = LongT5ForConditionalGeneration.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        padding_side="left",
        add_eos_token=True,
        add_bos_token=True,
    )
    tokenizer.pad_token = tokenizer.eos_token

    print("Reading the CSV")
    df = pd.read_csv(train_csv_name)
    print("Creating the dataset")
    train_dataset = CustomDataset(tokenizer, df, INPUT_MAX_LENGTH, OUTPUT_MAX_LENGTH)

    # Tokenizeand create dataset. Old, not using TensorDataset now
    print("Tokenize Inputs")
    inputs = tokenizer(df['input'].tolist(), max_length=INPUT_MAX_LENGTH, padding='max_length', truncation=True, return_tensors="pt")
    print("Tokenize Outputs")
    outputs = tokenizer(df['output'].tolist(), max_length=OUTPUT_MAX_LENGTH, padding='max_length', truncation=True, return_tensors="pt")

    # dataset = TensorDataset(inputs['input_ids'], inputs['attention_mask'], outputs['input_ids'])

    # Do QLoRA
    model.gradient_checkpointing_enable()
    model = prepare_model_for_kbit_training(model)
    lora_config = LoraConfig(
        r=32,
        lora_alpha=64,
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
            "lm_head",
        ],
        bias="none",
        lora_dropout=0.05,  # Conventional
        task_type="CAUSAL_LM",
    )

    model = get_peft_model(model, lora_config)

    print("Training the model")
    # Not sure this tokenizer.pad_token is necessary
    # tokenizer.pad_token = tokenizer.eos_token

    print(f"Train for {NUM_EPOCHS} epochs")
    training_args = TrainingArguments(
        output_dir='./results',
        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        warmup_steps=500,
        gradient_checkpointing=True,
        weight_decay=0.01,
        logging_dir='./logs',
        logging_steps=10,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        # eval_dataset=eval_dataset if you have it
    )
    
    trainer.train()

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
