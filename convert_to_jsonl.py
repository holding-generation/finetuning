import pandas as pd
import json
import sys

# I used this to convert training data to jsonl, for use with finetuning Llama2
# Ultimately we didn't end up using that finetuned model.
def truncate_to_limit(text, word_limit):
    words = text.split()
    return ' '.join(words[:word_limit]) if len(words) > word_limit else text

def convert(csv_file_path, jsonl_file_path):
    df = pd.read_csv(csv_file_path)

    with open(jsonl_file_path, 'w') as file:
        for index, row in df.iterrows():
            truncated_input = truncate_to_limit(row["input"], 3900)
            truncated_output = truncate_to_limit(row["output"], 128)

            json_object = {"input": truncated_input, "output": truncated_output}
            
            file.write(json.dumps(json_object) + "\n")

    print("Conversion to JSONL completed.")

if __name__ == "__main__":
    if len(sys.argv) > 2:
        print("I hope you put the csv filename first and the jsonl filename second.")
        csv_file_path = sys.argv[1]
        jsonl_file_path = sys.argv[2]
        convert(csv_file_path, jsonl_file_path)
    else:
        print("Please provide a filename to convert data.")

