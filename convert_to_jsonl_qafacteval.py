import pandas as pd
import json
import sys

def truncate_to_limit(text, word_limit):
    words = text.split()
    return ' '.join(words[:word_limit]) if len(words) > word_limit else text

def convert(csv_file_path, jsonl_file_path):
    # Load the CSV file  
    df = pd.read_csv(csv_file_path)

    # Open a file to write the JSONL content
    with open(jsonl_file_path, 'w') as file:
        for index, row in df.iterrows():
            truncated_input = truncate_to_limit(row["Input"], 3900)
            truncated_output = truncate_to_limit(row["Prediction"], 128)

            # Create a JSON object for each row
            json_object = {"document": {"text": truncated_input}, "claim": truncated_output}
            
            
            # Write the JSON object to the file
            file.write(json.dumps(json_object) + "\n")

    print("Conversion to JSONL completed.")

# csv_file_path = './test_set_for_jesse_clean.csv'
# jsonl_file_path = 'test_qlora.jsonl'

if __name__ == "__main__":
    if len(sys.argv) > 2:
        print("I hope you put the csv filename first and the jsonl filename second.")
        csv_file_path = sys.argv[1]
        jsonl_file_path = sys.argv[2]
        convert(csv_file_path, jsonl_file_path)
    else:
        print("Please provide a filename to convert data.")

