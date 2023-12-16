import pandas as pd
import json
import sys

def truncate_to_limit(text, word_limit):
    words = text.split()
    return ' '.join(words[:word_limit]) if len(words) > word_limit else text

def convert(csv_file_path, jsonl_file_path): 
    df = pd.read_csv(csv_file_path)

    # Open a file to write the JSONL content
    with open(jsonl_file_path, 'w') as file:
        for index, row in df.iterrows():
            # there shouldn't be candidates larger than 128 but 512 is the max input size for BLEURT
            if pd.isna(row["Prediction"]):
                truncated_candidate = "Nothing here" # Some of the models were generating Null outputs so this should catch that
            else:
                truncated_candidate = truncate_to_limit(row["Prediction"], 256)
            truncated_reference = truncate_to_limit(row["Reference"], 256)

            json_object = {"candidate": truncated_candidate, "reference": truncated_reference}

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

