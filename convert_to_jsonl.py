import pandas as pd
import json
import sys

def convert(csv_file_path, jsonl_file_path):
    # Load the CSV file  
    df = pd.read_csv(csv_file_path)

    # Open a file to write the JSONL content
    with open(jsonl_file_path, 'w') as file:
        for index, row in df.iterrows():
            # Create a JSON object for each row
            json_object = {"input": row["input"], "output": row["output"]}
            
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

