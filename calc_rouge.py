import csv
import sys
from rouge import Rouge 

# This was just to test rouge scores for myself. Lawrence was mainly in charge of this metric
def calculate_rouge(csv_file_path):
    hypotheses = []
    references = []

    print("Reading csv for results")
    with open(csv_file_path, 'r', encoding='utf-8') as file:
        reader = csv.DictReader(file)
        for row in reader:
            hypotheses.append(row['Prediction'])
            references.append(row['Reference'])

    rouge = Rouge()
    rouge_scores = rouge.get_scores(hypotheses, references, avg=True)

    print("Printing ROUGE Scores:")
    for key, value in rouge_scores.items():
        print(f"{key}: {value}")
    
    rouge_file = "rouge_scores.txt"
    print(f"Store Rouge scores in {rouge_file}")
    with open(rouge_file, 'w') as file:
        for key, value in rouge_scores.items():
            file.write(f"{key}: {value}\n")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Wrong inputs dummy: python script.py <path_to_csv_file>")
        sys.exit(1)

    csv_file_path = sys.argv[1]
    calculate_rouge(csv_file_path)