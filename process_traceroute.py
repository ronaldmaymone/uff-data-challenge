import os
import json
import numpy as np

def extract_traceroute_features(traces):
    hop_counts = [len(t["val"]) for t in traces if "val" in t]
    return [np.mean(hop_counts)] if hop_counts else [0]

def process_traceroute_folder_recursive(folder_path):
    results = []
    for root, _, files in os.walk(folder_path):
        for filename in files:
            if not filename.endswith(".json"):
                continue
            file_path = os.path.join(root, filename)
            try:
                with open(file_path, 'r') as f:
                    traces = json.load(f)  # lê o arquivo inteiro como JSON
                features = extract_traceroute_features(traces)
                results.append((file_path, features))
            except Exception as e:
                print(f"Erro ao processar {file_path}: {e}")
    return results

if __name__ == "__main__":
    folder_path = "./Train/traceroute"
    results = process_traceroute_folder_recursive(folder_path)
    for file_path, features in results:
        print(f"Arquivo: {file_path}")
        print(f"Média de hops: {features[0]}\n")
