import os
import json
import numpy as np

def merge_val_dicts(dicts):
    result = {}
    for d in dicts:
        for k, v in d.items():
            try:
                result[k] = result.get(k, 0) + int(v)
            except Exception as e:
                print(f"[merge] Erro em chave: {k}, valor: {v} -> {e}")
    return result

def process_rtt_folder_recursive(folder_path):
    rtt_data = []
    for root, _, files in os.walk(folder_path):
        for filename in files:
            if not filename.endswith(".json"):
                continue
            file_path = os.path.join(root, filename)
            try:
                with open(file_path, 'r') as f:
                    try:
                        data = json.load(f)  # <- Lê como um JSON inteiro, que é uma lista
                        val_list = []
                        for entry in data:
                            if isinstance(entry, dict) and "val" in entry and isinstance(entry["val"], dict):
                                val_list.append(entry["val"])
                        if val_list:
                            merged = merge_val_dicts(val_list)
                            rtt_data.append({"val": merged})
                            print(f"[OK] {file_path}: {len(val_list)} entradas.")
                        else:
                            print(f"[VAZIO] {file_path}")
                    except json.JSONDecodeError as e:
                        print(f"[JSON ERROR] {file_path}: {e}")
            except Exception as e:
                print(f"[ERRO] {file_path}: {e}")
    return rtt_data

def extract_rtt_features(rtt_data):
    values = []
    for entry in rtt_data:
        val_dict = entry["val"]
        for k, v in val_dict.items():
            try:
                values.extend([float(k)] * int(v))
            except Exception as e:
                print(f"[ERRO VALUE] k={k}, v={v} -> {e}")
    if values:
        return [np.mean(values), np.std(values)]
    else:
        return [0, 0]

if __name__ == "__main__":
    folder_path = "./Train/rtt"
    rtt_data = process_rtt_folder_recursive(folder_path)
    print(f"Total de arquivos processados: {len(rtt_data)}")
    features = extract_rtt_features
