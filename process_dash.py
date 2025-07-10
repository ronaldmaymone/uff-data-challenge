import os
import json
import numpy as np

def process_dash_folder(folder_path):
    dash_data = []
    for root, _, files in os.walk(folder_path):
        for filename in files:
            file_path = os.path.join(root, filename)
            try:
                with open(file_path, 'r') as f:
                    lines = f.readlines()
                    
                    # Ignora arquivos quebrados
                    if len(lines) < 15:
                        print(f"[SKIP] {file_path}: menos de 15 linhas")
                        continue
                    
                    # Os 15 primeiros são as requisições (1 por segmento)
                    rates = []
                    for i in range(15):
                        try:
                            req = json.loads(lines[i])
                            if "rate" in req:
                                rates.append(req["rate"])
                        except json.JSONDecodeError as e:
                            print(f"[ERRO JSON] {file_path} linha {i}: {e}")
                    
                    if rates:
                        dash_data.append({"rate": rates})
                        print(f"[OK] {file_path}: média={np.mean(rates):.2f}, std={np.std(rates):.2f}")
                    else:
                        print(f"[VAZIO] {file_path}: nenhuma rate extraída")

            except Exception as e:
                print(f"[ERRO ABERTURA] {file_path}: {e}")
    
    return dash_data

def extract_dash_features(dash_data):
    means = []
    stds = []
    for entry in dash_data:
        rates = entry["rate"]
        means.append(np.mean(rates))
        stds.append(np.std(rates))
    return means, stds

if __name__ == "__main__":
    folder_path = "/home/luis/trabalho_pesquisa/uff-data-challenge/Train/dash"
    dash_data = process_dash_folder(folder_path)
    print(f"\nTotal de arquivos processados: {len(dash_data)}")
    
    mean_rates, std_rates = extract_dash_features(dash_data)
    print(f"\nResumo das taxas (rate):")
    print(f"  Média global das médias: {np.mean(mean_rates):.2f}")
    print(f"  Média global dos desvios: {np.mean(std_rates):.2f}")
