import os
import json
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_percentage_error
from joblib import dump, load

class NetworkPredictor:
    def __init__(self, queries_dir="./Queries", models_dir="./Models"):
        self.queries_dir = queries_dir
        self.models_dir = models_dir
        os.makedirs(models_dir, exist_ok=True)
        
        # Dicionário para armazenar os modelos (um para cada métrica)
        self.models = {
            'mean_1': None,
            'stdev_1': None,
            'mean_2': None,
            'stdev_2': None
        }

    def load_queries(self):
        """Carrega todas as consultas da pasta especificada"""
        queries = []
        for filename in os.listdir(self.queries_dir):
            if filename.endswith('.json'):
                with open(os.path.join(self.queries_dir, filename), 'r') as f:
                    queries.append(json.load(f))
        return queries

    def extract_features(self, query):
        """Extrai features de uma consulta para treinamento"""
        features = {}
        
        # 1. Features das medições DASH (últimas 3 medições)
        last_3_dash = query['dash'][-3:] if len(query['dash']) >= 3 else query['dash']
        
        for i, dash in enumerate(last_3_dash, start=1):
            features.update({
                f'dash_{i}_rate_mean': np.mean(dash['rate']),
                f'dash_{i}_rate_std': np.std(dash['rate']),
                f'dash_{i}_elapsed_mean': np.mean(dash['elapsed']),
                f'dash_{i}_throughput': np.sum(dash['received']) / np.sum(dash['elapsed'])
            })
        
        # 2. Features de RTT (se existirem)
        if query['rtt']:
            rtt_values = []
            for entry in query['rtt']:
                for val, count in entry['val'].items():
                    rtt_values.extend([float(val)] * int(count))
            
            features.update({
                'rtt_mean': np.mean(rtt_values),
                'rtt_std': np.std(rtt_values),
                'rtt_min': min(rtt_values),
                'rtt_max': max(rtt_values)
            })
        
        # 3. Features de Traceroute (se existirem)
        if query['traceroute']:
            hop_counts = []
            rtts = []
            for entry in query['traceroute']:
                if 'val' in entry:
                    hop_counts.append(len(entry['val']))
                    for hop in entry['val']:
                        if 'rtt' in hop:
                            rtts.append(hop['rtt'])
            
            features.update({
                'hops_mean': np.mean(hop_counts) if hop_counts else 0,
                'traceroute_rtt_mean': np.mean(rtts) if rtts else 0
            })
        
        return features

    def prepare_training_data(self, queries):
        """Prepara os dados de treinamento com features e targets"""
        X, y = [], {'mean_1': [], 'stdev_1': [], 'mean_2': [], 'stdev_2': []}
        
        # Agrupar consultas por par cliente-servidor
        grouped = {}
        for query in queries:
            key = (query['cliente'], query['servidor'])
            grouped.setdefault(key, []).append(query)
        
        # Para cada par cliente-servidor, criar exemplos de treinamento
        for key, group_queries in grouped.items():
            group_queries.sort(key=lambda x: min(m['timestamp'][0] for m in x['dash']))
            
            # Criar pares (janela, próximas 2 medições)
            for i in range(len(group_queries) - 2):
                # Features da janela atual
                X.append(self.extract_features(group_queries[i]))
                
                # Targets (próximas 2 medições)
                next_1 = group_queries[i+1]
                next_2 = group_queries[i+2]
                
                # Calcular médias e desvios padrão
                y['mean_1'].append(np.mean([r for m in next_1['dash'] for r in m['rate']]))
                y['stdev_1'].append(np.std([r for m in next_1['dash'] for r in m['rate']]))
                y['mean_2'].append(np.mean([r for m in next_2['dash'] for r in m['rate']]))
                y['stdev_2'].append(np.std([r for m in next_2['dash'] for r in m['rate']]))
        
        return pd.DataFrame(X), pd.DataFrame(y)

    def train_models(self, X, y):
        """Treina um modelo separado para cada métrica alvo"""
        # Dividir em treino e validação
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
        
        for target in ['mean_1', 'stdev_1', 'mean_2', 'stdev_2']:
            print(f"\nTreinando modelo para {target}...")
            
            model = RandomForestRegressor(
                n_estimators=200,
                max_depth=10,
                min_samples_split=5,
                random_state=42,
                n_jobs=-1
            )
            
            model.fit(X_train, y_train[target])
            
            # Avaliar
            val_pred = model.predict(X_val)
            mape = mean_absolute_percentage_error(y_val[target], val_pred)
            print(f"MAPE para {target}: {mape:.4f}")
            
            self.models[target] = model
            
            # Salvar modelo
            dump(model, os.path.join(self.models_dir, f'{target}.joblib'))
    
    def predict_for_submission(self, test_queries):
        """Gera previsões no formato de submissão exigido"""
        predictions = []
        
        for query in test_queries:
            # Extrair features
            features = self.extract_features(query)
            features_df = pd.DataFrame([features])
            
            # Fazer previsões
            pred = {
                'id': query['id'],
                'mean_1': self.models['mean_1'].predict(features_df)[0],
                'stdev_1': self.models['stdev_1'].predict(features_df)[0],
                'mean_2': self.models['mean_2'].predict(features_df)[0],
                'stdev_2': self.models['stdev_2'].predict(features_df)[0]
            }
            predictions.append(pred)
        
        # Criar DataFrame e salvar como CSV
        submission = pd.DataFrame(predictions)
        submission.to_csv('submission.csv', index=False)
        print("\nArquivo de submissão gerado: submission.csv")
        
        return submission

if __name__ == "__main__":
    print("=== Sistema de Predição para o Data Challenge ===")
    
    # 1. Instanciar o predictor
    predictor = NetworkPredictor()
    
    # 2. Carregar consultas de treinamento
    print("\nCarregando consultas de treinamento...")
    train_queries = predictor.load_queries()
    
    # 3. Preparar dados de treinamento
    print("Preparando dados de treinamento...")
    X, y = predictor.prepare_training_data(train_queries)
    
    # 4. Treinar modelos
    print("\nIniciando treinamento dos modelos...")
    predictor.train_models(X, y)
    
    # 5. [OPCIONAL] Para gerar submissão com dados de teste:
    # test_queries = [...]  # Carregar consultas de teste
    # submission = predictor.predict_for_submission(test_queries)
    
    print("\nProcesso concluído com sucesso!")
