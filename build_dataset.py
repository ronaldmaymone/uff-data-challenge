import os
import json
import uuid
from datetime import datetime
from collections import defaultdict

class DataUnifier:
    def __init__(self, base_path="./Train", output_path="./Queries"):
        self.base_path = base_path
        self.output_path = output_path
        os.makedirs(output_path, exist_ok=True)
        
        # Mapeamento de arquivos por par cliente-servidor e timestamp
        self.dash_files = defaultdict(list)
        self.rtt_files = defaultdict(dict)
        self.traceroute_files = defaultdict(dict)

    def scan_files(self):
        """Varre a estrutura de diretórios e organiza os arquivos por chave"""
        print("Escaneando estrutura de arquivos...")
        
        # Processar arquivos DASH (.jsonl)
        dash_root = os.path.join(self.base_path, "dash")
        for cliente in os.listdir(dash_root):
            cliente_path = os.path.join(dash_root, cliente)
            if not os.path.isdir(cliente_path):
                continue
                
            for servidor in os.listdir(cliente_path):
                servidor_path = os.path.join(cliente_path, servidor)
                if not os.path.isdir(servidor_path):
                    continue
                
                for file in os.listdir(servidor_path):
                    if file.endswith('.jsonl'):
                        # Extrair timestamp do nome do arquivo (formato: YYMMDD_HHMM.jsonl)
                        try:
                            dt_str = file.split('.')[0]
                            dt = datetime.strptime(dt_str, "%y%m%d_%H%M")
                            timestamp = int(dt.timestamp())
                            
                            key = (cliente, servidor, timestamp)
                            self.dash_files[key].append(os.path.join(servidor_path, file))
                        except:
                            continue
        
        # Processar arquivos RTT (.json)
        rtt_root = os.path.join(self.base_path, "rtt")
        for cliente in os.listdir(rtt_root):
            cliente_path = os.path.join(rtt_root, cliente)
            if not os.path.isdir(cliente_path):
                continue
                
            for file in os.listdir(cliente_path):
                if file.endswith('.json'):
                    servidor = file.split('.')[0]
                    key = (cliente, servidor)
                    self.rtt_files[key] = os.path.join(cliente_path, file)
        
        # Processar arquivos Traceroute (.json)
        traceroute_root = os.path.join(self.base_path, "traceroute")
        for servidor in os.listdir(traceroute_root):
            servidor_path = os.path.join(traceroute_root, servidor)
            if not os.path.isdir(servidor_path):
                continue
                
            for file in os.listdir(servidor_path):
                if file.endswith('.json'):
                    cliente = file.split('.')[0]
                    key = (servidor, cliente)  # Note a ordem inversa
                    self.traceroute_files[key] = os.path.join(servidor_path, file)

    def create_queries(self, window_size=10):
        """Cria consultas agrupando medições em janelas temporais"""
        print("Criando consultas unificadas...")
        
        # Agrupar arquivos DASH por cliente-servidor e ordenar por timestamp
        grouped = defaultdict(list)
        for (cliente, servidor, ts), files in self.dash_files.items():
            grouped[(cliente, servidor)].append((ts, files[0]))  # Assume 1 arquivo por timestamp
        
        # Para cada par cliente-servidor, criar consultas com janelas de 'window_size' medições
        for (cliente, servidor), measurements in grouped.items():
            measurements.sort()  # Ordenar por timestamp
            
            # Criar janelas sobrepostas (slide=1)
            for i in range(len(measurements) - window_size + 1):
                window = measurements[i:i+window_size]
                start_ts = window[0][0]
                end_ts = window[-1][0]
                
                # Criar consulta
                query = {
                    "id": str(uuid.uuid4().hex),
                    "cliente": cliente,
                    "servidor": servidor,
                    "dash": [],
                    "rtt": [],
                    "traceroute": []
                }
                
                # Processar medições DASH na janela
                for ts, file in window:
                    dash_data = self._load_dash_file(file)
                    if dash_data:
                        query["dash"].append({
                            "elapsed": [d["elapsed"] for d in dash_data],
                            "request_ticks": [d["request_ticks"] for d in dash_data],
                            "rate": [d["rate"] for d in dash_data],
                            "received": [d["received"] for d in dash_data],
                            "timestamp": [d["timestamp"] - start_ts for d in dash_data]  # Timestamps relativos
                        })
                
                # Carregar RTTs no intervalo [start_ts, end_ts]
                rtt_key = (cliente, servidor)
                if rtt_key in self.rtt_files:
                    rtt_data = self._load_rtt_file(self.rtt_files[rtt_key], start_ts, end_ts)
                    query["rtt"] = rtt_data
                
                # Carregar Traceroutes no intervalo [start_ts, end_ts]
                traceroute_key = (servidor, cliente)  # Ordem inversa
                if traceroute_key in self.traceroute_files:
                    traceroute_data = self._load_traceroute_file(
                        self.traceroute_files[traceroute_key], start_ts, end_ts)
                    query["traceroute"] = traceroute_data
                
                # Salvar consulta
                output_file = os.path.join(self.output_path, f"{query['id']}.json")
                with open(output_file, 'w') as f:
                    json.dump(query, f, indent=2)
                
                print(f"Consulta {query['id']} criada: {cliente}->{servidor} ({start_ts} a {end_ts})")

    def _load_dash_file(self, file_path):
        """Carrega um arquivo DASH .jsonl"""
        try:
            with open(file_path, 'r') as f:
                return [json.loads(line) for line in f if line.strip()][:15]  # Primeiras 15 linhas
        except:
            return None

    def _load_rtt_file(self, file_path, start_ts, end_ts):
        """Carrega RTTs no intervalo especificado"""
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
                return [entry for entry in data 
                       if start_ts <= entry["ts"] <= end_ts]
        except:
            return []

    def _load_traceroute_file(self, file_path, start_ts, end_ts):
        """Carrega Traceroutes no intervalo especificado"""
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
                return [entry for entry in data 
                       if start_ts <= entry["ts"] <= end_ts]
        except:
            return []

if __name__ == "__main__":
    print("=== Unificador de Dados para Data Challenge ===")
    
    unifier = DataUnifier()
    
    # Passo 1: Varre a estrutura de arquivos
    unifier.scan_files()
    
    # Passo 2: Cria consultas unificadas (janela de 10 medições DASH)
    unifier.create_queries(window_size=10)
    
    print("Processo concluído. Consultas salvas em:", unifier.output_path)
