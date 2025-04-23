# Usa un'immagine base di Python leggera
FROM python:3.10-slim

# Imposta la directory di lavoro nel container
WORKDIR /app

# Aggiorna i repository e installa pacchetti di sistema essenziali
RUN apt-get update && apt-get install -y build-essential git && rm -rf /var/lib/apt/lists/*

# Installa le dipendenze necessarie
RUN pip install --no-cache-dir kg-gen llama-cpp-python networkx matplotlib ollama

# Copia solo il file necessario nel container
COPY generate_kg_with_local_model_newline_chunk.py /app/generate_kg_with_local_model_newline_chunk.py

# Comando di default per eseguire lo script corretto
CMD ["python", "/app/generate_kg_with_local_model_newline_chunk.py"]
