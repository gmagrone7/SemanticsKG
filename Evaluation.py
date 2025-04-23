import json
from sentence_transformers import SentenceTransformer, util

def calcola_copertura_semantica(file_kg1, file_kg2, soglia_similarita=0.8):
    """
    Calcola la copertura semantica delle relazioni del KG1 rispetto al KG2 (gold standard).
    Misura quante relazioni del golden standard (KG2) sono coperte dal KG1.
    
    Args:
        file_kg1 (str): Percorso al file JSON contenente il knowledge graph da valutare.
        file_kg2 (str): Percorso al file JSON contenente il golden standard.
        soglia_similarita (float, opzionale): Soglia di similarità semantica (tra 0 e 1).
                                              Default: 0.8.
    Returns:
        float: La frazione di relazioni del KG2 coperte dal KG1 (tra 0 e 1).
        None: Se uno o entrambi i file JSON non vengono trovati.
    """
    try:
        # Carica i knowledge graph
        with open(file_kg1, 'r', encoding='utf-8') as f1, open(file_kg2, 'r', encoding='utf-8') as f2:
            relazioni_kg1 = [tuple(relazione) for relazione in json.load(f1)['relations']]
            relazioni_kg2 = [tuple(relazione) for relazione in json.load(f2)['relations']]

        # Carica il modello per l'embedding semantico
        modello = SentenceTransformer('paraphrase-multilingual-mpnet-base-v2')

        # Pre-calcola gli embedding per tutte le relazioni (per efficienza)
        embedding_kg1 = [modello.encode(' '.join(rel)) for rel in relazioni_kg1]
        embedding_kg2 = [modello.encode(' '.join(rel)) for rel in relazioni_kg2]

        # Conta quante relazioni del KG2 hanno una corrispondenza nel KG1
        relazioni_coperte = 0

        for i, (rel2, emb2) in enumerate(zip(relazioni_kg2, embedding_kg2)):
            # Trova la similarità massima con qualsiasi relazione nel KG1
            max_similarita = max(
                (util.cos_sim(emb2, emb1).item() 
                for emb1 in embedding_kg1)
            )
            
            if max_similarita >= soglia_similarita:
                relazioni_coperte += 1

        # Calcola la frazione di copertura
        copertura = relazioni_coperte / len(relazioni_kg2) if relazioni_kg2 else 0.0
        
        # Stampa informazioni diagnostiche
        print(f"Relazioni nel golden standard (KG2): {len(relazioni_kg2)}")
        print(f"Relazioni coperte dal KG1: {relazioni_coperte}")
        print(f"Frazione di copertura: {copertura:.4f}")
        
        return copertura

    except FileNotFoundError:
        print("Errore: Uno o entrambi i file JSON non sono stati trovati.")
        return None

# Percorsi dei file JSON
file_kg1 = r'C:\Users\giggi\Documents\LLM_KG\clustered_knowledge_graphs\relations.json'
file_kg2 = r'C:\Users\giggi\Documents\LLM_KG\clustered_knowledge_graphs\relations2.json'

# Calcola la copertura
copertura = calcola_copertura_semantica(file_kg1, file_kg2)

if copertura is not None:
    print(f"Copertura semantica del KG1 rispetto al KG2 (gold standard): {copertura:.4f}")