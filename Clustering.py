import json
import os
import re
from pathlib import Path
from typing import Dict, List, Set, Tuple
from collections import defaultdict
from difflib import SequenceMatcher

def load_graphs_from_directory(directory: str) -> List[Dict]:
    """Carica tutti i file JSON di knowledge graph da una directory"""
    graphs = []
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith('.json') and not file.startswith('aggregated'):
                file_path = os.path.join(root, file)
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        graph = json.load(f)
                        if isinstance(graph, dict) and 'entities' in graph:
                            graphs.append(graph)
                    print(f"Loaded graph from {file_path}")
                except Exception as e:
                    print(f"Error loading {file_path}: {str(e)}")
    return graphs

def normalize_entity(entity: str) -> str:
    """Normalizza un'entità per il confronto"""
    # Rimuovi punteggiatura e articoli iniziali
    entity = re.sub(r'^\s*(il|la|lo|i|gli|le|un|uno|una)\s+', '', entity.lower())
    entity = re.sub(r'[^\w\s]', '', entity).strip()
    return entity

def similar(a: str, b: str, threshold: float = 0.8) -> bool:
    """Determina se due stringhe sono simili"""
    return SequenceMatcher(None, a, b).ratio() >= threshold

def cluster_entities(entities: List[str], similarity_threshold: float = 0.85) -> Dict[str, List[str]]:
    """Clusterizza entità simili con un algoritmo più sofisticato"""
    normalized_entities = [(normalize_entity(e), e) for e in entities]
    clusters = defaultdict(list)
    used_indices = set()
    
    for i, (norm1, orig1) in enumerate(normalized_entities):
        if i in used_indices:
            continue
            
        # Crea un nuovo cluster per questa entità
        clusters[orig1].append(orig1)
        used_indices.add(i)
        
        # Cerca entità simili
        for j, (norm2, orig2) in enumerate(normalized_entities[i+1:], start=i+1):
            if j in used_indices:
                continue
                
            # Confronta usando similarità e prefissi comuni
            if (similar(norm1, norm2, similarity_threshold) or
                norm1.split()[0] == norm2.split()[0] or
                norm1[:5] == norm2[:5]):
                clusters[orig1].append(orig2)
                used_indices.add(j)
    
    return dict(clusters)

def merge_relations(relations: List[Tuple[str, str, str]], 
                   entity_clusters: Dict[str, List[str]]) -> List[Tuple[str, str, str]]:
    """Unisce relazioni basate sui cluster di entità con controllo qualità"""
    merged_relations = set()
    entity_to_cluster = {}
    
    # Costruisci mappa entità -> cluster
    for cluster_rep, members in entity_clusters.items():
        for member in members:
            entity_to_cluster[member] = cluster_rep
    
    for src, rel, tgt in relations:
        # Gestisci entità mancanti nei cluster
        new_src = entity_to_cluster.get(src, src)
        new_tgt = entity_to_cluster.get(tgt, tgt)
        
        # Filtra relazioni non informative
        if (new_src.lower() == new_tgt.lower() or 
            len(rel.strip()) < 3 or
            rel.lower() in ['is', 'has', 'of']):
            continue
            
        merged_relations.add((new_src, rel, new_tgt))
    
    return sorted(merged_relations)

def analyze_relations(relations: List[Tuple[str, str, str]]) -> Dict:
    """Analizza le relazioni per identificare quelle più comuni"""
    rel_counts = defaultdict(int)
    entity_pairs = defaultdict(int)
    
    for src, rel, tgt in relations:
        rel_counts[rel.lower()] += 1
        entity_pairs[(src.lower(), tgt.lower())] += 1
    
    return {
        'top_relations': sorted(rel_counts.items(), key=lambda x: -x[1])[:10],
        'common_entity_pairs': sorted(entity_pairs.items(), key=lambda x: -x[1])[:5]
    }

def cluster_knowledge_graphs(graphs: List[Dict], similarity_threshold: float = 0.85) -> Dict:
    """Clusterizza e combina più knowledge graph con parametri configurabili"""
    all_entities = set()
    all_relations = set()
    
    for graph in graphs:
        all_entities.update(graph.get('entities', []))
        all_relations.update(tuple(rel) for rel in graph.get('relations', []))
    
    print(f"Pre-clustering: {len(all_entities)} entities, {len(all_relations)} relations")
    
    # Clusterizzazione avanzata
    entity_clusters = cluster_entities(list(all_entities), similarity_threshold)
    
    # Unione relazioni con controllo qualità
    merged_relations = merge_relations(list(all_relations), entity_clusters)
    
    # Analisi delle relazioni
    relation_stats = analyze_relations(merged_relations)
    
    return {
        'entities': sorted(entity_clusters.keys()),
        'relations': merged_relations,
        'edges': sorted({rel for _, rel, _ in merged_relations}),
        'entity_clusters': entity_clusters,
        'stats': {
            'original_entities': len(all_entities),
            'clustered_entities': len(entity_clusters),
            'original_relations': len(all_relations),
            'merged_relations': len(merged_relations),
            'relation_analysis': relation_stats
        }
    }

def save_clustered_graph(graph: Dict, output_path: str):
    """Salva il grafo clusterizzato con formattazione leggibile"""
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(graph, f, indent=2, ensure_ascii=False, sort_keys=True)
    print(f"Clustered graph saved to {output_path}")

def process_directory(input_dir: str, output_dir: str, similarity_threshold: float = 0.85):
    """
    Processa tutti i grafi della conoscenza (Knowledge Graphs, KG) presenti in una directory, 
    applicando un algoritmo di clustering basato su una soglia di similarità configurabile.
    Args:
        input_dir (str): Percorso della directory contenente i file dei grafi della conoscenza da processare.
        output_dir (str): Percorso della directory in cui salvare i risultati del clustering.
        similarity_threshold (float, opzionale): Soglia di similarità per il clustering. 
            Valore predefinito: 0.85.
    Funzionalità:
        - Carica i grafi della conoscenza dalla directory di input.
        - Esegue il clustering dei grafi basandosi sulla soglia di similarità fornita.
        - Salva il grafo risultante dal clustering in formato JSON nella directory di output.
        - Salva i dettagli dei cluster (es. cluster di entità e statistiche) in un file separato.
        - Stampa statistiche e analisi delle relazioni principali.
    Note:
        - Se non vengono trovati grafi validi nella directory di input, il processo termina con un messaggio.
        - La directory di output viene creata automaticamente se non esiste.
    Returns:
        None
    """
    """Processa tutti i KG in una directory con parametri configurabili"""
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    print(f"\n{'='*50}")
    print(f"Loading knowledge graphs from {input_dir}")
    graphs = load_graphs_from_directory(input_dir)
    
    if not graphs:
        print("No valid knowledge graphs found!")
        return
    
    print(f"\nClustering {len(graphs)} knowledge graphs (similarity threshold: {similarity_threshold})")
    clustered_graph = cluster_knowledge_graphs(graphs, similarity_threshold)
    
    # Salva il grafo principale
    output_file = os.path.join(output_dir, "clustered_kg.json")
    save_clustered_graph(clustered_graph, output_file)
    
    # Salva i dettagli dei cluster separatamente
    clusters_output = {
        'entity_clusters': clustered_graph['entity_clusters'],
        'stats': clustered_graph['stats']
    }
    clusters_file = os.path.join(output_dir, "clustering_details.json")
    with open(clusters_file, 'w', encoding='utf-8') as f:
        json.dump(clusters_output, f, indent=2, ensure_ascii=False)
    
    # Stampa statistiche
    stats = clustered_graph['stats']
    print("\nClusterization results:")
    print(f"- Original entities: {stats['original_entities']}")
    print(f"- Clustered entities: {stats['clustered_entities']} (reduction: {100*(1-stats['clustered_entities']/stats['original_entities']):.1f}%)")
    print(f"- Original relations: {stats['original_relations']}")
    print(f"- Merged relations: {stats['merged_relations']}")
    
    print("\nTop relations:")
    for rel, count in stats['relation_analysis']['top_relations']:
        print(f"  {rel}: {count} occurrences")
    
    print(f"\n{'='*50}")

if __name__ == "__main__":
    input_directory = r"C:\Users\giggi\Documents\LLM_KG\knowledge_graphs_output\8217481FD4"
    output_directory = r"C:\Users\giggi\Documents\LLM_KG\clustered_knowledge_graphs"
    
    print("Starting advanced knowledge graph clustering...")
    
    # Puoi regolare la soglia di similarità (0.7-0.9)
    process_directory(input_directory, output_directory, similarity_threshold=0.8)