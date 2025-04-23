import json
import time
import ollama
from typing import List, Tuple, Dict, Set, Optional
import os
from pathlib import Path
from litellm.exceptions import RateLimitError
from kg_gen import KGGen


def transform_to_triples(response: Dict) -> List[Tuple[str, str, str]]:
    """Convert the nodes/edges format to list of triples"""
    triples = []
    node_map = {str(node['id']): node['label'] for node in response.get('nodes', [])}
    
    for edge in response.get('edges', []):
        source_label = node_map.get(str(edge['source']), str(edge['source']))
        target_label = node_map.get(str(edge['target']), str(edge['target']))
        relation = edge['relation']
        triples.append((source_label, relation, target_label))
    
    return triples

def extract_json(text: str) -> Optional[Dict]:
    try:
        if '```json' in text:
            text = text.split('```json')[1].split('```')[0]
        json_start = text.find('{')
        json_end = text.rfind('}') + 1
        json_str = text[json_start:json_end]
        return json.loads(json_str)
    except Exception as e:
        print(f"JSON parsing error: {e}")
        return None

def safe_generate(model: str, prompt: str) -> Optional[Dict]:
    delay = 10
    max_attempts = 5
    attempts = 0
    
    while attempts < max_attempts:
        try:
            response = ollama.chat(model=model, messages=[{"role": "user", "content": prompt}])
            print("DEBUG: Ollama response:", response)
            
            response_text = response.get("message", {}).get("content", "")
            parsed_output = extract_json(response_text)
            
            if parsed_output:
                return parsed_output
            else:
                print("Error: No valid JSON found in response.")
                return None
        except RateLimitError:
            print(f"RateLimitError: Waiting {delay} seconds... ({attempts+1}/{max_attempts})")
            time.sleep(delay)
            attempts += 1
        except Exception as e:
            print(f"Unexpected error: {e}. Skipping chunk.")
            return None
    
    print("Max attempts reached, skipping.")
    return None

def chunk_text(text: str, chunk_size: int) -> List[str]:
    """Split text into chunks of specified size, trying to respect paragraph boundaries"""
    chunks = []
    for i in range(0, len(text), chunk_size):
        chunk = text[i:i + chunk_size]
        last_paragraph = chunk.rfind('\n\n')
        if last_paragraph > 0 and (i + chunk_size) < len(text):
            chunk = chunk[:last_paragraph]
        chunks.append(chunk)
    return chunks

def aggregate_graphs(graphs: List[Dict]) -> Dict:
    """
    Aggregate multiple knowledge graphs in KG-GEN style
    Args:
        graphs: List of knowledge graph dictionaries
    Returns:
        Combined knowledge graph
    """
    all_entities: Set[str] = set()
    all_relations: Set[Tuple[str, str, str]] = set()
    all_edges: Set[str] = set()
    
    for graph in graphs:
        all_entities.update(graph.get('entities', []))
        all_relations.update(graph.get('relations', []))
        all_edges.update(graph.get('edges', []))
    
    return {
        'entities': sorted(all_entities),
        'relations': sorted(all_relations),
        'edges': sorted(all_edges)
    }

# ==============================================
# NUOVE FUNZIONI PER GENERATE-ON-GRAPH
# ==============================================

def complete_graph(graph: Dict, model: str) -> Dict:
    """Ask the LLM to suggest missing relations between existing entities"""
    entities = graph['entities']
    existing_relations = graph['relations']
    
    prompt = f"""
    Given these entities: {entities[:50]}... (showing first 50)
    And some existing relations (for reference): {existing_relations[:5]}...
    
    Suggest 3-5 NEW plausible relations that might be missing between these entities.
    Use EXACTLY this JSON format for each relation:
    {{"source": "subject", "target": "object", "relation": "predicate"}}
    
    Return ONLY a JSON list of such relation objects.
    """
    
    new_edges = safe_generate(model, prompt)
    if new_edges and isinstance(new_edges, list):
        for edge in new_edges:
            try:
                triple = (edge['source'], edge['relation'], edge['target'])
                if triple not in existing_relations:
                    graph['relations'].append(triple)
                    print(f"Added new relation: {triple}")
            except KeyError:
                continue
    return graph

def refine_knowledge_graph(graph: Dict, model: str, max_iterations: int = 2) -> Dict:
    """Iteratively improve the KG by finding missing connections"""
    for iteration in range(max_iterations):
        prev_count = len(graph['relations'])
        graph = complete_graph(graph, model)
        new_count = len(graph['relations'])
        
        print(f"Iteration {iteration+1}: Relations {prev_count} → {new_count}")
        if new_count == prev_count:
            break
    return graph

    """  Genera un grafo della conoscenza a partire da uno o più file di input.
    Args:
        input_files (List[str]): Lista di percorsi ai file di testo da analizzare.
        output_file (str): Percorso del file di output dove salvare il grafo della conoscenza generato in formato JSON.
        model (str): Nome del modello da utilizzare per generare il grafo della conoscenza.
        chunk_size (int, opzionale): Dimensione massima dei chunk di testo in caratteri. Default: 5000.
        refine (bool, opzionale): Se True, applica una fase di raffinamento Generate-on-Graph al grafo generato. Default: True.
    Returns:
        Dict: Dizionario contenente il grafo della conoscenza con le seguenti chiavi:
            - 'entities': Lista di entità uniche identificate.
            - 'relations': Lista di tuple (soggetto, relazione, oggetto) che rappresentano le relazioni.
            - 'edges': Lista di relazioni uniche.
    Note:
        - Il testo di ogni file viene suddiviso in chunk per facilitare l'elaborazione.
        - Ogni chunk viene analizzato per generare nodi e relazioni in formato JSON.
        - Se `refine` è abilitato, il grafo iniziale viene ulteriormente raffinato per migliorare la qualità delle relazioni.
        - Il grafo finale viene salvato nel file di output specificato in formato JSON."""
def generate_knowledge_graph(input_files: List[str], output_file: str, model: str, chunk_size: int = 5000, refine: bool = True) -> Dict:
    """Generate knowledge graph with optional Generate-on-Graph refinement"""
    # Parte originale invariata
    all_relations: Set[Tuple[str, str, str]] = set()
    
    for input_file in input_files:
        print(f"\nProcessing file: {input_file}")
        with open(input_file, "r", encoding="utf-8") as f:
            text = f.read()

        chunks = chunk_text(text, chunk_size)
        print(f"Split into {len(chunks)} chunk(s)")

        for i, chunk in enumerate(chunks):
            print(f"Processing chunk {i+1}/{len(chunks)}")
            prompt = f"""
                Analyze this text and generate a knowledge graph in JSON format with:
                - "nodes": list of objects with "id" (number) and "label" (string)
                - "edges": list of objects with "source", "target" (node ids), and "relation" (string)
                Example format:
                {{
                  "nodes": [{{"id": 1, "label": "Concept1"}}, {{"id": 2, "label": "Concept2"}}],
                  "edges": [{{"source": 1, "target": 2, "relation": "related_to"}}]
                }}
                Text to analyze:
                {chunk}
                Return only the JSON, no additional text or markdown.
            """
            
            response = safe_generate(model, prompt)
            
            if response:
                try:
                    triples = transform_to_triples(response)
                    print(f"Generated {len(triples)} triples from chunk")
                    for triple in triples:
                        all_relations.add(triple)
                except Exception as e:
                    print(f"Error processing triples: {str(e)}")
            else:
                print(f"Chunk {i+1} skipped due to errors.")
    
    if not all_relations:
        print("No valid relations generated. Exiting.")
        return {}

    # Parte originale invariata
    entities = set()
    for src, rel, tgt in all_relations:
        entities.add(src)
        entities.add(tgt)

    graph = {
        'entities': list(entities),
        'relations': list(all_relations),
        'edges': list({rel for _, rel, _ in all_relations})
    }
    
    # NUOVA PARTE: Generate-on-Graph refinement
    if refine:
        print("\nStarting Generate-on-Graph refinement...")
        print(f"Initial graph has {len(graph['relations'])} relations")
        graph = refine_knowledge_graph(graph, model)
        print(f"After refinement: {len(graph['relations'])} relations")
    
    # Parte originale invariata
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(graph, f, indent=2, ensure_ascii=False)
    print(f"\nKnowledge graph saved to {output_file}")
    
    return graph


def process_all_files(input_dir: str, output_dir: str, model: str, 
                     chunk_size: int = 5000, refine: bool = True):
    """Process all files with optional Generate-on-Graph refinement"""
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    processed_files = 0
    generated_graphs = []
    
    for root, dirs, files in os.walk(input_dir):
        for file in files:
            input_path = os.path.join(root, file)
            rel_path = os.path.relpath(root, input_dir)
            output_subdir = os.path.join(output_dir, rel_path)
            Path(output_subdir).mkdir(parents=True, exist_ok=True)
                
            output_file = os.path.join(output_subdir, f"{os.path.splitext(file)[0]}_kg.json")
                
            print(f"\n{'='*50}")
            print(f"Processing file {processed_files+1}: {input_path}")
            
            # Chiamata modificata per includere il parametro refine
            graph = generate_knowledge_graph(
                [input_path], 
                output_file, 
                model, 
                chunk_size,
                refine=refine
            )
            if graph:
                generated_graphs.append(graph)
            
            processed_files += 1
            print(f"{'='*50}")
    
    # Parte originale invariata
    if len(generated_graphs) > 1:
        aggregated_graph = aggregate_graphs(generated_graphs)
        aggregated_file = os.path.join(output_dir, "aggregated_kg.json")
        with open(aggregated_file, "w", encoding="utf-8") as f:
            json.dump(aggregated_graph, f, indent=2, ensure_ascii=False)
        print(f"\nAggregated knowledge graph saved to {aggregated_file}")
    
    print(f"\nProcessing complete! Processed {processed_files} files.")

if __name__ == "__main__":
    input_directory = r"C:\Users\giggi\Documents\LLM_KG\TESTFILES"
    output_directory = r"C:\Users\giggi\Documents\LLM_KG\knowledge_graphs_output"
    model_name = "hf.co/GPT4All-Community/Meta-Llama-3.1-8B-Instruct-128k-GGUF:latest"
    
    print("Starting batch processing with Generate-on-Graph...")
    process_all_files(
        input_directory, 
        output_directory, 
        model_name, 
        chunk_size=5000,
        refine=True  # <-- Nuovo parametro per attivare il refinement
    )