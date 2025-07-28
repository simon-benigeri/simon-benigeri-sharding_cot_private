#!/usr/bin/env python3
"""
Generate Reasoning Paths

Simple script to generate alternative reasoning paths from pipeline results.
Reads graphs, generates paths (exhaustive or sampling), saves results.
"""

import os
import json
import random
from typing import List, Dict, Tuple
from pathlib import Path
import networkx as nx
from tqdm import tqdm


def estimate_complexity(G: nx.DiGraph) -> Dict:
    """
    Estimate the computational complexity of enumerating all paths in a DAG.
    Uses constraint ratio and source/sink analysis for accurate prediction.
    """
    if not nx.is_directed_acyclic_graph(G):
        return {'complexity': 'ERROR', 'error': 'Graph contains cycles'}

    n_nodes = len(G.nodes)
    n_edges = len(G.edges)
    constraint_ratio = n_edges / max(n_nodes - 1, 1)

    sources = [n for n in G.nodes if G.in_degree(n) == 0]
    sinks = [n for n in G.nodes if G.out_degree(n) == 0]
    source_sink_pairs = len(sources) * len(sinks)

    if constraint_ratio >= 2.0:
        complexity = "LOW"
        estimated_paths = "1–10"
    elif constraint_ratio >= 1.0:
        complexity = "MEDIUM"
        estimated_paths = "10–1,000"
    elif constraint_ratio >= 0.5:
        complexity = "HIGH"
        estimated_paths = "1,000–100,000"
    else:
        complexity = "VERY HIGH"
        estimated_paths = "100,000+"

    return {
        'complexity': complexity,
        'estimated_paths': estimated_paths,
        'nodes': n_nodes,
        'edges': n_edges,
        'constraint_ratio': round(constraint_ratio, 2),
        'source_nodes': len(sources),
        'sink_nodes': len(sinks),
        'source_sink_pairs': source_sink_pairs,
        'use_sampling': complexity in ['HIGH', 'VERY HIGH']
    }


def build_graph(nodes: List[Dict], edges: List[List[str]]) -> Tuple[nx.DiGraph, bool]:
    """Build NetworkX graph. Returns (graph, is_valid)."""
    G = nx.DiGraph()
    
    # Add nodes
    for node in nodes:
        G.add_node(node['id'], text=node['text'])
    
    # Add valid edges
    for edge in edges:
        if len(edge) == 2 and edge[0] in G.nodes and edge[1] in G.nodes:
            G.add_edge(edge[0], edge[1])
    
    return G, nx.is_directed_acyclic_graph(G)


def generate_all_paths(G: nx.DiGraph, max_paths: int = 10000) -> List[List[str]]:
    """Generate all topological orderings."""
    paths = []
    count = 0
    
    try:
        for ordering in nx.all_topological_sorts(G):
            paths.append(list(ordering))
            count += 1
            if count >= max_paths:
                break
    except:
        return []
    
    return paths


def sample_paths(G: nx.DiGraph, n_samples: int = 500) -> List[List[str]]:
    """Generate random sample of topological orderings."""
    paths = []
    attempts = 0
    max_attempts = n_samples * 5
    
    while len(paths) < n_samples and attempts < max_attempts:
        attempts += 1
        
        # Random topological sort using Kahn's algorithm
        in_degree = dict(G.in_degree())
        queue = [node for node in G.nodes() if in_degree[node] == 0]
        random.shuffle(queue)
        
        path = []
        temp_in_degree = in_degree.copy()
        
        while queue:
            current = queue.pop(random.randint(0, len(queue) - 1))
            path.append(current)
            
            for neighbor in G.successors(current):
                temp_in_degree[neighbor] -= 1
                if temp_in_degree[neighbor] == 0:
                    queue.append(neighbor)
            
            random.shuffle(queue)
        
        if len(path) == len(G.nodes) and path not in paths:
            paths.append(path)
    
    return paths


def process_example(result: Dict, for_cots: bool = False) -> Dict:
    """Process single example: build graph and generate paths."""
    example_id = result.get('example_id', 'unknown')
    
    # Skip failed results
    if not result.get('success') or not result.get('nodes') or not result.get('edges'):
        return {'example_id': example_id, 'error': 'Invalid input'}
    
    # Build graph
    G, is_valid = build_graph(result['nodes'], result['edges'])
    if not is_valid:
        return {'example_id': example_id, 'error': 'Graph contains cycles'}
    
    # Estimate complexity using sophisticated analysis
    complexity_info = estimate_complexity(G)
    
    if 'error' in complexity_info:
        return {'example_id': example_id, 'error': complexity_info['error']}
    
    # Choose generation method based on complexity and purpose
    if for_cots:
        # Conservative limits for COT generation (since we only need ~10-50 alternatives)
        if complexity_info['use_sampling']:
            paths = sample_paths(G, 50)  # Much lower for COTs
            method = "sampling"
        else:
            complexity = complexity_info['complexity']
            max_paths = 100 if complexity == "MEDIUM" else 200  # Conservative limits
            paths = generate_all_paths(G, max_paths)
            method = "exhaustive"
    else:
        # Full analysis limits for research purposes
        if complexity_info['use_sampling']:
            paths = sample_paths(G, 500)
            method = "sampling"
        else:
            complexity = complexity_info['complexity']
            max_paths = 1000 if complexity == "MEDIUM" else 10000
            paths = generate_all_paths(G, max_paths)
            method = "exhaustive"
    
    # Create result with full complexity information
    return {
        'example_id': example_id,
        'complexity_estimate': complexity_info,
        'method': method,
        'total_paths': len(paths),
        'paths': [
            {
                'sequence': path,
                'texts': [G.nodes[node]['text'] for node in path]
            }
            for path in paths
        ]
    }


def generate_paths_for_file(input_file: str, output_file: str = None, for_cots: bool = False):
    """Generate paths for all examples in a pipeline results file."""
    # Determine output file
    if output_file is None:
        input_path = Path(input_file)
        output_file = input_path.parent / f"{input_path.stem}_reasoning_paths.json"
    
    # Load input
    with open(input_file, 'r') as f:
        pipeline_results = json.load(f)
    
    # Process examples
    results = []
    for result in tqdm(pipeline_results, desc="Processing examples"):
        processed = process_example(result, for_cots)
        results.append(processed)
    
    # Save results
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    # Summary
    successful = len([r for r in results if 'error' not in r])
    total_paths = sum(r.get('total_paths', 0) for r in results)
    
    mode = "COT mode (conservative limits)" if for_cots else "Full analysis mode"
    print(f"✅ Generated paths for {successful}/{len(results)} examples ({mode})")
    print(f"📁 Output: {output_file}")
    print(f"🔢 Total paths: {total_paths}")


def main():
    """CLI interface."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate reasoning paths from pipeline results")
    parser.add_argument("input_file", help="Pipeline results JSON file")
    parser.add_argument("--output", help="Output file (default: same directory)")
    parser.add_argument("--for-cots", action="store_true", 
                       help="Use conservative limits for COT generation (max 50-200 paths vs 500-10000)")
    
    args = parser.parse_args()
    
    generate_paths_for_file(args.input_file, args.output, args.for_cots)


if __name__ == "__main__":
    main() 