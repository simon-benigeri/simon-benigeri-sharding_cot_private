#!/usr/bin/env python3
"""
Helper script demonstrating how to extract outputs from GSM8K prompts using XML-like tags.

This script shows the regex patterns to extract the desired content from each prompt's response.
"""

import re
import json


def extract_chain_of_thought(response_text: str) -> str:
    """
    Extract chain-of-thought reasoning from generate_cot.txt response.
    
    Args:
        response_text: The full model response
        
    Returns:
        Extracted chain-of-thought text or None if not found
    """
    pattern = r'<cot>\s*(.*?)\s*</cot>'
    match = re.search(pattern, response_text, re.DOTALL)
    
    if match:
        return match.group(1).strip()
    return None


def extract_segments_json(response_text: str) -> list:
    """
    Extract segmentation JSON from segment_cot.txt response.
    
    Args:
        response_text: The full model response
        
    Returns:
        Parsed JSON list or None if not found/invalid
    """
    pattern = r'<segments>\s*(.*?)\s*</segments>'
    match = re.search(pattern, response_text, re.DOTALL)
    
    if match:
        json_text = match.group(1).strip()
        try:
            return json.loads(json_text)
        except json.JSONDecodeError:
            print("Warning: Invalid JSON in segments response")
            return None
    return None


def extract_nodes_json(response_text: str) -> list:
    """
    Extract nodes JSON from create_graph.txt response.
    
    Args:
        response_text: The full model response
        
    Returns:
        Parsed JSON list or None if not found/invalid
    """
    pattern = r'<nodes>\s*(.*?)\s*</nodes>'
    match = re.search(pattern, response_text, re.DOTALL)
    
    if match:
        json_text = match.group(1).strip()
        try:
            return json.loads(json_text)
        except json.JSONDecodeError:
            print("Warning: Invalid JSON in nodes response")
            return None
    return None


def extract_edges_json(response_text: str) -> list:
    """
    Extract edges JSON from create_graph.txt response.
    
    Args:
        response_text: The full model response
        
    Returns:
        Parsed JSON list or None if not found/invalid
    """
    pattern = r'<edges>\s*(.*?)\s*</edges>'
    match = re.search(pattern, response_text, re.DOTALL)
    
    if match:
        json_text = match.group(1).strip()
        try:
            return json.loads(json_text)
        except json.JSONDecodeError:
            print("Warning: Invalid JSON in edges response")
            return None
    return None


def extract_graph_complete(response_text: str) -> dict:
    """
    Extract complete graph (nodes + edges) from create_graph.txt response.
    
    Args:
        response_text: The full model response
        
    Returns:
        Dict with 'nodes' and 'edges' keys, or None if extraction fails
    """
    nodes = extract_nodes_json(response_text)
    edges = extract_edges_json(response_text)
    
    if nodes is not None and edges is not None:
        return {
            "nodes": nodes,
            "edges": edges
        }
    return None


def filter_required_segments(segments: list) -> list:
    """
    Filter segments to only include required ones (is_required = 1).
    
    Args:
        segments: List of segment objects with 'is_required' field
        
    Returns:
        List containing only required segments
    """
    if not segments:
        return []
    
    return [seg for seg in segments if seg.get('is_required') == 1]


def filter_optional_segments(segments: list) -> list:
    """
    Filter segments to only include optional ones (is_required = 0).
    
    Args:
        segments: List of segment objects with 'is_required' field
        
    Returns:
        List containing only optional segments
    """
    if not segments:
        return []
    
    return [seg for seg in segments if seg.get('is_required') == 0]


def analyze_segments(segments: list) -> dict:
    """
    Analyze segments to provide statistics about required vs optional steps.
    
    Args:
        segments: List of segment objects with 'is_required' field
        
    Returns:
        Dict with analysis results
    """
    if not segments:
        return {"error": "No segments provided"}
    
    total_count = len(segments)
    required_segments = filter_required_segments(segments)
    optional_segments = filter_optional_segments(segments)
    
    return {
        "total_segments": total_count,
        "required_count": len(required_segments),
        "optional_count": len(optional_segments),
        "required_ratio": len(required_segments) / total_count if total_count > 0 else 0,
        "required_segments": [seg['segment'] for seg in required_segments],
        "optional_segments": [seg['segment'] for seg in optional_segments]
    }


def demo_extraction():
    """Demonstrate extraction with example responses."""
    
    # Example response from generate_cot prompt
    cot_response = """
    I need to solve this step by step.
    
    <cot>
    First, I need to find how many strawberries Matthew picked.
    Matthew picked 16 + 20 = 36 strawberries.
    Next, I need to find how many Natalie picked.
    Since Matthew picked twice as many as Natalie, Natalie picked 36/2 = 18 strawberries.
    Finally, the total is 16 + 36 + 18 = 70 strawberries.
    </cot>
    
    That's the complete reasoning.
    """
    
    # Example response from segment_cot prompt
    segments_response = """
    Here are the segments:
    
    <segments>
    [
      {"segment": "Matthew picked 16 + 20 = 36 strawberries", "is_required": 1},
      {"segment": "Natalie picked 36/2 = 18 strawberries", "is_required": 1},
      {"segment": "Total strawberries = 16 + 36 + 18 = 70", "is_required": 1}
    ]
    </segments>
    
    These are the atomic reasoning steps.
    """
    
    # Example response from create_graph prompt
    graph_response = """
    Here's the dependency graph:
    
    <nodes>
    [
      {"id": "s1", "text": "Matthew picked 16 + 20 = 36 strawberries"},
      {"id": "s2", "text": "Natalie picked 36/2 = 18 strawberries"},
      {"id": "s3", "text": "Total strawberries = 16 + 36 + 18 = 70"}
    ]
    </nodes>
    
    <edges>
    [
      ["s1", "s2"],
      ["s1", "s3"],
      ["s2", "s3"]
    ]
    </edges>
    
    This represents the logical dependencies.
    """
    
    print("=== Extraction Demo ===")
    
    # Test chain-of-thought extraction
    cot = extract_chain_of_thought(cot_response)
    print("Extracted Chain-of-Thought:")
    print(cot)
    print()
    
    # Test segments extraction
    segments = extract_segments_json(segments_response)
    print("Extracted Segments:")
    print(json.dumps(segments, indent=2))
    print()
    
    # Test segment analysis
    if segments:
        analysis = analyze_segments(segments)
        print("Segment Analysis:")
        print(f"  Total segments: {analysis['total_segments']}")
        print(f"  Required segments: {analysis['required_count']}")
        print(f"  Optional segments: {analysis['optional_count']}")
        print(f"  Required ratio: {analysis['required_ratio']:.2f}")
        print()
    
    # Test nodes extraction
    nodes = extract_nodes_json(graph_response)
    print("Extracted Nodes:")
    print(json.dumps(nodes, indent=2))
    print()
    
    # Test edges extraction
    edges = extract_edges_json(graph_response)
    print("Extracted Edges:")
    print(json.dumps(edges, indent=2))
    print()
    
    # Test complete graph extraction
    graph = extract_graph_complete(graph_response)
    print("Complete Graph:")
    print(json.dumps(graph, indent=2))


if __name__ == "__main__":
    print("GSM8K Prompt Output Extraction Patterns (XML Tags)")
    print("=" * 60)
    print()
    
    print("Regex Patterns:")
    print("1. Chain-of-Thought: r'<cot>\\s*(.*?)\\s*</cot>'")
    print("2. Segments JSON: r'<segments>\\s*(.*?)\\s*</segments>'") 
    print("3. Nodes JSON: r'<nodes>\\s*(.*?)\\s*</nodes>'")
    print("4. Edges JSON: r'<edges>\\s*(.*?)\\s*</edges>'")
    print()
    
    print("Enhanced Segment Features:")
    print("- Each segment now includes 'is_required' field (1=required, 0=optional)")
    print("- Maximum 15 segments per reasoning chain")
    print("- Helper functions: filter_required_segments(), filter_optional_segments(), analyze_segments()")
    print("- Atomic step requirement: one calculation/fact/decision per segment")
    print()
    
    demo_extraction() 