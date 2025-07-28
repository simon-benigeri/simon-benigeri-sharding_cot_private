#!/usr/bin/env python3
"""
Generalized Chain-of-Thought Reasoning Pipeline

This script runs the complete reasoning pipeline for any dataset:
1. Load data samples using provided dataset loader
2. Generate chain-of-thought reasoning (OpenAI API)
3. Segment the reasoning into atomic steps (OpenAI API) 
4. Filter required segments
5. Create dependency graph (OpenAI API)

Requires OpenAI API key in .env file.
Prompts should be organized as: prompts/{dataset_name}/{prompt_name}.txt
"""

import os
import re
import json
import time
from typing import List, Dict, Optional, Tuple, Any
from pathlib import Path
from openai import OpenAI
from dotenv import load_dotenv
from tqdm import tqdm

# Import the abstract base class and dataset loaders
from dataset_loader import DatasetLoader
from gsm8k_loader import GSM8KLoader

# Dataset mapping
DATASET_LOADERS = {
    'gsm8k': GSM8KLoader,
    # Add other datasets here as they become available
}


class ReasoningPipeline:
    """Generalized pipeline for processing reasoning problems through CoT generation, segmentation, and graph creation."""
    
    def __init__(self, dataset_name: str, model: str = "gpt-4o-mini-2024-07-18", delay: float = 1.0, 
                 segmentation_style: str = "regular"):
        """
        Initialize the pipeline.
        
        Args:
            dataset_name: Name of the dataset (used for prompt loading from prompts/{dataset_name}/)
            model: OpenAI model to use (default: gpt-4o-mini-2024-07-18)
            delay: Delay between API calls to avoid rate limits
            segmentation_style: "regular" for micro-segments, "consolidated" for substantial segments, 
                              "decontextualized" for self-contained segments, "direct" for direct answer segmentation,
                              "reasoning_only_decontextualized" for reasoning-only CoT segmentation,
                              "reasoning_only_direct" for reasoning-only solution segmentation
        """
        # Load environment variables
        load_dotenv()
        
        # Set up OpenAI client
        api_key = os.getenv('OPENAI_API_KEY')
        if not api_key:
            raise ValueError("OPENAI_API_KEY not found in environment variables. Please add it to .env file.")
        
        self.client = OpenAI(api_key=api_key)
        self.dataset_name = dataset_name
        self.model = model
        self.delay = delay
        self.segmentation_style = segmentation_style
        
        # Load prompts
        self.prompts = self._load_prompts()
        
        print(f"Pipeline initialized for dataset: {dataset_name}, model: {model}, segmentation: {segmentation_style}")
    
    def _load_prompts(self) -> Dict[str, str]:
        """Load all prompt templates from dataset-specific directory."""
        prompt_dir = Path(f"prompts/{self.dataset_name}")
        prompts = {}
        
        # Choose segmentation and graph prompts based on style
        if self.segmentation_style == "consolidated":
            segment_file = "segment_cot_consolidated.txt"
            graph_file = "create_graph.txt"
        elif self.segmentation_style == "decontextualized":
            segment_file = "segment_cot_decontextualized.txt"
            graph_file = "create_graph_decontextualized.txt"
        elif self.segmentation_style == "direct":
            segment_file = "segment_cot_direct.txt"
            graph_file = "create_graph_direct.txt"
        elif self.segmentation_style == "reasoning_only_decontextualized":
            segment_file = "segment_cot_reasoning_only_decontextualized.txt"
            graph_file = "create_graph_decontextualized.txt"
        elif self.segmentation_style == "reasoning_only_direct":
            segment_file = "segment_cot_reasoning_only_direct.txt"
            graph_file = "create_graph_direct.txt"
        else:  # regular
            segment_file = "segment_cot.txt"
            graph_file = "create_graph.txt"
        
        prompt_files = [
            ("generate_cot.txt", "generate_cot"),
            (segment_file, "segment_cot"), 
            (graph_file, "create_graph")
        ]
        
        for filename, key in prompt_files:
            filepath = prompt_dir / filename
            if filepath.exists():
                with open(filepath, 'r', encoding='utf-8') as f:
                    prompts[key] = f.read()
            else:
                raise FileNotFoundError(f"Prompt file not found: {filepath}")
        
        return prompts
    
    def _parse_prompt(self, prompt_text: str) -> Tuple[str, str]:
        """
        Parse prompt text to separate system and user messages.
        
        Args:
            prompt_text: Full prompt text with optional separator
            
        Returns:
            Tuple of (system_message, user_message)
        """
        separator = "---USER---"
        
        if separator in prompt_text:
            parts = prompt_text.split(separator, 1)
            system_message = parts[0].strip()
            user_message = parts[1].strip()
        else:
            # If no separator, treat entire prompt as user message with default system
            system_message = "You are a helpful assistant that follows instructions precisely."
            user_message = prompt_text.strip()
        
        return system_message, user_message
    
    def _call_openai(self, prompt: str, max_tokens: int = 1500) -> str:
        """
        Call OpenAI API with retry logic using system/user message structure.
        
        Args:
            prompt: The prompt to send (may contain system/user separator)
            max_tokens: Maximum tokens to generate
            
        Returns:
            The response text
        """
        # Parse system and user messages
        system_message, user_message = self._parse_prompt(prompt)
        
        max_retries = 3
        for attempt in range(max_retries):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": system_message},
                        {"role": "user", "content": user_message}
                    ],
                    max_tokens=max_tokens,
                    temperature=0.1  # Low temperature for consistency
                )
                
                result = response.choices[0].message.content.strip()
                
                # Add delay to avoid rate limiting
                time.sleep(self.delay)
                
                return result
                
            except Exception as e:
                print(f"API call failed (attempt {attempt + 1}/{max_retries}): {e}")
                if attempt == max_retries - 1:
                    raise
                time.sleep(2 ** attempt)  # Exponential backoff
    
    def extract_cot(self, response: str) -> Optional[str]:
        """Extract chain-of-thought from generate_cot response."""
        pattern = r'<cot>\s*(.*?)\s*</cot>'
        match = re.search(pattern, response, re.DOTALL)
        
        if match:
            return match.group(1).strip()
        
        print("Warning: No <cot> tags found in response")
        return None
    
    def extract_segments(self, response: str) -> Optional[List[Dict]]:
        """Extract segments JSON from segment_cot response."""
        pattern = r'<segments>\s*(.*?)\s*</segments>'
        match = re.search(pattern, response, re.DOTALL)
        
        if match:
            json_text = match.group(1).strip()
            try:
                return json.loads(json_text)
            except json.JSONDecodeError as e:
                print(f"Warning: Invalid JSON in segments response: {e}")
                return None
        
        print("Warning: No <segments> tags found in response")
        return None
    
    def extract_nodes(self, response: str) -> Optional[List[Dict]]:
        """Extract nodes JSON from create_graph response."""
        pattern = r'<nodes>\s*(.*?)\s*</nodes>'
        match = re.search(pattern, response, re.DOTALL)
        
        if match:
            json_text = match.group(1).strip()
            try:
                return json.loads(json_text)
            except json.JSONDecodeError as e:
                print(f"Warning: Invalid JSON in nodes response: {e}")
                return None
        
        print("Warning: No <nodes> tags found in response")
        return None
    
    def extract_edges(self, response: str) -> Optional[List[List[str]]]:
        """Extract edges JSON from create_graph response."""
        pattern = r'<edges>\s*(.*?)\s*</edges>'
        match = re.search(pattern, response, re.DOTALL)
        
        if match:
            json_text = match.group(1).strip()
            try:
                return json.loads(json_text)
            except json.JSONDecodeError as e:
                print(f"Warning: Invalid JSON in edges response: {e}")
                return None
        
        print("Warning: No <edges> tags found in response")
        return None
    
    def filter_required_segments(self, segments: List[Dict]) -> List[Dict]:
        """Filter segments to only include required ones (is_required = 1)."""
        if not segments:
            return []
        
        required = [seg for seg in segments if seg.get('is_required') == 1]
        return required
    
    def generate_cot(self, question: str, answer: str, final_answer: str) -> Optional[str]:
        """
        Step 1: Generate chain-of-thought reasoning.
        
        Args:
            question: The problem/question
            answer: The complete step-by-step answer
            final_answer: The final answer
            
        Returns:
            Generated chain-of-thought text
        """        
        prompt = self.prompts['generate_cot'].format(
            question=question,
            answer=answer,
            final_answer=final_answer
        )
        
        response = self._call_openai(prompt)
        cot = self.extract_cot(response)
        
        return cot
    
    def segment_cot(self, question: str, answer: str, final_answer: str, cot: str) -> Optional[List[Dict]]:
        """
        Step 2: Segment chain-of-thought into atomic reasoning steps.
        
        Args:
            question: The problem/question
            answer: The complete step-by-step answer
            final_answer: The final answer
            cot: The chain-of-thought reasoning
            
        Returns:
            List of segment objects with is_required field
        """
        prompt = self.prompts['segment_cot'].format(
            question=question,
            answer=answer,
            final_answer=final_answer,
            cot_output_from_prompt_1=cot
        )
        
        response = self._call_openai(prompt, max_tokens=2000)
        segments = self.extract_segments(response)
        
        return segments
    
    def create_graph(self, question: str, answer: str, final_answer: str, cot: str, segments: List[Dict]) -> Optional[Tuple[List[Dict], List[List[str]]]]:
        """
        Step 3: Create dependency graph from required segments.
        
        Args:
            question: The problem/question
            answer: The complete step-by-step answer
            final_answer: The final answer
            cot: The chain-of-thought reasoning
            segments: The segmented reasoning steps
            
        Returns:
            Tuple of (nodes, edges) if successful
        """
        prompt = self.prompts['create_graph'].format(
            question=question,
            answer=answer,
            final_answer=final_answer,
            cot_output_from_prompt_1=cot,
            segments_from_prompt_2=json.dumps(segments, indent=2)
        )
        
        response = self._call_openai(prompt, max_tokens=2000)
        nodes = self.extract_nodes(response)
        edges = self.extract_edges(response)
        
        if nodes and edges:
            return (nodes, edges)
        else:
            return None
    
    def process_sample(self, sample: Dict) -> Dict:
        """
        Process a single sample through the complete pipeline.
        
        Args:
            sample: Sample with example_id, question, answer, final_answer fields
            
        Returns:
            Dictionary with all pipeline outputs
        """
        example_id = sample['example_id']
        question = sample['question']
        answer = sample['answer']
        final_answer = sample['final_answer']
        
        # Removed verbose processing header
        
        result = {
            'example_id': example_id,
            'question': question,
            'answer': answer,
            'final_answer': final_answer,
            'cot': None,
            'segments': None,
            'required_segments': None,
            'nodes': None,
            'edges': None,
            'success': False
        }
        
        try:
            # Step 1: Generate CoT (skip if using direct segmentation)
            if self.segmentation_style == "direct" or self.segmentation_style == "reasoning_only_direct":
                # Use answer directly as CoT for direct segmentation
                cot = answer
                result['cot'] = cot
                result['pipeline_type'] = 'direct_answer_segmentation'
            else:
                # Generate new CoT
                cot = self.generate_cot(question, answer, final_answer)
                if not cot:
                    return result
                result['cot'] = cot
                result['pipeline_type'] = 'normal_with_cot_generation'
            
            # Step 2: Segment CoT
            segments = self.segment_cot(question, answer, final_answer, cot)
            if not segments:
                return result
            result['segments'] = segments
            
            # Filter required segments
            required_segments = self.filter_required_segments(segments)
            result['required_segments'] = required_segments
            
            # Step 3: Create graph
            graph_result = self.create_graph(question, answer, final_answer, cot, segments)
            if not graph_result:
                return result
            
            nodes, edges = graph_result
            result['nodes'] = nodes
            result['edges'] = edges
            result['success'] = True
            
        except Exception as e:
            print(f"❌ Pipeline failed: {e}")
            import traceback
            traceback.print_exc()  # Print full traceback
            result['error'] = str(e)
        
        return result
    
    def process_sample_direct(self, sample: Dict[str, str]) -> Dict[str, Any]:
        """
        Process a single sample using the existing answer as CoT (skips CoT generation).
        
        Args:
            sample: Dictionary with keys: example_id, question, answer, final_answer
            
        Returns:
            Dictionary with pipeline results
        """
        example_id = sample['example_id']
        question = sample['question']
        answer = sample['answer']  # This becomes our "cot"
        final_answer = sample['final_answer']
        
        result = {
            'example_id': example_id,
            'question': question,
            'answer': answer,
            'final_answer': final_answer,
            'cot': answer,  # Use answer as the cot
            'segments': None,
            'required_segments': None,
            'nodes': None,
            'edges': None,
            'success': False,
            'pipeline_type': 'direct_answer_segmentation'
        }
        
        try:
            # Step 1: Segment the answer directly (no CoT generation)
            segments = self.segment_cot(question, answer, final_answer, answer)  # Pass answer as cot
            if not segments:
                return result
            result['segments'] = segments
            
            # Filter required segments
            required_segments = self.filter_required_segments(segments)
            result['required_segments'] = required_segments
            
            # Step 2: Create graph
            graph_result = self.create_graph(question, answer, final_answer, answer, segments)  # Pass answer as cot
            if not graph_result:
                return result
            
            nodes, edges = graph_result
            result['nodes'] = nodes
            result['edges'] = edges
            result['success'] = True
            
        except Exception as e:
            print(f"❌ Direct pipeline failed: {e}")
            import traceback
            traceback.print_exc()
            
        return result
    
    def run_batch(self, samples: List[Dict], output_file: str = None) -> List[Dict]:
        """
        Run pipeline on a batch of samples.
        
        Args:
            samples: List of samples with example_id, question, answer, final_answer fields
            output_file: File to save results (defaults to {dataset_name}_pipeline_results.json)
            
        Returns:
            List of results
        """
        if output_file is None:
            # Create directory structure: sharded_data/dataset/
            import os
            output_dir = f"sharded_data/{self.dataset_name}"
            os.makedirs(output_dir, exist_ok=True)
            
            # Include segmentation style in filename
            output_file = f"{output_dir}/{self.dataset_name}_pipeline_results_{self.segmentation_style}.json"
        
        print(f"📝 Processing {len(samples)} samples → {output_file}")
        
        results = []
        successful = 0
        
        for sample in tqdm(samples, desc="Processing samples"):
            result = self.process_sample(sample)
            results.append(result)
            
            if result['success']:
                successful += 1
            
            # Save intermediate results every 5 samples
            if len(results) % 5 == 0 or len(results) == len(samples):
                with open(output_file, 'w', encoding='utf-8') as f:
                    json.dump(results, f, indent=2, ensure_ascii=False)
                tqdm.write(f"💾 Saved ({successful}/{len(results)} successful)")
        
        print(f"🎯 Complete: {successful}/{len(samples)} successful ({successful/len(samples)*100:.1f}%)")
        
        return results

    def run_batch_direct(self, samples: List[Dict[str, str]], output_file: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Run the direct answer segmentation pipeline on a batch of samples.
        
        Args:
            samples: List of samples to process
            output_file: Optional output file path for results
            
        Returns:
            List of pipeline results
        """
        if not output_file:
            timestamp = int(time.time())
            output_file = f"sharded_data/{self.dataset_name}/{self.dataset_name}_direct_pipeline_results_{self.segmentation_style}_{timestamp}.json"
            
        # Ensure output directory exists
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        print(f"🚀 Running DIRECT pipeline: {len(samples)} samples")
        print(f"📁 Results: {output_file}")
        
        results = []
        successful = 0
        
        for sample in tqdm(samples, desc="Processing samples (direct)"):
            result = self.process_sample_direct(sample)
            results.append(result)
            
            if result['success']:
                successful += 1
            
            # Save intermediate results every 5 samples
            if len(results) % 5 == 0 or len(results) == len(samples):
                with open(output_file, 'w', encoding='utf-8') as f:
                    json.dump(results, f, indent=2, ensure_ascii=False)
                tqdm.write(f"💾 Saved ({successful}/{len(results)} successful)")
        
        print(f"🎯 Complete: {successful}/{len(samples)} successful ({successful/len(samples)*100:.1f}%)")
        
        return results


def run_pipeline(dataset: str, n_samples: int = 3, 
                model: str = "gpt-4o-mini-2024-07-18", delay: float = 1.0, seed: int = 42,
                segmentation_style: str = "regular") -> List[Dict]:
    """
    Convenience function to run the complete pipeline.
    
    Args:
        dataset: Dataset name (e.g., 'gsm8k')
        n_samples: Number of samples to process
        model: OpenAI model to use
        delay: Delay between API calls
        seed: Random seed for sampling
        segmentation_style: "regular" for micro-segments, "consolidated" for substantial segments, 
                          "decontextualized" for self-contained segments, "direct" for direct answer segmentation,
                          "reasoning_only_decontextualized" for reasoning-only CoT segmentation,
                          "reasoning_only_direct" for reasoning-only solution segmentation
        
    Returns:
        List of pipeline results (each with example_id)
    """
    # Get loader class from mapping
    if dataset not in DATASET_LOADERS:
        available = list(DATASET_LOADERS.keys())
        raise ValueError(f"Dataset '{dataset}' not supported. Available datasets: {available}")
    
    # Initialize loader
    loader_class = DATASET_LOADERS[dataset]
    loader = loader_class()
    print(f"📊 Loading {dataset} dataset...")
    loader.load_dataset()
    
    # Sample data
    samples = loader.sample_train_data(n=n_samples, seed=seed)
    
    # Initialize pipeline
    pipeline = ReasoningPipeline(dataset_name=dataset, model=model, delay=delay, 
                               segmentation_style=segmentation_style)
    
    # Run pipeline
    results = pipeline.run_batch(samples)
    
    # Print summary
    successful = sum(1 for r in results if r['success'])
    print(f"✅ Final: {successful}/{len(results)} successful ({successful/len(results)*100:.1f}%)")
    
    return results


def run_direct_pipeline(dataset: str, n_samples: int = 3, 
                       model: str = "gpt-4o-mini-2024-07-18", delay: float = 1.0, seed: int = 42,
                       segmentation_style: str = "regular") -> List[Dict]:
    """
    Convenience function to run the direct answer segmentation pipeline.
    
    Args:
        dataset: Dataset name (e.g., 'gsm8k')
        n_samples: Number of samples to process
        model: OpenAI model to use
        delay: Delay between API calls
        seed: Random seed for sampling
        segmentation_style: "regular", "consolidated", "decontextualized", "reasoning_only_decontextualized", 
                          or "reasoning_only_direct" for segments
        
    Returns:
        List of pipeline results (each with example_id)
    """
    # Get loader class from mapping
    if dataset not in DATASET_LOADERS:
        available = list(DATASET_LOADERS.keys())
        raise ValueError(f"Dataset '{dataset}' not supported. Available datasets: {available}")
    
    # Initialize loader
    loader_class = DATASET_LOADERS[dataset]
    loader = loader_class()
    print(f"📊 Loading {dataset} dataset...")
    loader.load_dataset()
    
    # Sample data
    samples = loader.sample_train_data(n=n_samples, seed=seed)
    
    # Initialize pipeline
    pipeline = ReasoningPipeline(dataset_name=dataset, model=model, delay=delay, 
                               segmentation_style=segmentation_style)
    
    # Run direct pipeline
    results = pipeline.run_batch_direct(samples)
    
    # Print summary
    successful = sum(1 for r in results if r['success'])
    print(f"✅ Final: {successful}/{len(results)} successful ({successful/len(results)*100:.1f}%)")
    
    return results


def list_datasets():
    """List available datasets."""
    print("Available datasets:")
    for dataset_name in DATASET_LOADERS.keys():
        print(f"  - {dataset_name}")
    return list(DATASET_LOADERS.keys())


def main():
    """Main function with argument parser for command-line usage."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Run reasoning pipeline on different datasets")
    parser.add_argument("--dataset", type=str, default="gsm8k", 
                       help=f"Dataset to use (available: {list(DATASET_LOADERS.keys())})")
    parser.add_argument("--n_samples", type=int, default=3,
                       help="Number of samples to process")
    parser.add_argument("--model", type=str, default="gpt-4o-mini-2024-07-18",
                       help="OpenAI model to use")
    parser.add_argument("--delay", type=float, default=1.0,
                       help="Delay between API calls (seconds)")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed for sampling")
    parser.add_argument("--segmentation", type=str, default="regular", 
                       choices=["regular", "consolidated", "decontextualized", "direct", "reasoning_only_decontextualized", "reasoning_only_direct"],
                       help="Segmentation style: 'regular' for micro-segments, 'consolidated' for substantial segments, 'decontextualized' for self-contained segments, 'direct' for direct answer segmentation, 'reasoning_only_decontextualized' for reasoning-only CoT segmentation, 'reasoning_only_direct' for reasoning-only solution segmentation")
    parser.add_argument("--pipeline", type=str, default="normal",
                       choices=["normal", "direct"],
                       help="Pipeline type: 'normal' generates CoT then segments, 'direct' segments existing answer")
    parser.add_argument("--list", action="store_true",
                       help="List available datasets and exit")
    
    args = parser.parse_args()
    
    # List datasets if requested
    if args.list:
        list_datasets()
        return
    
    # Validate dataset
    if args.dataset not in DATASET_LOADERS:
        print(f"❌ Error: Dataset '{args.dataset}' not supported.")
        list_datasets()
        return
    
    print(f"🚀 {args.dataset} | {args.n_samples} samples | {args.segmentation} segmentation | {args.pipeline} pipeline")
    
    # Run pipeline
    if args.pipeline == "direct":
        results = run_direct_pipeline(
            dataset=args.dataset,
            n_samples=args.n_samples,
            model=args.model,
            delay=args.delay,
            seed=args.seed,
            segmentation_style=args.segmentation
        )
    else:
        results = run_pipeline(
            dataset=args.dataset,
            n_samples=args.n_samples,
            model=args.model,
            delay=args.delay,
            seed=args.seed,
            segmentation_style=args.segmentation
        )
    
    print("\n✅ Pipeline completed!")


if __name__ == "__main__":
    main() 