import os
import json
from datasets import load_dataset
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def load_xlam_function_calling_data(num_samples=None):
    """
    Load the xlam-function-calling-60k dataset using standard HuggingFace approach.
    
    Args:
        num_samples: Number of samples to load. If None, loads all samples.
        
    Returns:
        List of samples from the dataset
    """
    hf_token = os.getenv('HF_TOKEN')
    if not hf_token:
        raise ValueError("HF_TOKEN not found in environment variables. Please add it to your .env file.")
    
    print("Loading xlam-function-calling dataset...")
    
    # Load the dataset
    dataset = load_dataset(
        "Salesforce/xlam-function-calling-60k",
        token=hf_token,
        trust_remote_code=True
    )
    
    # Get the train split
    train_data = dataset['train']
    
    if num_samples:
        samples = train_data.select(range(min(num_samples, len(train_data))))
    else:
        samples = train_data
    
    print(f"Loaded {len(samples)} samples")
    return samples

def parse_sample(sample):
    """Parse a sample to extract structured data."""
    try:
        tools = json.loads(sample['tools']) if sample['tools'] else []
        answers = json.loads(sample['answers']) if sample['answers'] else []
    except json.JSONDecodeError as e:
        print(f"JSON parsing error: {e}")
        tools = []
        answers = []
    
    return {
        'id': sample['id'],
        'query': sample['query'],
        'tools': tools,
        'answers': answers
    }

def print_sample(sample, index=0):
    """Print a sample in a readable format."""
    print(f"\n=== Sample {index} ===")
    
    parsed = parse_sample(sample)
    
    print(f"\nID: {parsed['id']}")
    print(f"\nQUERY: {parsed['query']}")
    
    print(f"\nAVAILABLE TOOLS ({len(parsed['tools'])}):")
    for i, tool in enumerate(parsed['tools']):
        name = tool.get('name', 'Unknown')
        desc = tool.get('description', 'No description')
        print(f"  {i+1}. {name}: {desc}")
    
    print(f"\nEXPECTED FUNCTION CALLS ({len(parsed['answers'])}):")
    for i, answer in enumerate(parsed['answers']):
        name = answer.get('name', 'Unknown')
        args = answer.get('arguments', {})
        print(f"  {i+1}. {name}({args})")

# Test the loader
if __name__ == "__main__":
    try:
        # Load one sample
        samples = load_xlam_function_calling_data(num_samples=1)
        
        # Print the sample
        if len(samples) > 0:
            print_sample(samples[0], 0)
        
    except Exception as e:
        print(f"Error: {e}") 