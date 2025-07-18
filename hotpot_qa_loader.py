import os
from datasets import load_dataset
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def load_hotpot_qa_data(config="distractor", num_samples=None):
    """
    Load the HotpotQA dataset using standard HuggingFace approach.
    
    Args:
        config: Dataset configuration - "distractor" or "fullwiki" (default: "distractor")
        num_samples: Number of samples to load. If None, loads all samples.
        
    Returns:
        Dataset samples from HotpotQA
    """
    print(f"Loading HotpotQA dataset (config: {config})...")
    
    # Load the dataset
    dataset = load_dataset(
        "hotpotqa/hotpot_qa",
        config,
        trust_remote_code=True
    )
    
    # Get the validation split (more manageable for exploration)
    validation_data = dataset['validation']
    
    if num_samples:
        samples = validation_data.select(range(min(num_samples, len(validation_data))))
    else:
        samples = validation_data
    
    print(f"Loaded {len(samples)} samples from HotpotQA ({config})")
    return samples

def print_sample(sample, index=0):
    """Print a sample in a readable format."""
    print(f"\n=== HotpotQA Sample {index} ===")
    
    print(f"\nID: {sample['id']}")
    print(f"TYPE: {sample['type']} | LEVEL: {sample['level']}")
    print(f"\nQUESTION: {sample['question']}")
    print(f"\nANSWER: {sample['answer']}")
    
    # Show supporting facts
    supporting_facts = sample['supporting_facts']
    print(f"\nSUPPORTING FACTS ({len(supporting_facts['title'])}):")
    for i, (title, sent_id) in enumerate(zip(supporting_facts['title'], supporting_facts['sent_id'])):
        print(f"  {i+1}. {title} (sentence {sent_id})")
    
    # Show context paragraphs
    context = sample['context']
    print(f"\nCONTEXT ({len(context['title'])} paragraphs):")
    for i, (title, sentences) in enumerate(zip(context['title'], context['sentences'])):
        print(f"  [{i+1}] {title}: {len(sentences)} sentences")
        # Show first sentence as preview
        if sentences:
            preview = sentences[0][:100] + "..." if len(sentences[0]) > 100 else sentences[0]
            print(f"      \"{preview}\"")

def parse_sample(sample):
    """Parse a sample to extract key information."""
    return {
        'id': sample['id'],
        'question': sample['question'], 
        'answer': sample['answer'],
        'type': sample['type'],
        'level': sample['level'],
        'supporting_facts': sample['supporting_facts'],
        'context': sample['context']
    }

# Test the loader
if __name__ == "__main__":
    try:
        # Load one sample from distractor config
        samples = load_hotpot_qa_data(config="distractor", num_samples=1)
        
        # Print the sample
        if len(samples) > 0:
            print_sample(samples[0], 0)
        
    except Exception as e:
        print(f"Error: {e}") 