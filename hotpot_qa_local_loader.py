import json
import os

def load_hotpot_qa_local(file_path, num_samples=None):
    """
    Load HotpotQA data from a local JSON file using the official format.
    
    Args:
        file_path: Path to the JSON file
        num_samples: Number of samples to load. If None, loads all samples.
        
    Returns:
        List of HotpotQA samples in the official format
    """
    print(f"Loading HotpotQA from local file: {file_path}")
    
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Process each example following the official format
    processed_samples = []
    for idx, example in enumerate(data):
        if num_samples and idx >= num_samples:
            break
            
        # Handle missing keys (like in test set)
        for k in ["answer", "type", "level"]:
            if k not in example.keys():
                example[k] = None

        if "supporting_facts" not in example.keys():
            example["supporting_facts"] = []

        # Format according to official structure
        processed_example = {
            "id": example["_id"],
            "question": example["question"],
            "answer": example["answer"],
            "type": example["type"],
            "level": example["level"],
            "supporting_facts": [{"title": f[0], "sent_id": f[1]} for f in example["supporting_facts"]],
            "context": [{"title": f[0], "sentences": f[1]} for f in example["context"]],
        }
        processed_samples.append(processed_example)
    
    print(f"Loaded {len(processed_samples)} samples from local HotpotQA file")
    return processed_samples

def print_sample(sample, index=0):
    """Print a sample in a readable format."""
    print(f"\n=== HotpotQA Sample {index} ===")
    
    print(f"\nID: {sample['id']}")
    print(f"TYPE: {sample['type']} | LEVEL: {sample['level']}")
    print(f"\nQUESTION: {sample['question']}")
    print(f"\nANSWER: {sample['answer']}")
    
    # Show supporting facts
    supporting_facts = sample['supporting_facts']
    print(f"\nSUPPORTING FACTS ({len(supporting_facts)}):")
    for i, fact in enumerate(supporting_facts):
        print(f"  {i+1}. {fact['title']} (sentence {fact['sent_id']})")
    
    # Show context paragraphs
    context = sample['context']
    print(f"\nCONTEXT ({len(context)} paragraphs):")
    for i, paragraph in enumerate(context):
        title = paragraph['title']
        sentences = paragraph['sentences']
        print(f"  [{i+1}] {title}: {len(sentences)} sentences")
        # Show first sentence as preview
        if sentences:
            preview = sentences[0][:100] + "..." if len(sentences[0]) > 100 else sentences[0]
            print(f"      \"{preview}\"")

def get_problem_text(sample):
    """Extract the question as problem text."""
    return sample['question']

def get_answer_text(sample):
    """Extract the answer text."""
    return sample['answer']

def get_supporting_context(sample):
    """Get the supporting sentences based on supporting_facts."""
    supporting_context = []
    
    # Create a lookup for context paragraphs by title
    context_lookup = {para['title']: para['sentences'] for para in sample['context']}
    
    # Get the actual supporting sentences
    for fact in sample['supporting_facts']:
        title = fact['title']
        sent_id = fact['sent_id']
        
        if title in context_lookup and sent_id < len(context_lookup[title]):
            sentence = context_lookup[title][sent_id]
            supporting_context.append({
                'title': title,
                'sent_id': sent_id,
                'sentence': sentence
            })
    
    return supporting_context

# Test the loader
if __name__ == "__main__":
    try:
        # Path to your local file
        file_path = "/Users/benigerisimon/Desktop/PhD/SRA/sharding_cot_sra/hotpot_qa/hotpot_train_v1.1.json"
        
        # Load one sample
        samples = load_hotpot_qa_local(file_path, num_samples=1)
        
        # Print the sample
        if samples:
            sample = samples[0]
            print_sample(sample, 0)
            
            # Show supporting context
            supporting_context = get_supporting_context(sample)
            print(f"\nSUPPORTING SENTENCES:")
            for i, ctx in enumerate(supporting_context):
                print(f"  {i+1}. [{ctx['title']}] {ctx['sentence']}")
        
    except Exception as e:
        print(f"Error: {e}") 