"""
GSM8K Dataset Loader

This module provides functionality to load the GSM8K dataset from Hugging Face
and sample n items from the training set.
"""

import random
import re
from typing import List, Dict, Optional
from datasets import load_dataset
from dataset_loader import DatasetLoader


class GSM8KLoader(DatasetLoader):
    """
    A class to load and sample from the GSM8K dataset.
    Inherits from DatasetLoader to ensure pipeline compatibility.
    """
    
    DATASET_NAME = "gsm8k"
    
    def load_dataset(self) -> None:
        """
        Load the GSM8K dataset from Hugging Face.
        """
        print("Loading GSM8K dataset from Hugging Face...")
        
        # Load dataset
        self.dataset = load_dataset("openai/gsm8k", "main", cache_dir=self.cache_dir)
        
        # Extract train and test data
        self.train_data = self.dataset['train']
        self.test_data = self.dataset['test']
        
        print(f"Dataset loaded successfully!")
        print(f"Train size: {len(self.train_data)}")
        print(f"Test size: {len(self.test_data)}")
    
    def sample_train_data(self, n: int, seed: Optional[int] = None) -> List[Dict[str, str]]:
        """
        Sample n items from the training data.
        
        Args:
            n: Number of items to sample
            seed: Random seed for reproducibility
            
        Returns:
            List of dictionaries with keys: example_id, question, answer, final_answer
        """
        if self.train_data is None:
            raise ValueError("Dataset not loaded. Call load_dataset() first.")
        
        if n > len(self.train_data):
            raise ValueError(f"Requested {n} samples but only {len(self.train_data)} available in train set.")
        
        # Set random seed if provided
        if seed is not None:
            random.seed(seed)
        
        # Sample indices
        indices = random.sample(range(len(self.train_data)), n)
        
        # Extract samples and parse final answers
        samples = []
        for idx in indices:
            item = self.train_data[idx]
            final_answer = self.parse_final_answer(item['answer'])
            
            samples.append({
                'example_id': f"gsm8k_train_{idx}",
                'question': item['question'],
                'answer': item['answer'],
                'final_answer': final_answer
            })
        
        print(f"Sampled {n} items from training data.")
        return samples
    
    def sample_test_data(self, n: int, seed: Optional[int] = None) -> List[Dict[str, str]]:
        """
        Sample n items from the test data.
        
        Args:
            n: Number of items to sample
            seed: Random seed for reproducibility
            
        Returns:
            List of dictionaries with keys: example_id, question, answer, final_answer
        """
        if self.test_data is None:
            raise ValueError("Dataset not loaded. Call load_dataset() first.")
        
        if n > len(self.test_data):
            raise ValueError(f"Requested {n} samples but only {len(self.test_data)} available in test set.")
        
        # Set random seed if provided
        if seed is not None:
            random.seed(seed)
        
        # Sample indices
        indices = random.sample(range(len(self.test_data)), n)
        
        # Extract samples and parse final answers
        samples = []
        for idx in indices:
            item = self.test_data[idx]
            final_answer = self.parse_final_answer(item['answer'])
            
            samples.append({
                'example_id': f"gsm8k_test_{idx}",
                'question': item['question'],
                'answer': item['answer'],
                'final_answer': final_answer
            })
        
        print(f"Sampled {n} items from test data.")
        return samples
    
    @staticmethod
    def parse_final_answer(answer_text: str) -> Optional[str]:
        """
        Parse the final answer from the GSM8K answer format.
        
        The final answer is typically indicated by '#### number' at the end.
        
        Args:
            answer_text: The full answer text from GSM8K
            
        Returns:
            The final numerical answer as a string, or None if not found
        """
        # Look for the pattern #### followed by a number (possibly with decimals, commas, etc.)
        pattern = r'####\s*([0-9,.-]+)'
        match = re.search(pattern, answer_text)
        
        if match:
            # Clean up the number (remove commas, extra spaces)
            answer = match.group(1).strip()
            # Remove commas from numbers like "1,000"
            answer = answer.replace(',', '')
            return answer
        
        return None
    
    def get_all_train_data(self) -> List[Dict[str, str]]:
        """
        Get all training data items.
        
        Returns:
            List of all training items with parsed final answers
        """
        if self.train_data is None:
            raise ValueError("Dataset not loaded. Call load_dataset() first.")
        
        all_data = []
        for idx, item in enumerate(self.train_data):
            final_answer = self.parse_final_answer(item['answer'])
            all_data.append({
                'example_id': f"gsm8k_train_{idx}",
                'question': item['question'],
                'answer': item['answer'],
                'final_answer': final_answer
            })
        
        return all_data
    
    def get_all_test_data(self) -> List[Dict[str, str]]:
        """
        Get all test data items.
        
        Returns:
            List of all test items with parsed final answers
        """
        if self.test_data is None:
            raise ValueError("Dataset not loaded. Call load_dataset() first.")
        
        all_data = []
        for idx, item in enumerate(self.test_data):
            final_answer = self.parse_final_answer(item['answer'])
            all_data.append({
                'example_id': f"gsm8k_test_{idx}",
                'question': item['question'],
                'answer': item['answer'],
                'final_answer': final_answer
            })
        
        return all_data 