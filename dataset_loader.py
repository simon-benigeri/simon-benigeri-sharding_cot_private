"""
Abstract Dataset Loader Interface

This module provides the base interface for dataset loaders to ensure
consistency across different datasets in the reasoning pipeline.
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Optional, Any


class DatasetLoader(ABC):
    """
    Abstract base class for dataset loaders.
    
    All dataset loaders should inherit from this class and implement
    the required methods to ensure compatibility with the reasoning pipeline.
    """
    
    # Override this in subclasses to specify the expected dataset name
    DATASET_NAME: str = None
    
    def __init__(self, cache_dir: Optional[str] = None):
        """
        Initialize the dataset loader.
        
        Args:
            cache_dir: Optional directory to cache the dataset
        """
        self.cache_dir = cache_dir
        self.dataset = None
        self.train_data = None
        self.test_data = None
    
    def get_dataset_name(self) -> str:
        """
        Get the expected dataset name for this loader.
        
        Returns:
            The dataset name that should be used with this loader
        """
        if self.DATASET_NAME is None:
            raise NotImplementedError("Subclasses must define DATASET_NAME class variable")
        return self.DATASET_NAME
    
    @abstractmethod
    def load_dataset(self) -> None:
        """
        Load the dataset from its source (e.g., Hugging Face, local files).
        
        This method should populate self.dataset, self.train_data, and self.test_data.
        """
        pass
    
    @abstractmethod
    def sample_train_data(self, n: int, seed: Optional[int] = None) -> List[Dict[str, str]]:
        """
        Sample n items from the training data.
        
        Args:
            n: Number of items to sample
            seed: Random seed for reproducibility
            
        Returns:
            List of dictionaries with required fields:
            - example_id: Unique identifier for the example
            - question: The problem/question text
            - answer: The full solution/answer text
            - final_answer: The final numerical/categorical answer
        """
        pass
    
    @abstractmethod
    def sample_test_data(self, n: int, seed: Optional[int] = None) -> List[Dict[str, str]]:
        """
        Sample n items from the test data.
        
        Args:
            n: Number of items to sample
            seed: Random seed for reproducibility
            
        Returns:
            List of dictionaries with required fields:
            - example_id: Unique identifier for the example
            - question: The problem/question text
            - answer: The full solution/answer text  
            - final_answer: The final numerical/categorical answer
        """
        pass
    
    @staticmethod
    @abstractmethod
    def parse_final_answer(answer_text: str) -> Optional[str]:
        """
        Parse the final answer from the dataset's answer format.
        
        Each dataset may have a different format for encoding the final answer.
        This method should extract just the final answer value.
        
        Args:
            answer_text: The full answer text from the dataset
            
        Returns:
            The final answer as a string, or None if not found
        """
        pass
    
    def get_train_size(self) -> int:
        """Get the size of the training set."""
        if self.train_data is None:
            raise ValueError("Dataset not loaded. Call load_dataset() first.")
        return len(self.train_data)
    
    def get_test_size(self) -> int:
        """Get the size of the test set."""
        if self.test_data is None:
            raise ValueError("Dataset not loaded. Call load_dataset() first.")
        return len(self.test_data)
    
    def save_samples(self, samples: List[Dict], filename: str, format: str = 'json') -> None:
        """
        Save samples to a file.
        
        Args:
            samples: List of sample dictionaries
            filename: Output filename
            format: Output format ('json', 'jsonl', 'csv')
        """
        import json
        import pandas as pd
        from pathlib import Path
        
        if format == 'json':
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(samples, f, indent=2, ensure_ascii=False)
        elif format == 'jsonl':
            with open(filename, 'w', encoding='utf-8') as f:
                for sample in samples:
                    f.write(json.dumps(sample, ensure_ascii=False) + '\n')
        elif format == 'csv':
            df = pd.DataFrame(samples)
            df.to_csv(filename, index=False, encoding='utf-8')
        else:
            raise ValueError(f"Unsupported format: {format}")
        
        print(f"Saved {len(samples)} samples to {filename}") 