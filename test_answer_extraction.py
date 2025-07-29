#!/usr/bin/env python3
"""
Test script for answer extraction logic.
"""

import re

# Answer extraction regex - matches numbers after ####
ANS_RE = re.compile(r"#### (\-?[0-9,]+\.?[0-9]*)\s*$", re.MULTILINE)
INVALID_ANS = "[invalid]"


def extract_final_answer(text: str) -> str:
    """
    Extract final numerical answer from text using #### pattern.
    
    Args:
        text: Text containing answer (e.g., "Step 1...\n#### 42")
        
    Returns:
        Final numerical answer as string
    """
    match = ANS_RE.search(text)
    if match:
        answer = match.group(1).strip().replace(',', '')
        return answer
    return INVALID_ANS


def normalize_number(num_str: str) -> str:
    """
    Normalize number string for comparison.
    
    Args:
        num_str: Number as string
        
    Returns:
        Normalized number string
    """
    if not num_str or num_str == INVALID_ANS:
        return ""
    
    # Remove commas and spaces
    normalized = num_str.replace(',', '').replace(' ', '').strip()
    
    # Try to convert to float and back to handle different formats
    try:
        num = float(normalized)
        
        # Handle special cases
        if num == 0:
            return "0"
        
        # Handle integers vs floats
        if num.is_integer():
            return str(int(num))
        else:
            # For decimals, limit to reasonable precision
            return f"{num:.6f}".rstrip('0').rstrip('.')
            
    except (ValueError, TypeError):
        # If we can't convert to float, return empty string
        return ""


def test_extraction():
    """Test various answer extraction scenarios."""
    
    test_cases = [
        # Good cases - should extract numbers after ####
        ("Let me solve this step by step...\n#### 42", "42"),
        ("The answer is 42\n#### 42", "42"),
        ("Step 1: 2+2=4\nStep 2: 4*2=8\n#### 8", "8"),
        ("The result is 3.14\n#### 3.14", "3.14"),
        ("Answer: -5\n#### -5", "-5"),
        ("Final answer: 1,000\n#### 1,000", "1000"),  # Commas removed in normalize
        
        # Edge cases that should work
        ("#### 0", "0"),
        ("#### -0", "0"),  # -0 becomes 0 in normalize
        ("#### 3.14159", "3.14159"),
        ("#### 1,234,567", "1234567"),  # Commas removed
        
        # Cases with no #### - should return INVALID_ANS
        ("The answer is 42", ""),  # No #### found -> INVALID_ANS -> ""
        ("No numbers here", ""),
        ("The answer is ####", ""),  # No number after ####
        
        # Cases that don't match the regex pattern
        ("#### abc", ""),  # Not a number -> INVALID_ANS -> ""
        ("#### 42abc", ""),  # Has letters -> INVALID_ANS -> ""
        ("#### abc42", ""),  # Starts with letters -> INVALID_ANS -> ""
        
        # Empty cases
        ("", ""),
    ]
    
    print("🧪 Testing Answer Extraction Logic")
    print("=" * 50)
    
    passed = 0
    failed = 0
    
    for i, (input_text, expected) in enumerate(test_cases, 1):
        result = extract_final_answer(input_text)
        normalized_result = normalize_number(result)
        
        if normalized_result == expected:
            status = "✅ PASS"
            passed += 1
        else:
            status = "❌ FAIL"
            failed += 1
        
        print(f"{i:2d}. {status}")
        print(f"    Input:     {repr(input_text)}")
        print(f"    Expected:  {repr(expected)}")
        print(f"    Raw:       {repr(result)}")
        print(f"    Normalized: {repr(normalized_result)}")
        print()
    
    print(f"Results: {passed} passed, {failed} failed")
    
    if failed == 0:
        print("🎉 All tests passed!")
    else:
        print("⚠️  Some tests failed - check the logic!")


if __name__ == "__main__":
    test_extraction() 