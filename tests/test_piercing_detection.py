#!/usr/bin/env python3
"""
Simple test script for piercing detection functionality
"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from piercing_benchmark import load_specification, get_piercing_counts, compare_with_specification

def test_basic_functionality():
    """Test basic functionality of the piercing detection system"""
    
    print("Testing Piercing Detection System")
    print("=" * 40)
    
    # Test 1: Load specification
    print("1. Loading specification...")
    spec = load_specification()
    print(f"   Loaded specification for character: {spec['character_id']}")
    print(f"   Piercing requirements: {len(spec['piercing_requirements'])}")
    
    # Test 2: Show specification details
    for req in spec["piercing_requirements"]:
        print(f"   - {req['expected_location']}: {req['expected_count']} x {req['artifact_id']}")
    
    # Test 3: Test get_piercing_counts function (mocked)
    print("\n2. Testing piercing counts function...")
    mock_counts = get_piercing_counts("test_image.png")
    print(f"   Mock detection results: {mock_counts}")
    
    # Test 4: Compare with specification
    print("\n3. Comparing with specification...")
    mismatches = compare_with_specification(mock_counts, spec)
    print(f"   Mismatches found: {len(mismatches)}")
    
    if mismatches:
        for mismatch in mismatches:
            print(f"   - {mismatch['location']}: expected {mismatch['expected']}, got {mismatch['detected']}")
    else:
        print("   All piercings match specification!")
    
    print("\nTest completed successfully!")

if __name__ == "__main__":
    test_basic_functionality()