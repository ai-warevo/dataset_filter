#!/usr/bin/env python3
"""
Test script for ComfyUI Piercing Dataset Workflow
================================================

This script demonstrates how to use the ComfyUI workflow for generating
and validating piercing datasets.
"""

import os
import sys
import json
from pathlib import Path

# Add the project root to Python path
sys.path.insert(0, str(Path(__file__).parent))

from comfyui_workflow_piercing_dataset import ComfyUIPiercingWorkflow


def test_workflow():
    """Test the ComfyUI workflow functionality"""
    print("Testing ComfyUI Piercing Dataset Workflow")
    print("=" * 50)
    
    # Initialize workflow
    workflow = ComfyUIPiercingWorkflow()
    
    # Test 1: Load specifications
    print("1. Loading specifications...")
    specs = workflow.specifications
    print(f"   Character ID: {specs['character_id']}")
    print(f"   Piercing requirements: {len(specs['piercing_requirements'])}")
    
    for i, req in enumerate(specs['piercing_requirements']):
        print(f"   {i+1}. {req['artifact_id']} at {req['expected_location']}: {req['expected_count']} piece(s)")
    
    # Test 2: Create a single workflow
    print("\n2. Creating workflow for RIGHT_EYEBROW piercing...")
    right_eyebrow_spec = specs['piercing_requirements'][0]
    workflow_dict = workflow.create_piercing_generation_workflow(right_eyebrow_spec)
    
    print(f"   Generated workflow with {len(workflow_dict)} nodes")
    
    # Test 3: Generate dataset
    print("\n3. Generating dataset...")
    generated_files = workflow.generate_dataset("data/test_assets")
    print(f"   Generated {len(generated_files)} workflow files")
    
    # Test 4: Validate dataset (mock)
    print("\n4. Validating dataset...")
    validation_results = workflow.validate_dataset("data/test_assets")
    print(f"   Validation summary:")
    print(f"     Total images: {validation_results['summary']['total_images']}")
    print(f"     Valid images: {validation_results['summary']['valid_images']}")
    print(f"     Invalid images: {validation_results['summary']['invalid_images']}")
    
    print("\nWorkflow test completed successfully!")


if __name__ == "__main__":
    test_workflow()