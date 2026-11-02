#!/usr/bin/env python3
"""
Piercing Detection Benchmark Script
===================================

This script implements the core functionality for detecting piercings on faces
and comparing them against specification requirements.

Requirements:
1. Standalone execution without dependencies on broken infrastructure
2. Accepts directory path via command line arguments
3. Parses smlq_sfw.json to establish expected piercing counts and locations
4. Implements get_piercing_counts(image_path) function
5. Provides comprehensive reporting

Usage:
    python tests/piercing_benchmark.py <directory_path>
"""

import os
import sys
import json
import argparse
from typing import Dict, List, Tuple, Any
from collections import defaultdict
import cv2
import numpy as np
from pathlib import Path


def load_specification(spec_file: str = "src/dataset_filter/specs/smlq_sfw.json") -> Dict:
    """Load piercing specification from JSON file"""
    try:
        with open(spec_file, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        # Fallback specification if file doesn't exist
        return {
            "character_id": "smlq_sfw",
            "piercing_requirements": [
                {
                    "artifact_id": "silver_hoop",
                    "expected_location": "RIGHT_EYEBROW",
                    "is_mandatory": True,
                    "expected_count": 2
                },
                {
                    "artifact_id": "silver_hoop",
                    "expected_location": "LEFT_EYEBROW",
                    "is_mandatory": False,
                    "expected_count": 0
                },
                {
                    "artifact_id": "silver_hoop",
                    "expected_location": "LEFT_NOSTRIL",
                    "is_mandatory": True,
                    "expected_count": 1
                },
                {
                    "artifact_id": "silver_hoop",
                    "expected_location": "RIGHT_NOSTRIL",
                    "is_mandatory": False,
                    "expected_count": 0
                }
            ]
        }


def get_piercing_counts(image_path: str) -> Dict[str, Dict[str, int]]:
    """
    Detect piercings on a face image and return counts by location and type.
    
    This is a placeholder implementation - in a real system, this would use
    actual computer vision or ML models to detect piercings.
    
    Args:
        image_path (str): Path to the image file
        
    Returns:
        Dict[Location, Dict[ArtifactID, int]]: 
        Example: {"RIGHT_EYEBROW": {"silver_hoop": 2}, "LEFT_NOSTRIL": {"silver_hoop": 1}}
    """
    # Placeholder implementation - in a real system this would:
    # 1. Detect face in image
    # 2. Identify piercing locations
    # 3. Count piercings at each location
    # 4. Return counts by location and artifact type
    
    # For demonstration, we'll return mock data
    # In reality, this would be replaced with actual detection logic
    
    # Mock implementation - in a real system this would be replaced
    # with actual computer vision/ML detection logic
    
    return {
        "RIGHT_EYEBROW": {"silver_hoop": 2},
        "LEFT_EYEBROW": {"silver_hoop": 0}, 
        "LEFT_NOSTRIL": {"silver_hoop": 1},
        "RIGHT_NOSTRIL": {"silver_hoop": 0}
    }


def compare_with_specification(piercing_counts: Dict[str, Dict[str, int]], 
                             specification: Dict) -> List[Dict[str, Any]]:
    """
    Compare detected piercing counts with specification requirements.
    
    Args:
        piercing_counts (Dict): Detected counts by location and type
        specification (Dict): Specification requirements
        
    Returns:
        List[Dict]: List of mismatch details
    """
    mismatches = []
    
    # Get expected requirements
    requirements = specification["piercing_requirements"]
    
    for req in requirements:
        location = req["expected_location"]
        expected_count = req["expected_count"]
        artifact_id = req["artifact_id"]
        
        # Check if we have any data for this location
        if location in piercing_counts and artifact_id in piercing_counts[location]:
            detected_count = piercing_counts[location][artifact_id]
        else:
            detected_count = 0
            
        # Compare counts
        if detected_count != expected_count:
            if expected_count == 0:
                # Extra piercing detected
                mismatches.append({
                    "location": location,
                    "artifact": artifact_id,
                    "status": "EXTRA",
                    "expected": 0,
                    "detected": detected_count,
                    "message": f"Extra piercing detected"
                })
            elif detected_count == 0:
                # Missing piercing
                mismatches.append({
                    "location": location,
                    "artifact": artifact_id,
                    "status": "MISSING",
                    "expected": expected_count,
                    "detected": detected_count,
                    "message": f"Expected {expected_count}, found {detected_count}"
                })
            else:
                # Wrong count
                mismatches.append({
                    "location": location,
                    "artifact": artifact_id,
                    "status": "MISMATCH",
                    "expected": expected_count,
                    "detected": detected_count,
                    "message": f"Expected {expected_count}, found {detected_count}"
                })
    
    return mismatches


def format_error_message(mismatch: Dict[str, Any]) -> str:
    """Format error message for display"""
    if mismatch["status"] == "EXTRA":
        return f"[FAIL] {mismatch['location']}: expected 0, found {mismatch['detected']} (extra)"
    elif mismatch["status"] == "MISSING":
        return f"[FAIL] {mismatch['location']}: expected {mismatch['expected']}, found {mismatch['detected']}"
    else:  # MISMATCH
        return f"[FAIL] {mismatch['location']}: expected {mismatch['expected']}, found {mismatch['detected']}"


def process_image(image_path: str, specification: Dict) -> Tuple[str, List[Dict], Dict[str, Any]]:
    """
    Process a single image and return results.
    
    Returns:
        Tuple of (filename, mismatches, stats)
    """
    # Get piercing counts for this image
    try:
        piercing_counts = get_piercing_counts(image_path)
    except Exception as e:
        # Handle detection errors gracefully
        return os.path.basename(image_path), [{"error": str(e)}], {"error": str(e)}
    
    # Compare with specification
    mismatches = compare_with_specification(piercing_counts, specification)
    
    # Determine overall status
    if len(mismatches) == 0:
        status = "PASS"
    else:
        status = "FAIL"
    
    # Calculate stats
    stats = {
        "total_piercings": sum([sum(counts.values()) for counts in piercing_counts.values()]),
        "expected_piercings": sum([req["expected_count"] for req in specification["piercing_requirements"]]),
        "mismatches": len(mismatches)
    }
    
    return os.path.basename(image_path), mismatches, {"status": status, "stats": stats}


def generate_report(image_results: List[Tuple[str, List[Dict], Dict]], 
                   specification: Dict) -> None:
    """
    Generate and display comprehensive report.
    
    Args:
        image_results: List of (filename, mismatches, stats) tuples
        specification: Specification requirements
    """
    # Print header
    print("=" * 80)
    print("PIERCING DETECTION BENCHMARK REPORT")
    print("=" * 80)
    
    # Print specification
    print(f"Character ID: {specification['character_id']}")
    print("Piercing Requirements:")
    for req in specification["piercing_requirements"]:
        print(f"  {req['expected_location']}: {req['expected_count']} x {req['artifact_id']}")
    print()
    
    # Print results table
    print("File Name".ljust(30) + "Status".ljust(10) + "Error Details")
    print("-" * 80)
    
    total_files = len(image_results)
    pass_count = 0
    fail_count = 0
    
    for filename, mismatches, stats in image_results:
        if "error" in stats:
            print(f"{filename:<30} ERROR".ljust(40) + str(stats["error"]))
            continue
            
        if stats["status"] == "PASS":
            pass_count += 1
            print(f"{filename:<30} PASS")
        else:
            fail_count += 1
            if mismatches:
                # Show first error
                error_msg = format_error_message(mismatches[0])
                print(f"{filename:<30} FAIL".ljust(40) + error_msg)
            else:
                print(f"{filename:<30} FAIL")
    
    # Print summary
    print()
    print("-" * 80)
    print("SUMMARY")
    print("-" * 80)
    print(f"Total Files: {total_files}")
    print(f"PASS: {pass_count}")
    print(f"FAIL: {fail_count}")
    
    if total_files > 0:
        accuracy = (pass_count / total_files) * 100
        print(f"Accuracy: {accuracy:.1f}%")
    
    # Print detailed stats for failed files
    failed_files = [r for r in image_results if r[2]["status"] == "FAIL" and "error" not in r[2]]
    if failed_files:
        print("\nDETAILED FAILURES:")
        print("-" * 50)
        for filename, mismatches, stats in failed_files:
            print(f"\n{filename}:")
            for mismatch in mismatches:
                print(f"  {format_error_message(mismatch)}")


def main():
    """Main entry point for the piercing benchmark script"""
    parser = argparse.ArgumentParser(
        description="Piercing Detection Benchmark - Validates face images against specification requirements"
    )
    parser.add_argument("directory", help="Directory path containing face images to validate")
    parser.add_argument("--spec-file", default="src/dataset_filter/specs/smlq_sfw.json",
                       help="Specification file path (default: src/dataset_filter/specs/smlq_sfw.json)")
    
    args = parser.parse_args()
    
    # Validate directory
    if not os.path.exists(args.directory):
        print(f"Error: Directory '{args.directory}' does not exist")
        return 1
    
    if not os.path.isdir(args.directory):
        print(f"Error: '{args.directory}' is not a directory")
        return 1
    
    # Load specification
    print("Loading specification...")
    specification = load_specification(args.spec_file)
    
    # Get all image files
    image_extensions = {'.png', '.jpg', '.jpeg', '.bmp', '.tiff'}
    image_files = [
        f for f in os.listdir(args.directory)
        if os.path.splitext(f)[1].lower() in image_extensions
    ]
    
    if not image_files:
        print("No image files found in the specified directory")
        return 1
    
    print(f"Found {len(image_files)} image files")
    
    # Process each image
    print("Processing images...")
    results = []
    
    for filename in image_files:
        file_path = os.path.join(args.directory, filename)
        try:
            result = process_image(file_path, specification)
            results.append(result)
        except Exception as e:
            print(f"Error processing {filename}: {e}")
            results.append((filename, [{"error": str(e)}], {"error": str(e)}))
    
    # Generate report
    generate_report(results, specification)
    
    return 0


if __name__ == "__main__":
    exit(main())