#!/usr/bin/env python3
"""
Example usage of the dataset download script
============================================

This script demonstrates various ways to use the download_dataset.py script
for downloading datasets for PIE-Net training.
"""

import subprocess
import sys
import os

def run_command(cmd, description):
    """Run a command and print its description."""
    print(f"\n{description}")
    print(f"Command: {' '.join(cmd)}")
    print("-" * 50)
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print("✓ Success!")
        if result.stdout:
            print(result.stdout)
    except subprocess.CalledProcessError as e:
        print(f"✗ Failed: {e}")
        if e.stderr:
            print(f"Error: {e.stderr}")
    except FileNotFoundError:
        print("✗ download_dataset.py not found. Make sure it's in the current directory.")

def main():
    print("PIE-Net Dataset Download Examples")
    print("=" * 40)
    
    # Check if download_dataset.py exists
    if not os.path.exists("download_dataset.py"):
        print("✗ download_dataset.py not found in current directory.")
        print("Please make sure the download script is available.")
        sys.exit(1)
    
    # Example 1: Download a single ZIP file
    run_command([
        "python", "download_dataset.py",
        "--url", "https://httpbin.org/uuid",  # Example URL that returns JSON
        "--output", "./example_downloads",
        "--filename", "test_file.json",
        "--no-extract"
    ], "Example 1: Download single file without extraction")
    
    # Example 2: Download with extraction (if you have a real dataset URL)
    print(f"\nExample 2: Download with automatic extraction")
    print("Command: python download_dataset.py --url <DATASET_URL> --output ./datasets --extract")
    print("Note: Replace <DATASET_URL> with actual dataset URL")
    
    # Example 3: Download multiple files from URLs file
    print(f"\nExample 3: Download multiple files from URL list")
    print("Command: python download_dataset.py --urls dataset_urls.txt --output ./datasets")
    print("Note: Add actual URLs to dataset_urls.txt file")
    
    # Example 4: Resume interrupted download
    print(f"\nExample 4: Resume interrupted download")
    print("Command: python download_dataset.py --url <URL> --output ./datasets --resume")
    
    # Example 5: Download with checksum verification
    print(f"\nExample 5: Download with checksum verification")
    print("Command: python download_dataset.py --url <URL> --output ./datasets --checksum <SHA256_HASH>")
    
    print(f"\n{'='*40}")
    print("To use with real datasets:")
    print("1. Find the dataset URL (e.g., MIT Intrinsics, NYU Depth, etc.)")
    print("2. Replace the example URLs with real dataset URLs")
    print("3. Run the download script with appropriate parameters")
    print("\nExample for a real dataset:")
    print("python download_dataset.py \\")
    print("  --url 'https://example.com/intrinsic_dataset.zip' \\")
    print("  --output ./datasets \\")
    print("  --extract")

if __name__ == "__main__":
    main()
