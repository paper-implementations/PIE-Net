#!/usr/bin/env python3
"""
Dataset Download Script for PIE-Net
===================================

This script downloads dataset files from URLs with progress tracking,
resume capability, and automatic extraction for common archive formats.

Usage:
    python download_dataset.py --url <URL> --output <OUTPUT_DIR> [options]

Examples:
    # Download and extract a dataset
    python download_dataset.py --url "https://example.com/dataset.zip" --output ./datasets

    # Download with custom filename
    python download_dataset.py --url "https://example.com/data.tar.gz" --output ./datasets --filename "my_dataset.tar.gz"

    # Download without extraction
    python download_dataset.py --url "https://example.com/dataset.zip" --output ./datasets --no-extract

    # Resume interrupted download
    python download_dataset.py --url "https://example.com/dataset.zip" --output ./datasets --resume
"""

import os
import sys
import argparse
import requests
import zipfile
import tarfile
import gzip
import shutil
from pathlib import Path
from tqdm import tqdm
from urllib.parse import urlparse
import hashlib
import time

class DatasetDownloader:
    def __init__(self, url, output_dir, filename=None, chunk_size=8192):
        self.url = url
        self.output_dir = Path(output_dir)
        self.chunk_size = chunk_size
        
        # Create output directory if it doesn't exist
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Determine filename
        if filename:
            self.filename = filename
        else:
            parsed_url = urlparse(url)
            self.filename = os.path.basename(parsed_url.path) or "dataset_file"
        
        self.filepath = self.output_dir / self.filename
        
    def get_file_size(self):
        """Get the total file size from the server."""
        try:
            response = requests.head(self.url, allow_redirects=True)
            return int(response.headers.get('content-length', 0))
        except:
            return 0
    
    def download_with_progress(self, resume=False):
        """Download file with progress bar and resume capability."""
        # Check if file already exists and get its size
        initial_pos = 0
        if resume and self.filepath.exists():
            initial_pos = self.filepath.stat().st_size
            print(f"Resuming download from byte {initial_pos}")
        
        # Get total file size
        total_size = self.get_file_size()
        
        # Setup headers for resume
        headers = {}
        if resume and initial_pos > 0:
            headers['Range'] = f'bytes={initial_pos}-'
        
        try:
            # Start the download
            response = requests.get(self.url, headers=headers, stream=True)
            response.raise_for_status()
            
            # Open file in appropriate mode
            mode = 'ab' if resume and initial_pos > 0 else 'wb'
            
            with open(self.filepath, mode) as file:
                # Setup progress bar
                if total_size > 0:
                    progress_bar = tqdm(
                        total=total_size,
                        initial=initial_pos,
                        unit='B',
                        unit_scale=True,
                        desc=f"Downloading {self.filename}"
                    )
                else:
                    progress_bar = tqdm(
                        unit='B',
                        unit_scale=True,
                        desc=f"Downloading {self.filename}"
                    )
                
                # Download in chunks
                for chunk in response.iter_content(chunk_size=self.chunk_size):
                    if chunk:
                        file.write(chunk)
                        progress_bar.update(len(chunk))
                
                progress_bar.close()
            
            print(f"✓ Download completed: {self.filepath}")
            return True
            
        except requests.exceptions.RequestException as e:
            print(f"✗ Download failed: {e}")
            return False
        except KeyboardInterrupt:
            print(f"\n✗ Download interrupted. Use --resume to continue later.")
            return False
    
    def verify_checksum(self, expected_hash, algorithm='sha256'):
        """Verify file integrity using checksum."""
        if not self.filepath.exists():
            return False
        
        hash_func = getattr(hashlib, algorithm)()
        
        print(f"Verifying {algorithm} checksum...")
        with open(self.filepath, 'rb') as file:
            for chunk in iter(lambda: file.read(4096), b""):
                hash_func.update(chunk)
        
        calculated_hash = hash_func.hexdigest()
        
        if calculated_hash.lower() == expected_hash.lower():
            print(f"✓ Checksum verification passed")
            return True
        else:
            print(f"✗ Checksum verification failed")
            print(f"  Expected: {expected_hash}")
            print(f"  Got:      {calculated_hash}")
            return False
    
    def extract_archive(self, extract_dir=None):
        """Extract downloaded archive file."""
        if not self.filepath.exists():
            print(f"✗ File not found: {self.filepath}")
            return False
        
        if extract_dir is None:
            extract_dir = self.output_dir / self.filepath.stem
        else:
            extract_dir = Path(extract_dir)
        
        extract_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"Extracting {self.filename} to {extract_dir}...")
        
        try:
            # Handle different archive formats
            if self.filename.endswith('.zip'):
                with zipfile.ZipFile(self.filepath, 'r') as zip_ref:
                    zip_ref.extractall(extract_dir)
            
            elif self.filename.endswith(('.tar.gz', '.tgz')):
                with tarfile.open(self.filepath, 'r:gz') as tar_ref:
                    tar_ref.extractall(extract_dir)
            
            elif self.filename.endswith(('.tar.bz2', '.tbz2')):
                with tarfile.open(self.filepath, 'r:bz2') as tar_ref:
                    tar_ref.extractall(extract_dir)
            
            elif self.filename.endswith('.tar'):
                with tarfile.open(self.filepath, 'r') as tar_ref:
                    tar_ref.extractall(extract_dir)
            
            elif self.filename.endswith('.gz') and not self.filename.endswith('.tar.gz'):
                # Handle single .gz files
                output_file = extract_dir / self.filepath.stem
                with gzip.open(self.filepath, 'rb') as f_in:
                    with open(output_file, 'wb') as f_out:
                        shutil.copyfileobj(f_in, f_out)
            
            else:
                print(f"✗ Unsupported archive format: {self.filename}")
                return False
            
            print(f"✓ Extraction completed: {extract_dir}")
            return True
            
        except Exception as e:
            print(f"✗ Extraction failed: {e}")
            return False

def download_multiple_files(urls, output_dir, extract=True, resume=False):
    """Download multiple files from a list of URLs."""
    successful_downloads = []
    failed_downloads = []
    
    for i, url in enumerate(urls, 1):
        print(f"\n[{i}/{len(urls)}] Processing: {url}")
        
        downloader = DatasetDownloader(url, output_dir)
        
        if downloader.download_with_progress(resume=resume):
            successful_downloads.append(url)
            
            if extract:
                downloader.extract_archive()
        else:
            failed_downloads.append(url)
    
    print(f"\n=== Download Summary ===")
    print(f"✓ Successful: {len(successful_downloads)}")
    print(f"✗ Failed: {len(failed_downloads)}")
    
    if failed_downloads:
        print(f"\nFailed downloads:")
        for url in failed_downloads:
            print(f"  - {url}")

def main():
    parser = argparse.ArgumentParser(
        description="Download dataset files from URLs with progress tracking and extraction",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    # Required arguments
    parser.add_argument('--url', type=str, help='URL to download from')
    parser.add_argument('--urls', type=str, help='File containing list of URLs (one per line)')
    parser.add_argument('--output', '-o', type=str, required=True,
                       help='Output directory for downloaded files')
    
    # Optional arguments
    parser.add_argument('--filename', '-f', type=str,
                       help='Custom filename for downloaded file')
    parser.add_argument('--extract', '-e', action='store_true', default=True,
                       help='Extract archive files after download (default: True)')
    parser.add_argument('--no-extract', action='store_true',
                       help='Do not extract archive files')
    parser.add_argument('--resume', '-r', action='store_true',
                       help='Resume interrupted downloads')
    parser.add_argument('--extract-dir', type=str,
                       help='Custom extraction directory')
    parser.add_argument('--checksum', type=str,
                       help='Expected checksum for verification')
    parser.add_argument('--checksum-algorithm', type=str, default='sha256',
                       choices=['md5', 'sha1', 'sha256', 'sha512'],
                       help='Checksum algorithm (default: sha256)')
    parser.add_argument('--chunk-size', type=int, default=8192,
                       help='Download chunk size in bytes (default: 8192)')
    
    args = parser.parse_args()
    
    # Validate arguments
    if not args.url and not args.urls:
        parser.error("Either --url or --urls must be provided")
    
    if args.url and args.urls:
        parser.error("Cannot use both --url and --urls simultaneously")
    
    # Set extraction flag
    extract = args.extract and not args.no_extract
    
    try:
        if args.urls:
            # Download multiple files from file
            if not os.path.exists(args.urls):
                print(f"✗ URL file not found: {args.urls}")
                sys.exit(1)
            
            with open(args.urls, 'r') as f:
                urls = [line.strip() for line in f if line.strip() and not line.startswith('#')]
            
            if not urls:
                print(f"✗ No valid URLs found in {args.urls}")
                sys.exit(1)
            
            download_multiple_files(urls, args.output, extract, args.resume)
        
        else:
            # Download single file
            downloader = DatasetDownloader(
                args.url, 
                args.output, 
                args.filename, 
                args.chunk_size
            )
            
            # Download the file
            if downloader.download_with_progress(resume=args.resume):
                
                # Verify checksum if provided
                if args.checksum:
                    if not downloader.verify_checksum(args.checksum, args.checksum_algorithm):
                        print("✗ Checksum verification failed. File may be corrupted.")
                        sys.exit(1)
                
                # Extract if requested
                if extract:
                    if not downloader.extract_archive(args.extract_dir):
                        print("✗ Extraction failed")
                        sys.exit(1)
                
                print("✓ All operations completed successfully!")
            else:
                print("✗ Download failed")
                sys.exit(1)
    
    except KeyboardInterrupt:
        print("\n✗ Operation cancelled by user")
        sys.exit(1)
    except Exception as e:
        print(f"✗ Unexpected error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
