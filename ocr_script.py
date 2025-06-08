#!/usr/bin/env python3
"""
Arabic OCR Processing Script

This script processes PDF files by extracting images, applying transformations,
and performing OCR to extract Arabic text.

Usage:
    python ocr_script.py [--input INPUT_FILE] [--output OUTPUT_FILE] [--transform]
    

"""
import argparse
import sys
from pathlib import Path
from typing import List, Optional, Union
import json
from datetime import datetime
from tqdm import tqdm
import ocr_functions
from Qariv03 import ArabicOCR


def setup_argument_parser():
    """Set up command line argument parser."""
    parser = argparse.ArgumentParser(
        description="Process PDF files with Arabic OCR",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    parser.add_argument(
        '--input', '-i',
        type=str,
        default='test_OCR_2025.pdf',
        help='Input PDF file path (default: test_OCR_2025.pdf)'
    )
    
    parser.add_argument(
        '--output', '-o',
        type=str,
        help='Output file path (default: auto-generated based on input)'
    )
    
    parser.add_argument(
        '--transform', '-t',
        action='store_true',
        help='Apply scan transformation to images before OCR'
    )
    
    parser.add_argument(
        '--format',
        choices=['txt', 'json'],
        default='txt',
        help='Output format (default: txt)'
    )

 
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Enable verbose output'
    )
    
    return parser


def validate_input_file(file_path):
    """Validate that the input file exists and is a PDF."""
    path = Path(file_path)
    
    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {file_path}")
    
    if not path.suffix.lower() == '.pdf':
        raise ValueError(f"Input file must be a PDF: {file_path}")
    
    return path


def generate_output_path(input_path, output_format):
    """Generate output file path based on input file."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_name = f"{input_path.stem}_ocr_results_{timestamp}.{output_format}"
    return input_path.parent / output_name


def process_pdf_images(pdf_path, apply_transform = False, verbose= False) :
    """
    Process PDF file and extract images.
    
    Args:
        pdf_path: Path to the PDF file
        apply_transform: Whether to apply scan transformation
        verbose: Whether to show detailed progress
        
    Returns:
        Tuple of (original_images, transformed_images)
    """
    if verbose:
        print(f"📄 Processing PDF: {pdf_path}")
    
    try:
        # Extract images from PDF
        with tqdm(desc="Extracting images from PDF", disable=not verbose) as pbar:
            pdf_images = ocr_functions.process_pdf(str(pdf_path))
            pbar.update(1)
        
        if not pdf_images:
            raise ValueError("No images found in PDF")
        
        if verbose:
            print(f"✅ Extracted {len(pdf_images)} images from PDF")
        
        # Apply transformations if requested
        transformed_images = []
        if apply_transform:
            desc = "Applying scan transformations"
            for image in tqdm(pdf_images, desc=desc, disable=not verbose):
                transformed = ocr_functions.scan_transform_image(image)
                transformed_images.append(transformed)
            
            if verbose:
                print("✅ Transformation completed")
        else:
            transformed_images = pdf_images
            if verbose:
                print("ℹ️  Using original images without transformation")
        
        return pdf_images, transformed_images
        
    except Exception as e:
        print(f"❌ Error processing PDF: {e}")
        raise


def perform_ocr(images, model_path, max_size= 1024, verbose: bool = False):
    """
    Perform OCR on the provided images.
    
    Args:
        images: List of images to process
        model_path: Optional path to local model
        max_size: Maximum image size for processing
        verbose: Whether to show detailed progress
        
    Returns:
        List of extracted texts
    """
    if verbose:
        print("🤖 Initializing OCR model...")
    
    try:
        # Initialize OCR with optional model path
        if model_path:
            ocr = ArabicOCR(model_name=model_path, max_size=max_size)
            if verbose:
                print(f"✅ Using local model: {model_path}")
        else:
            ocr = ArabicOCR(max_size=max_size)
            if verbose:
                print("✅ Using default HuggingFace model")
        
        # Perform OCR with progress bar
        if verbose:
            print(f"🔤 Starting OCR processing for {len(images)} images...")
        
        # Create a custom progress bar that works with the OCR class
        texts = []
        with tqdm(total=len(images), desc="Processing OCR", unit="page") as pbar:
            for i, image in enumerate(images):
                if verbose:
                    pbar.set_description(f"Processing page {i+1}")
                
                # Process single image
                text = ocr._process_single_image(image, max_size, 1500)
                if hasattr(ocr, 'clean_text'):
                    text = ocr.clean_text(text)
                
                texts.append(text)
                pbar.update(1)
                
                # Update progress bar with character count
                if verbose:
                    pbar.set_postfix(chars=len(text))
        
        if verbose:
            print("✅ OCR processing completed")
        return texts
        
    except Exception as e:
        print(f"❌ Error during OCR processing: {e}")
        raise


def save_results(texts, output_path, output_format, input_info, verbose= False):
    """
    Save OCR results to file.
    
    Args:
        texts: List of extracted texts
        output_path: Output file path
        output_format: Output format ('txt' or 'json')
        input_info: Information about the input file
        verbose: Whether to show detailed progress
    """
    if verbose:
        print(f"💾 Saving results to: {output_path}")
    
    try:
        with tqdm(desc="Saving results", total=1, disable=not verbose) as pbar:
            if output_format == 'json':
                # Save as JSON with metadata
                results = {
                    "metadata": {
                        "input_file": input_info["file_path"],
                        "processing_date": datetime.now().isoformat(),
                        "total_images": len(texts),
                        "transformation_applied": input_info.get("transform_applied", False)
                    },
                    "results": [
                        {
                            "page": i + 1,
                            "text": text,
                            "character_count": len(text)
                        }
                        for i, text in enumerate(texts)
                    ]
                }
                
                with open(output_path, 'w', encoding='utf-8') as f:
                    json.dump(results, f, ensure_ascii=False, indent=2)
                    
            else:  # txt format
                with open(output_path, 'w', encoding='utf-8') as f:
                    f.write(f"OCR Results for: {input_info['file_path']}\n")
                    f.write(f"Processing Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                    f.write(f"Total Pages: {len(texts)}\n")
                    f.write("=" * 50 + "\n\n")
                    
                    for i, text in enumerate(texts):
                        f.write(f"--- Page {i + 1} ---\n")
                        f.write(text)
                        f.write(f"\n\n{'=' * 30}\n\n")
            
            pbar.update(1)
        
        if verbose:
            print(f"✅ Results saved successfully ({len(texts)} pages processed)")
        
    except Exception as e:
        print(f"❌ Error saving results: {e}")
        raise


def main():
    """Main function to orchestrate the OCR processing."""
    parser = setup_argument_parser()
    args = parser.parse_args()
    
    try:
        # Validate input file
        input_path = validate_input_file(args.input)
        
        # Generate output path if not provided
        if args.output:
            output_path = Path(args.output)
        else:
            output_path = generate_output_path(input_path, args.format)
        
        if args.verbose:
            print("🚀 === Starting OCR Processing ===")
            print(f"📁 Input: {input_path}")
            print(f"📁 Output: {output_path}")
            print(f"🔄 Transform: {args.transform}")
            print(f"📝 Format: {args.format}")
            print()
        
        # Process PDF and extract images
        original_images, processed_images = process_pdf_images(
            input_path, 
            apply_transform=args.transform,
            verbose=args.verbose
        )
        
        # Perform OCR
        extracted_texts = perform_ocr(
            processed_images, 
            model_path=args.model_path,
            max_size=args.max_size,
            verbose=args.verbose
        )
        
        # Save results
        input_info = {
            "file_path": str(input_path),
            "transform_applied": args.transform
        }
        
        save_results(
            extracted_texts, 
            output_path, 
            args.format, 
            input_info,
            verbose=args.verbose
        )
        
        # Print summary
        total_chars = sum(len(text) for text in extracted_texts)
        
        if args.verbose:
            print()
            print("🎉 === Processing Complete ===")
            print(f"📊 Pages processed: {len(extracted_texts)}")
            print(f"📊 Total characters extracted: {total_chars:,}")
            print(f"📁 Results saved to: {output_path}")
        else:
            # Simple summary for non-verbose mode
            print(f"✅ Processed {len(extracted_texts)} pages → {output_path}")
        
    except KeyboardInterrupt:
        print("\n⚠️  Processing interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Processing failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()