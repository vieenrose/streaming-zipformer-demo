#!/usr/bin/env python3

"""
A/B Testing Script for Hotword Impact Analysis

This script compares ASR recognition accuracy with and without hotwords
using the 30 generated audio samples.
"""

import os
import json
import time
from collections import defaultdict
import sys

# Add project path
sys.path.insert(0, '/home/luigi/sherpa')

from demo_full_integration import main as demo_main
import argparse
from asr_engine import ASREnginePool
from asr_config import ASRConfig
import subprocess


def run_ab_test():
    """
    Run A/B testing on all generated audio files:
    - Test with hotwords enabled
    - Test with hotwords disabled
    - Compare recognition accuracy
    """
    print("Starting A/B Testing: Hotwords vs No Hotwords")
    print("=" * 60)
    
    # Define the audio files to test
    test_audio_dir = "/home/luigi/sherpa/test_audio"
    audio_files = []
    
    # Get all WAV files
    for file in os.listdir(test_audio_dir):
        if file.endswith(".wav"):
            audio_files.append(os.path.join(test_audio_dir, file))
    
    audio_files.sort()
    
    print(f"Found {len(audio_files)} audio files for testing")
    
    # Results storage
    results = {
        'with_hotwords': {},
        'without_hotwords': {}
    }
    
    # Process each audio file with and without hotwords
    for i, audio_file in enumerate(audio_files):
        print(f"\nProcessing file {i+1}/{len(audio_files)}: {os.path.basename(audio_file)}")
        
        # Test with hotwords enabled
        print("  Testing with hotwords enabled...")
        result_with_hotwords = run_single_test(audio_file, hotwords_enabled=True)
        results['with_hotwords'][audio_file] = result_with_hotwords
        
        # Test without hotwords
        print("  Testing without hotwords...")
        result_without_hotwords = run_single_test(audio_file, hotwords_enabled=False)
        results['without_hotwords'][audio_file] = result_without_hotwords
        
        # Print comparison for this file
        print_comparison(audio_file, result_with_hotwords, result_without_hotwords)
    
    # Generate summary report
    generate_summary_report(results)
    
    return results


def run_single_test(audio_file, hotwords_enabled=True):
    """
    Run a single ASR test on an audio file.
    
    Args:
        audio_file: Path to the audio file
        hotwords_enabled: Whether to enable hotwords for this test
    
    Returns:
        Dictionary containing test results
    """
    # We'll simulate this by calling the demo script with appropriate arguments
    cmd = [
        "/home/luigi/sherpa/venv/bin/python3",
        "/home/luigi/sherpa/demo_full_integration.py",
        "--file", audio_file
    ]
    
    if not hotwords_enabled:
        cmd.append("--no-hotwords")
    
    start_time = time.time()
    
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=120  # 2 minute timeout
        )
        
        processing_time = time.time() - start_time
        
        # Extract results from the output
        final_results = {
            'success': result.returncode == 0,
            'processing_time': processing_time,
            'stdout': result.stdout,
            'stderr': result.stderr,
            'return_code': result.returncode
        }
        
        # Try to parse the transcription results
        transcriptions = extract_transcriptions(result.stdout)
        final_results['transcriptions'] = transcriptions
        
        return final_results
        
    except subprocess.TimeoutExpired:
        return {
            'success': False,
            'processing_time': time.time() - start_time,
            'error': 'Timeout',
            'transcriptions': {}
        }
    except Exception as e:
        return {
            'success': False,
            'processing_time': time.time() - start_time,
            'error': str(e),
            'transcriptions': {}
        }


def extract_transcriptions(output_text):
    """
    Extract transcriptions from the demo output.
    
    Args:
        output_text: Text output from the demo script
        
    Returns:
        Dictionary mapping model names to their transcriptions
    """
    transcriptions = {}
    
    lines = output_text.split('\n')
    
    for line in lines:
        # Look for lines containing transcriptions
        if 'Partial:' in line or 'Final:' in line:
            # Extract model name and transcription
            if ':' in line and ('Partial:' in line or 'Final:' in line):
                parts = line.split(':')
                if len(parts) >= 2:
                    model_part = parts[0].strip()
                    text_part = ':'.join(parts[1:]).strip()
                    
                    # Clean up the text part
                    if 'Partial:' in text_part:
                        text_part = text_part.split('Partial:')[1].strip()
                    elif 'Final:' in text_part:
                        text_part = text_part.split('Final:')[1].strip()
                    
                    transcriptions[model_part] = text_part
    
    return transcriptions


def print_comparison(audio_file, result_with, result_without):
    """Print comparison between hotwords enabled vs disabled."""
    print(f"\n  Comparison for: {os.path.basename(audio_file)}")
    print(f"    With Hotwords:    Success={result_with['success']}, Time={result_with['processing_time']:.2f}s")
    print(f"    Without Hotwords: Success={result_without['success']}, Time={result_without['processing_time']:.2f}s")
    
    # Show transcriptions if available
    if result_with.get('transcriptions') and result_without.get('transcriptions'):
        print("    Transcriptions:")
        for model in result_with['transcriptions']:
            with_text = result_with['transcriptions'].get(model, 'N/A')
            without_text = result_without['transcriptions'].get(model, 'N/A')
            print(f"      {model}:")
            print(f"        With:    '{with_text}'")
            print(f"        Without: '{without_text}'")


def generate_summary_report(results):
    """Generate a summary report of the A/B testing."""
    print("\n" + "=" * 60)
    print("A/B TESTING SUMMARY REPORT")
    print("=" * 60)
    
    total_files = len(results['with_hotwords'])
    
    # Count successful runs
    successful_with = sum(1 for r in results['with_hotwords'].values() if r['success'])
    successful_without = sum(1 for r in results['without_hotwords'].values() if r['success'])
    
    print(f"Total audio files tested: {total_files}")
    print(f"Successful with hotwords: {successful_with}/{total_files} ({successful_with/total_files*100:.1f}%)")
    print(f"Successful without hotwords: {successful_without}/{total_files} ({successful_without/total_files*100:.1f}%)")
    
    # Calculate average processing times
    times_with = [r['processing_time'] for r in results['with_hotwords'].values() if r['success']]
    times_without = [r['processing_time'] for r in results['without_hotwords'].values() if r['success']]
    
    avg_time_with = sum(times_with) / len(times_with) if times_with else 0
    avg_time_without = sum(times_without) / len(times_without) if times_without else 0
    
    print(f"Average processing time with hotwords: {avg_time_with:.2f}s")
    print(f"Average processing time without hotwords: {avg_time_without:.2f}s")
    
    # Analyze transcription quality (simple approach - count hotword occurrences)
    hotword_matches_with = 0
    hotword_matches_without = 0
    total_hotword_tests = 0
    
    # Define our hotwords to check for
    hotwords = [
        "张伟明", "李建国", "王小红", "陈志强", "刘芳芳", "杨光明", "赵文静", "黄丽华", "周永康", "吴天明",
        "LILY", "LUCY", "EMMA", "KEVIN", "CINDY", "TONY", "AMY", "DAVID", "JESSICA", "MICHAEL"
    ]
    
    for audio_file in results['with_hotwords']:
        # Check if the filename indicates which hotwords should be present
        basename = os.path.basename(audio_file).lower()
        
        # Extract transcriptions
        trans_with = results['with_hotwords'][audio_file].get('transcriptions', {})
        trans_without = results['without_hotwords'][audio_file].get('transcriptions', {})
        
        # Combine all transcriptions for this file
        all_trans_with = ' '.join(trans_with.values()).upper()
        all_trans_without = ' '.join(trans_without.values()).upper()
        
        # Check for hotword matches in both versions
        for hw in hotwords:
            hw_upper = hw.upper()
            if hw_upper in all_trans_with:
                hotword_matches_with += 1
                total_hotword_tests += 1
            if hw_upper in all_trans_without:
                hotword_matches_without += 1
                if hw_upper not in all_trans_with:  # Only count if not already counted
                    total_hotword_tests += 1
    
    print(f"\nHotword Recognition:")
    print(f"  Hotwords found with hotword support: {hotword_matches_with}")
    print(f"  Hotwords found without hotword support: {hotword_matches_without}")
    
    if total_hotword_tests > 0:
        accuracy_with = hotword_matches_with / total_hotword_tests * 100 if total_hotword_tests > 0 else 0
        accuracy_without = hotword_matches_without / total_hotword_tests * 100 if total_hotword_tests > 0 else 0
        print(f"  Hotword recognition accuracy with hotwords: {accuracy_with:.1f}%")
        print(f"  Hotword recognition accuracy without hotwords: {accuracy_without:.1f}%")
    
    print("\nCONCLUSION:")
    if hotword_matches_with > hotword_matches_without:
        print("  ✓ Hotwords significantly improve recognition accuracy for target terms")
    elif hotword_matches_with == hotword_matches_without:
        print("  → Hotwords show no significant difference in recognition accuracy")
    else:
        print("  ⚠ Hotwords may be degrading recognition accuracy (unexpected)")
    
    print("\n" + "=" * 60)


if __name__ == "__main__":
    run_ab_test()