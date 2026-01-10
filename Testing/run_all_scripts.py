#!/usr/bin/env python3
"""
Script to run all face matching analysis scripts in sequence.
"""

import subprocess
import sys
import os
from pathlib import Path

def run_script(script_name):
    """
    Run a Python script and handle any errors.

    Args:
        script_name (str): Name of the Python script to run

    Returns:
        bool: True if successful, False if failed
    """
    script_path = Path(__file__).parent / script_name

    if not script_path.exists():
        print(f"Error: {script_name} not found!")
        return False

    print(f"\n{'='*50}")
    print(f"Running {script_name}...")
    print(f"{'='*50}")

    try:
        result = subprocess.run([sys.executable, str(script_path)],
                              capture_output=True, text=True, cwd=Path(__file__).parent)

        # Print stdout
        if result.stdout:
            print(result.stdout)

        # Print stderr if there was an error
        if result.stderr:
            print("STDERR:", result.stderr)

        if result.returncode == 0:
            print(f"✓ {script_name} completed successfully")
            return True
        else:
            print(f"✗ {script_name} failed with return code {result.returncode}")
            return False

    except Exception as e:
        print(f"✗ Error running {script_name}: {e}")
        return False

def main():
    """Main function to run all scripts in sequence."""
    print("Starting Face Matching Analysis Pipeline")
    print("=" * 60)

    # List of scripts to run in order
    scripts = [
        "false_positive_test.py",
        "calc_avg_score.py",
        "calc_total_avg_score.py",
        "calc_max_positive_comparision.py",
        "cal_minimum_in_max_score.py"
    ]

    failed_scripts = []

    # Run each script
    for script in scripts:
        success = run_script(script)
        if not success:
            failed_scripts.append(script)
            # Continue with next script even if one fails
            print(f"Continuing with next script despite failure of {script}")

    print(f"\n{'='*60}")
    print("Pipeline execution completed")

    if failed_scripts:
        print(f"\n⚠️  The following scripts failed: {', '.join(failed_scripts)}")
        sys.exit(1)
    else:
        print("\n✅ All scripts completed successfully!")
        sys.exit(0)

if __name__ == "__main__":
    main()
