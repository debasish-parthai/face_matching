import json
import os
from pathlib import Path

def categorize_score(score):
    """Categorize score into predefined ranges"""
    if score >= 90:
        return "above_90"
    elif 80 <= score < 90:
        return "80_to_90"
    elif 70 <= score < 80:
        return "70_to_80"
    elif 60 <= score < 70:
        return "60_to_70"
    elif 50 <= score < 60:
        return "50_to_60"
    elif 40 <= score < 50:
        return "40_to_50"
    elif 20 <= score < 40:
        return "20_to_40"
    elif 0 <= score < 20:
        return "0_to_20"
    else:
        return "negative_or_invalid"

def main():
    # Input directory
    input_dir = Path("TestQA/User_Matching_Results")

    # Generate file paths from cross_user_matching_results1.json to cross_user_matching_results567.json
    input_files = []
    for i in range(1, 568):  # 1 to 567 inclusive
        file_path = input_dir / f"cross_user_matching_results{i}.json"
        input_files.append(str(file_path))

    # Output directory
    output_dir = Path("TestQA/summary_results")
    output_dir.mkdir(exist_ok=True)

    # Initialize categories with empty lists
    categories = {
        "above_90": [],
        "80_to_90": [],
        "70_to_80": [],
        "60_to_70": [],
        "50_to_60": [],
        "40_to_50": [],
        "20_to_40": [],
        "0_to_20": [],
        "negative_or_invalid": []
    }

    # Process each input file
    for input_file in input_files:
        if not os.path.exists(input_file):
            print(f"Warning: {input_file} not found, skipping...")
            continue

        print(f"Processing {input_file}...")

        try:
            with open(input_file, 'r', encoding='utf-8') as f:
                data = json.load(f)

            # Process each entry
            for entry in data:
                score = entry.get('score', 0)
                category = categorize_score(score)
                categories[category].append(entry)

        except json.JSONDecodeError as e:
            print(f"Error reading {input_file}: {e}")
            continue
        except Exception as e:
            print(f"Error processing {input_file}: {e}")
            continue

    # Write categorized data to separate JSON files
    file_mappings = {
        "above_90.json": ["above_90"],
        "80_to_90.json": ["80_to_90"],
        "70_to_80.json": ["70_to_80"],
        "60_to_70.json": ["60_to_70"],
        "50_to_60.json": ["50_to_60"],
        "40_to_50.json": ["40_to_50"],
        "20_to_40.json": ["20_to_40"],
        "0_to_20.json": ["0_to_20"],
        "negative_or_invalid.json": ["negative_or_invalid"]
    }

    for output_file, category_keys in file_mappings.items():
        output_path = output_dir / output_file

        # Combine data from specified categories
        combined_data = []
        for key in category_keys:
            combined_data.extend(categories[key])

        # Sort by score in descending order
        combined_data.sort(key=lambda x: x.get('score', 0), reverse=True)

        # Write to file
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(combined_data, f, indent=2, ensure_ascii=False)

        print(f"Created {output_file} with {len(combined_data)} entries")

    # Print summary statistics
    print("\nSummary Statistics:")
    print("=" * 50)
    for category, data in categories.items():
        print(f"{category.replace('_', ' ').title()}: {len(data)} entries")

    total_entries = sum(len(data) for data in categories.values())
    print(f"\nTotal entries processed: {total_entries}")

    # Create count summary
    count_summary = {}
    for category, data in categories.items():
        count_summary[category] = len(data)
    count_summary["total_entries"] = total_entries

    # Write count summary to JSON file
    count_summary_path = output_dir / "count_summary.json"
    with open(count_summary_path, 'w', encoding='utf-8') as f:
        json.dump(count_summary, f, indent=2, ensure_ascii=False)

    print(f"\nCount summary saved to: {count_summary_path}")

if __name__ == "__main__":
    main()