import os
import csv
from pathlib import Path
from tqdm import tqdm
import sys
import datetime


def count_points_in_txt(txt_path):
    """Read a TXT file and return the number of lines (number of point clouds)"""
    try:
        with open(txt_path, 'r', encoding='utf-8') as f:
            
            line_count = 0
            for line in f:
                if line.strip():  
                    line_count += 1
        return line_count
    except Exception as e:
        print(f"Read file {txt_path} error occurs: {e}")
        return 0


def process_tree_files(input_folder, output_path):
    """
    Process tree point cloud to statistical CSV files

    Args:
        input_folder: 
        output_path: 
    """
  
    input_path = Path(input_folder)
    txt_files = list(input_path.glob("*.txt"))

    if not txt_files:
        print(f"No TXT file found {input_folder} ")
        return False

    print(f"Found {len(txt_files)} TXT files")

   
    tree_files = {}
    for file in txt_files:
        filename = file.stem  

       
        if "-canopy" in filename:
            tree_id = filename.replace("-canopy", "")
            if tree_id not in tree_files:
                tree_files[tree_id] = {"full": None, "canopy": file, "trunk": None}
            else:
                tree_files[tree_id]["canopy"] = file
        elif "-trunk" in filename:
            tree_id = filename.replace("-trunk", "")
            if tree_id not in tree_files:
                tree_files[tree_id] = {"full": None, "canopy": None, "trunk": file}
            else:
                tree_files[tree_id]["trunk"] = file
        else:
            
            tree_id = filename
            if tree_id not in tree_files:
                tree_files[tree_id] = {"full": file, "canopy": None, "trunk": None}
            else:
                tree_files[tree_id]["full"] = file

    
    output_path = Path(output_path)
    if output_path.is_dir():
        
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        output_csv = output_path / f"tree_pointcloud_stats_{timestamp}.csv"
    else:
        output_csv = output_path

   
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    try:
        with open(output_csv, 'w', newline='', encoding='utf-8') as csvfile:
            csv_writer = csv.writer(csvfile)
            csv_writer.writerow(['Tree ID', 'Total number ', 'Number of Crown Points', 'Number of Trunk Points'])

           
            for tree_id, files in tqdm(tree_files.items(), desc="Processing tree files"):
                full_count = count_points_in_txt(files["full"]) if files["full"] else 0
                canopy_count = count_points_in_txt(files["canopy"]) if files["canopy"] else 0
                trunk_count = count_points_in_txt(files["trunk"]) if files["trunk"] else 0

                csv_writer.writerow([tree_id, full_count, canopy_count, trunk_count])

        print(f"Processing complete! Results saved to {output_csv}")

      
        print("\nStatistical Summary:")
        print(f"A total of {len(tree_files)} trees were processed")

        return True
    except PermissionError:
        print(f"Error: Permission denied to create a file at {output_csv}")
        print("Please try selecting a directory you have write permissions for, or run the program as an administrator.")
        return False
    except Exception as e:
        print(f"Error occurred writing to CSV file: {e}")
        return False


def main():
    
    if len(sys.argv) < 3:
        print("Tree-Point Cloud File Counting")
        print()

        
        input_folder = input("Please enter the folder path containing the tree point cloud files: ").strip()
        output_path = input("Please enter the path or folder for the CSV file output: ").strip()

        if not input_folder or not output_path:
            print("Error: Both the input folder and output path must be provided.")
            return
    else:
        input_folder = sys.argv[1]
        output_path = sys.argv[2]

    if not os.path.exists(input_folder):
        print(f"Error: The folder '{input_folder}' does not exist")
        return

    process_tree_files(input_folder, output_path)


if __name__ == "__main__":
    main()