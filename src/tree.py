import os
import argparse
# %%
def print_tree(startpath, prefix=''):
    for root, dirs, files in os.walk(startpath):
        level = root.replace(startpath, '').count(os.sep)
        indent = '│   ' * level
        print(f"{indent}├── {os.path.basename(root)}/")
        subindent = '│   ' * (level + 1)
        for f in files:
            print(f"{subindent}└── {f}")
        # prevent deep traversal for display purposes
        dirs[:] = [d for d in dirs if not d.startswith('.')]
# %%
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Print directory structure of datasets.")
    parser.add_argument("--dataset", default="datasets/MPI_Sintel", help="Path to the dataset directory.")
    parser.add_argument("--output", default="datasets/", help="Output file name if saving.")
    args = parser.parse_args([])
    dataset_path = args.dataset
    if not os.path.exists(dataset_path):
        print(f"Dataset path '{dataset_path}' does not exist.")
    else:
        if args.output:
            output_file = os.path.join(args.output, 'tree.txt')
            # Redirect print output to file
            import sys
            original_stdout = sys.stdout
            with open(output_file, 'w') as f:
                sys.stdout = f
                print_tree(dataset_path)
                sys.stdout = original_stdout
            print(f"Dataset structure saved to {output_file}.")
        else:
            print_tree(dataset_path)
            print(f"Dataset structure printed for {dataset_path}.")

# %%
