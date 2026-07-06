import os
import numpy as np

def main():
    # 1) List all .npy files
    npy_files = [f for f in os.listdir('.') if f.endswith('.npy')]
    if not npy_files:
        print("[ERROR] No .npy files found in the current directory!")
        return

    print("Available .npy files:")
    for idx, fname in enumerate(npy_files):
        print(f"[{idx}] {fname}")

    # 2) Prompt user for multiple indices (comma-separated, e.g. '1,2,3')
    choice_str = input("Enter one or more indices to merge (e.g. '1,2,3'): ")
    parts = [x.strip() for x in choice_str.split(",") if x.strip()]

    # Must have at least 2 indices to merge
    if len(parts) < 2:
        print("[ERROR] You must specify at least two indices, separated by commas.")
        return

    # Convert each to an integer index
    try:
        chosen_indices = list(map(int, parts))
    except ValueError:
        print("[ERROR] Invalid input. Indices must be integers.")
        return

    # 3) Load all chosen arrays and verify shapes
    arrays = []
    shapes = []
    chosen_filenames = []
    for i in chosen_indices:
        try:
            fpath = npy_files[i]
        except IndexError:
            print(f"[ERROR] Index {i} is out of range.")
            return
        arr = np.load(fpath)
        arrays.append(arr)
        shapes.append(arr.shape)
        chosen_filenames.append(os.path.splitext(fpath)[0])
        print(f"[INFO] Loaded '{fpath}' with shape {arr.shape}")

    # Check if all shapes match
    first_shape = shapes[0]
    for s in shapes[1:]:
        if s != first_shape:
            print("[ERROR] At least two arrays have different shapes; cannot stack them.")
            return

    # 4) Stack them along a new axis (here, the last axis)
    merged = np.stack(arrays, axis=-1)
    print(f"[INFO] Stacked shape: {merged.shape}")

    # 5) Build an output filename. For clarity, join the chosen filenames with '_'
    #    Truncate or adjust if you get very long names.
    merged_name = "_".join(chosen_filenames)
    out_name = f"merged_{merged_name}.npy"
    np.save(out_name, merged)
    print(f"[INFO] Merged array saved to '{out_name}'.")

if __name__ == "__main__":
    main()

