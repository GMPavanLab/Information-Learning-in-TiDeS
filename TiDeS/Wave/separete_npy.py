import os
import numpy as np

def main():
    # 1) List available .npy files
    npy_files = [f for f in os.listdir('.') if f.endswith('.npy')]
    if not npy_files:
        print("[ERROR] No .npy files found in current directory!")
        return

    print("Available .npy files:")
    for idx, f in enumerate(npy_files):
        print(f"[{idx}] {f}")

    # 2) Ask user which file to analyze
    choice_str = input("\nPick file #: ")
    try:
        choice_idx = int(choice_str)
        chosen_file = npy_files[choice_idx]
    except (ValueError, IndexError):
        print("[ERROR] Invalid choice. Exiting.")
        return

    # 3) Prepare output folder
    base_name = os.path.splitext(chosen_file)[0]
    out_folder = f"{base_name}_scompose"
    os.makedirs(out_folder, exist_ok=True)
    print(f"[INFO] Outputs will be saved in: {out_folder}")

    # 4) Load data (assuming shape=(N, T, D))
    data = np.load(chosen_file)
    print(f"[INFO] Loaded data shape: {data.shape}")
    if len(data.shape) != 3:
        print("[ERROR] Expected data to have shape (N, T, D). Exiting.")
        return

    N, T, D = data.shape
    # For each dimension i in D, save data[..., i] as an .npy
    for i in range(D):
        component_i = data[..., i]
        out_npy = os.path.join(out_folder, f"component_{i}.npy")
        np.save(out_npy, component_i)
        print(f"[INFO] Saved component #{i} with shape {component_i.shape} => {out_npy}")

    print("[INFO] Done. All components saved.")

if __name__ == '__main__':
    main()
