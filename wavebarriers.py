import numpy as np
import cv2
import pandas as pd
import os
import time
import glob

# ==========================================
# CONFIGURATION
# ==========================================
BASE_PATH = r"C:\Users\KarlRoesch\physVLA\wavetheory"
IN_DIR = os.path.join(BASE_PATH, "wavein")
OUT_DIR = os.path.join(BASE_PATH, "waveana")
VID_DIR = os.path.join(BASE_PATH, "wavevides")

NX, NY = 600, 400         
SCALE_FACTOR = 100.0       
NSTEPS = 400 
FPS = 30.0

for d in [IN_DIR, OUT_DIR, VID_DIR]:
    if not os.path.exists(d): os.makedirs(d)

def get_next_sequence_number(directory):
    """Checks the directory for existing files to find the next N+1 number."""
    files = glob.glob(os.path.join(directory, "integrated_results_*.csv"))
    if not files:
        return 1
    nums = []
    for f in files:
        try:
            # Extracts only the digits from the filename
            num_part = "".join(filter(str.isdigit, os.path.basename(f)))
            if num_part: nums.append(int(num_part))
        except ValueError:
            continue
    return max(nums) + 1 if nums else 1

def generate_scattering_geometry(wall_id):
    max_height, wall_length = 4.0, 6.0
    num_points = 10 
    bb = np.array([[0, 0, 0], [6, 0, 0], [6, 4, 0], [0, 4, 0], [0, 0, 0]], dtype=np.float32)
    x_vals = np.linspace(0, wall_length, num_points)
    y_vals = -(max_height / ((wall_length/2)**2)) * (x_vals - (wall_length/2))**2 + max_height
    return {'ID': f"basewall{wall_id}", 'bb': bb, 'curve': np.column_stack((x_vals, y_vals))}

def run_simulation(plane_data, sx_m, sy_m):
    wall_id, curve_pts = plane_data['ID'], plane_data['curve']
    phi, psi = np.zeros((NY, NX), dtype=np.float32), np.zeros((NY, NX), dtype=np.float32)
    mask = np.ones((NY, NX), dtype=np.uint8) * 255
    cv2.polylines(mask, [(curve_pts * SCALE_FACTOR).astype(np.int32)], False, 0, 3)

    sx, sy = int(sx_m * SCALE_FACTOR), int(sy_m * SCALE_FACTOR)
    c_sq = np.full((NY, NX), 0.25, dtype=np.float32)
    c_sq[mask == 0] = 0.0 

    for frame in range(NSTEPS):
        if frame < 100: phi[sy, sx] += 1.5 * np.sin(2 * np.pi * frame / 15)
        laplacian = (np.roll(phi, 1, 0) + np.roll(phi, -1, 0) + np.roll(phi, 1, 1) + np.roll(phi, -1, 1) - 4 * phi)
        new_phi = (2 * phi - psi + (c_sq * laplacian)) * 0.985 
        psi[:], phi[:] = phi, new_phi

    intensity = cv2.resize(np.abs(phi), (20, 30)).flatten()
    return [wall_id, sx_m, sy_m] + intensity.tolist()

def main():
    print(f"--- Continuous Simulation Loop Started ---")
    print(f"Waiting for config file in {IN_DIR} to authorize start...")
    
    while not glob.glob(os.path.join(IN_DIR, "config*.txt")):
        time.sleep(2)

    # Initialize sequence number based on existing files
    iteration = get_next_sequence_number(OUT_DIR)
    
    while True:
        print(f"\n[Iteration {iteration}] Starting Batch...")
        batch_results = []
        
        for w_id in range(1, 11):
            sx_rand = np.random.uniform(1.0, 5.0)
            sy_rand = np.random.uniform(0.3, 1.2)
            
            plane = generate_scattering_geometry(w_id)
            print(f"  Simulating {plane['ID']} | Source: ({sx_rand:.2f}, {sy_rand:.2f})")
            result = run_simulation(plane, sx_rand, sy_rand)
            batch_results.append(result)
        
        # Save with sequential number
        headers = ["Plane_ID", "Source_X", "Source_Y"] + [f"p{i}" for i in range(600)]
        out_file = os.path.join(OUT_DIR, f"integrated_results_{iteration}.csv")
        pd.DataFrame(batch_results, columns=headers).to_csv(out_file, index=False)
        
        print(f"[✓] Batch {iteration} saved to waveana.")
        iteration += 1
        time.sleep(5)

if __name__ == "__main__":
    main()