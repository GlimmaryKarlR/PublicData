import numpy as np
import cv2
import pandas as pd
import re
import os
import glob
import time

# CONFIGURATION
IN_DIR = r"C:\Users\KarlRoesch\physVLA\wavetheory\wavein"
OUT_DIR = r"C:\Users\KarlRoesch\physVLA\wavetheory\waveana"
VID_DIR = r"C:\Users\KarlRoesch\physVLA\wavetheory\wavevides"
NX, NY = 150, 100         
NSTEPS = 400               # Increased steps to see more reflections
SCALE_FACTOR = 180.0
OFFSET_X, OFFSET_Y = 25, 5

# Ensure directories exist
for d in [OUT_DIR, VID_DIR]:
    if not os.path.exists(d):
        os.makedirs(d)

def parse_multi_plane_config(filename):
    with open(filename, 'r') as f:
        content = f.read()
    wall_ids = re.findall(r'(basewall\d+)points', content)
    planes = {}
    for wall_id in wall_ids:
        pts_block = re.search(fr'{wall_id}points\s+([\s\S]*?)(?=basewall\d+points|$)', content)
        if pts_block:
            coords = re.findall(r'\(([-.\d]+),\s+([-.\d]+),\s+([-.\d]+)\)', pts_block.group(1))
            planes[wall_id] = np.array([[float(c[0]), float(c[1])] for c in coords])
    return planes

def run_independent_sim(wall_id, pts, config_name):
    phi, psi = np.zeros((NX, NY)), np.zeros((NX, NY))
    
    # 1. Create the barrier mask
    mask = np.ones((NX, NY), dtype=np.uint8) * 255
    grid_pts = (pts * SCALE_FACTOR + [OFFSET_X, OFFSET_Y]).astype(np.int32)
    cv2.polylines(mask, [grid_pts], False, 0, 2)

    # 2. Define Wave Speed Grid (The "Mirror" Logic)
    # 0.25 is the speed in open space; 0.0 is the speed inside a wall
    c_sq = np.full((NX, NY), 0.25, dtype=np.float32)
    c_sq[mask == 0] = 0.0 

    # Find valid source point
    valid_source = False
    sx, sy = 0, 0
    while not valid_source:
        sx = np.random.randint(10, NX - 10)
        sy = np.random.randint(10, NY - 10)
        if mask[sx, sy] == 255:
            valid_source = True
    
    video_filename = os.path.join(VID_DIR, f"{config_name}_{wall_id}.avi")
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    video_out = cv2.VideoWriter(video_filename, fourcc, 30.0, (NX, NY))

    # 3. Wave Engine with Reflection
    for frame in range(NSTEPS):
        # Source injection
        if frame < 100: 
            phi[sx, sy] += 0.7 * np.sin(2 * np.pi * frame / 10)
            
        # Standard Laplacian (Neighbor interaction)
        laplacian = (np.roll(phi, 1, 0) + np.roll(phi, -1, 0) + 
                     np.roll(phi, 1, 1) + np.roll(phi, -1, 1) - 4 * phi)
        
        # APPLY REFLECTION: Multiply the laplacian by the velocity grid (c_sq)
        # This prevents wave energy from entering the '0.0' zones
        new_phi = (2 * phi - psi + (c_sq * laplacian)) * 0.99 # 0.99 is damping
        
        # Boundaries: Hard kill at the edges of the simulation box
        new_phi[0,:] = new_phi[-1,:] = new_phi[:,0] = new_phi[:,-1] = 0
        
        psi[:], phi[:] = phi, new_phi

        # Visualization
        display_frame = np.uint8(np.clip((phi + 0.5) * 255, 0, 255))
        color_frame = cv2.applyColorMap(display_frame, cv2.COLORMAP_JET)
        
        # Make walls bright white so they are visible over the waves
        color_frame[mask == 0] = [255, 255, 255]
        
        final_frame = cv2.resize(color_frame, (NX, NY))
        video_out.write(final_frame)
        
        cv2.imshow("Wave Simulation", final_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    video_out.release()
    cv2.destroyAllWindows()

    # CSV Data Prep
    intensity = cv2.resize(np.abs(phi), (20, 30)).flatten()
    wall_map = cv2.resize((mask == 0).astype(np.float32), (20, 30)).flatten()
    peak_y, peak_x = np.unravel_index(np.argmax(intensity.reshape(30, 20)), (30, 20))
    ref_x, ref_y = peak_x * (NX/20), peak_y * (NY/30)
    vx, vy = ref_x - sx, ref_y - sy
    scat_x, scat_y = ref_x + vx, ref_y + vy
    
    start_cad = [(ref_x - OFFSET_X)/SCALE_FACTOR, (ref_y - OFFSET_Y)/SCALE_FACTOR]
    end_cad = [(scat_x - OFFSET_X)/SCALE_FACTOR, (scat_y - OFFSET_Y)/SCALE_FACTOR]
    source_cad = [(sx - OFFSET_X)/SCALE_FACTOR, (sy - OFFSET_Y)/SCALE_FACTOR]
    
    return [wall_id, source_cad[0], source_cad[1], start_cad[0], start_cad[1], end_cad[0], end_cad[1]] + intensity.tolist() + wall_map.tolist()

def process_and_poll():
    processed = set()
    while True:
        files = glob.glob(os.path.join(IN_DIR, "config*.txt"))
        for f_path in files:
            if f_path in processed: continue
            
            config_name = os.path.basename(f_path).replace('.txt', '')
            print(f"Processing: {config_name}")
            planes = parse_multi_plane_config(f_path)
            batch_results = []
            
            for wall_id, pts in planes.items():
                row = run_independent_sim(wall_id, pts, config_name)
                batch_results.append(row)
            
            headers = ["Plane_ID", "Source_X", "Source_Y", "Scat_Start_X", "Scat_Start_Y", "Scat_End_X", "Scat_End_Y"]
            headers += [f"p{i}" for i in range(600)] + [f"w{i}" for i in range(600)]
            
            out_file = os.path.join(OUT_DIR, f"scatter_batch_{config_name}.csv")
            pd.DataFrame(batch_results, columns=headers).to_csv(out_file, index=False)
            processed.add(f_path)
        time.sleep(2)

if __name__ == "__main__":
    process_and_poll()