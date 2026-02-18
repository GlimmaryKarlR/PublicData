import os
import time
import glob
import pandas as pd
import numpy as np
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

# --- CONFIGURATION ---
BASE_PATH = r"C:\Users\KarlRoesch\physVLA\wavetheory"
SOURCE_DIR = os.path.join(BASE_PATH, "waveana")
TARGET_DIR = os.path.join(BASE_PATH, "wavescattercurve")

class WaveScatterHandler(FileSystemEventHandler):
    def on_created(self, event):
        if not event.is_directory and "integrated_results_" in event.src_path:
            self.process_csv(event.src_path)

    def process_csv(self, file_path):
        try:
            # Extract the sequential number from the filename (e.g., '1' from 'integrated_results_1.csv')
            file_num = "".join(filter(str.isdigit, os.path.basename(file_path)))
            
            time.sleep(1) # Ensure file is fully written
            df = pd.read_csv(file_path)
            output_rows = []
            
            # Physics constants: y = Ax^2 + Bx
            A, B, STEP_SIZE = -0.4444, 2.6667, 1.5

            for _, row in df.iterrows():
                plane_id = row['Plane_ID']
                try: wall_num = int(''.join(filter(str.isdigit, str(plane_id))))
                except: wall_num = 1
                y_offset = (wall_num - 1) * 10.0
                src_x, src_y = row['Source_X'], row['Source_Y']

                # Generate unique scattering for 10 points on the curve
                wall_x_points = np.linspace(0, 6, 10)
                for i, px in enumerate(wall_x_points):
                    py = (A * (px**2) + B * px)
                    slope = 2 * A * px + B
                    
                    # Normal Vector (N)
                    nx, ny = -slope, 1 
                    mag_n = np.sqrt(nx**2 + ny**2)
                    nx, ny = nx/mag_n, ny/mag_n

                    # Incident Vector (I)
                    ix, iy = px - src_x, py - src_y
                    mag_i = np.sqrt(ix**2 + iy**2)
                    ix, iy = (ix/mag_i, iy/mag_i) if mag_i > 0 else (0, 1)

                    # Specular Reflection: S = I - 2(I . N)N
                    dot_product = (ix * nx) + (iy * ny)
                    sx = ix - 2 * dot_product * nx
                    sy = iy - 2 * dot_product * ny

                    output_rows.append([
                        plane_id, i, px + (sx * STEP_SIZE), (py + y_offset) + (sy * STEP_SIZE), 0.0
                    ])

            # Save with matching sequential number
            out_path = os.path.join(TARGET_DIR, f"scatter_curve_{file_num}.csv")
            pd.DataFrame(output_rows, columns=['Plane_ID', 'Point_Index', 'X', 'Y', 'Z']).to_csv(out_path, index=False)
            print(f"[SUCCESS] Processed scattering for batch {file_num}")

        except Exception as e:
            print(f"[!] Error: {e}")

if __name__ == "__main__":
    if not os.path.exists(TARGET_DIR): os.makedirs(TARGET_DIR)
    handler = WaveScatterHandler()
    observer = Observer()
    observer.schedule(handler, SOURCE_DIR, recursive=False)
    print(f"--- Wave Automation Monitoring Started ---")
    observer.start()
    try:
        while True: time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()
    observer.join()