import os
import pandas as pd
from tqdm import tqdm
from datetime import datetime

INPUT_DIR = r"C:"
OUTPUT_FILE = "MASSPA_2017_Time_Series_Ready.csv"

def parse_file(file_path):
    rows = []
    fname = os.path.basename(file_path)
    
    try:
        with open(file_path, 'r', errors='ignore') as f:
            lines = f.readlines()
            if not lines: return None

            # Get timestamp from the first line
            header = ' '.join(lines[0].split())
            ts = datetime.strptime(header, "%b %d, %Y %I:%M:%S %p")

            # Extract mass/pressure data
            for line in lines:
                parts = [p.strip() for p in line.split(',')]
                if len(parts) >= 2:
                    try:
                        m, p = float(parts[0]), float(parts[1])
                        if 0 <= m <= 200:
                            rows.append({
                                'Timestamp': ts, # Combined for code-use
                                'Mass': m,
                                'Partial_Pressure': p,
                                'Source_File': fname # For traceability
                            })
                    except ValueError: continue
    except Exception as e:
        print(f"Error in {fname}: {e}")
    return rows

if __name__ == "__main__":
    files = [os.path.join(INPUT_DIR, f) for f in os.listdir(INPUT_DIR) if f.endswith('.txt')]
    all_data = []
    
    for f_path in tqdm(files, desc="Preparing Colleague's Dataset"):
        data = parse_file(f_path)
        if data: all_data.extend(data)

    if all_data:
        df = pd.DataFrame(all_data)
        # Ensure it is sorted chronologically
        df = df.sort_values(['Timestamp', 'Mass'])
        df.to_csv(OUTPUT_FILE, index=False)
        print(f"\n✅ Professional Dataset Ready: {OUTPUT_FILE}")