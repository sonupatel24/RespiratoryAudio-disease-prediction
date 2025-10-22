import os
import csv

# Paths
workspace_root = os.path.dirname(os.path.dirname(__file__))
bundled_patients = os.path.join(workspace_root, 'Respiratory_Sound_Database', 'Respiratory_Sound_Database', 'patient_diagnosis.csv')
bundled_audio = os.path.join(workspace_root, 'Respiratory_Sound_Database', 'Respiratory_Sound_Database', 'audio_and_txt_files')
output_dir = os.path.join(workspace_root, 'data')
output_csv = os.path.join(output_dir, 'annotations.csv')

# Map diagnoses to binary label: Healthy=0, others=1
healthy_labels = {'Healthy'}

# Read patient diagnosis map
patient_map = {}
with open(bundled_patients, 'r', newline='', encoding='utf-8') as f:
    reader = csv.reader(f)
    for row in reader:
        if not row:
            continue
        pid = row[0].strip()
        diag = row[1].strip()
        patient_map[pid] = diag

# Collect audio files
records = []
for fname in os.listdir(bundled_audio):
    if not fname.lower().endswith('.wav'):
        continue
    # filenames start like 101_1b1_Al_sc_Meditron.wav -> patient id is before first '_'
    pid = fname.split('_', 1)[0]
    diag = patient_map.get(pid, 'Unknown')
    label = 0 if diag in healthy_labels else 1
    # Use relative path that main.py will resolve (either data/audio/ or bundled path)
    records.append((fname, label))

# Ensure output dir exists
os.makedirs(output_dir, exist_ok=True)

# Write annotations.csv with header filename,label
with open(output_csv, 'w', newline='', encoding='utf-8') as f:
    writer = csv.writer(f)
    writer.writerow(['filename', 'label'])
    for fname, label in records:
        writer.writerow([fname, label])

print(f"Wrote {len(records)} annotations to {output_csv}")
