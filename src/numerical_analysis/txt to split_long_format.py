# txt_to_split_long_format.py (Expanded to handle B, L, and U, with overwrite prompt)

import os
import pandas as pd

input_dir = 'raw_txt'  # Example input directory
output_files = {
    'B': 'gemini_db_long_B.csv',
    'L': 'gemini_db_long_L.csv',
    'U': 'gemini_db_long_U.csv'
}

def normalize_spectrum(df, light_source):
    if light_source == 'B':
        target_wavelength = 650
        target_intensity = 50000
    elif light_source == 'L':
        target_wavelength = 450
        target_intensity = 50000
    elif light_source == 'U':
        peak_region = df[(df['wavelength'] >= 810) & (df['wavelength'] <= 812)]
        if not peak_region.empty:
            max_idx = peak_region['intensity'].idxmax()
            max_intensity = df.loc[max_idx, 'intensity']
            scale_factor = 15000 / max_intensity
            df['intensity'] = df['intensity'] * scale_factor
            return df
        else:
            print("⚠️ No peak found in 810–812 nm range for UV normalization.")
            return df
            print("⚠️ No peak found in 810–812 nm range for UV normalization.")
            return df
    else:
        raise ValueError(f"Unknown light source: {light_source}")
    closest_idx = (df['wavelength'] - target_wavelength).abs().idxmin()
    scale_factor = target_intensity / df.loc[closest_idx, 'intensity']
    df['intensity'] = df['intensity'] * scale_factor
    return df

# Process each spectrum file
for file in os.listdir(input_dir):
    if file.endswith('.txt'):
        file_path = os.path.join(input_dir, file)
        gem_name = os.path.splitext(file)[0]

        # Identify light source from filename
        if 'B' in gem_name.upper():
            light_source = 'B'
        elif 'L' in gem_name.upper():
            light_source = 'L'
        elif 'U' in gem_name.upper():
            light_source = 'U'
        else:
            print(f"⚠️ Unknown light source in filename: {file}")
            continue

        # Load and normalize
        df = pd.read_csv(file_path, sep='[\s,]+', header=None, names=['wavelength', 'intensity'], skiprows=1, engine='python')
        df['full_name'] = gem_name
        df = normalize_spectrum(df, light_source)

        # Append to appropriate DB file
        output_file = output_files[light_source]
        if os.path.exists(output_file):
            existing_df = pd.read_csv(output_file)
            if gem_name in existing_df['full_name'].values:
                user_choice = input(f"⚠️ {gem_name} already exists in {output_file}. Overwrite (o) or skip (s)? ").strip().lower()
                if user_choice == 's':
                    print(f"⏭️ Skipped {gem_name}.")
                    continue
                elif user_choice == 'o':
                    existing_df = existing_df[existing_df['full_name'] != gem_name]
                    print(f"♻️ Overwriting {gem_name}.")
                else:
                    print(f"⚠️ Invalid choice. Skipping {gem_name}.")
                    continue
            combined_df = pd.concat([existing_df, df], ignore_index=True)
            print(f"✅ Updated {output_file} with {gem_name}.")
        else:
            combined_df = df
            print(f"✅ Created new {output_file} with {gem_name}.")

        combined_df.to_csv(output_file, index=False)
        print(f"✅ Database saved to {output_file}.")

