from pathlib import Path

import pandas as pd


def convert_manual_annotations_from_frames_to_seconds(parent_folder, fps=3.47):
    raise ValueError("Don't run this more than once!")
    for subfolder in Path(parent_folder).iterdir():
        if subfolder.is_dir():
            # Get the actual subfolder
            subfolder = subfolder / 'behavior'
            print(subfolder)
            for file in subfolder.iterdir():
                # print(file)
                if file.name.endswith('_manual_annotation.csv'):
                    df = pd.read_csv(file)
                    df['start'] = df['start'] / fps
                    df['end'] = df['end'] / fps
                    # Update filename to check
                    # file = file.parent / (file.stem + '_seconds.csv')
                    print(f"Saving to {file}")
                    df.to_csv(file, index=False)


if __name__ == '__main__':
    convert_manual_annotations_from_frames_to_seconds(
        '/scratch/neurobiology/zimmer/fieseler/wbfm_projects/2022-12-12_immob/')
