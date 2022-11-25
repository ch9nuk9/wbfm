import argparse
import re
import pandas as pd


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Build interactive gridplot')
    parser.add_argument('--csv_path', '-c', default=None,
                        help='path to config file')
    parser.add_argument('--DEBUG', default=False,
                        help='')
    args = parser.parse_args()
    csv_path = args.csv_path
    DEBUG = args.DEBUG

    # Read and replace
    df = pd.read_csv(csv_path)
    regex = "\s\d\s"
    for name in df.index:
        if isinstance(df.at[name, 'List ID'], str):
            matches = re.finditer(regex, df.at[name, 'List ID'], re.MULTILINE)
            for m in matches:
                print(f"Fixing entry: {df.at[name, 'List ID']}")
                df.at[name, 'List ID'] = int(m.group())
                break

    # Overwrite the file
    if not DEBUG:
        df.to_csv(csv_path)
