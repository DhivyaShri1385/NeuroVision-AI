import pandas as pd, sys
sys.path.insert(0, ".")

for s in ["train", "val", "test"]:
    df = pd.read_csv(f"outputs/manifest_{s}.csv")
    print(f"\n{s.upper()} MANIFEST")
    print(f"  Rows     : {len(df)}")
    print(f"  Columns  : {list(df.columns)}")
    print(f"  Classes  : {df['class_name'].value_counts().to_dict()}")
    print(f"  Labels   : {sorted(df['label'].unique().tolist())}")
    print(f"  Sample   : {df['filepath'].iloc[0]}")
    assert df['filepath'].apply(lambda p: __import__('pathlib').Path(p).exists()).all(), \
        f"Some paths in {s} manifest do not exist!"
print("\nAll manifests valid - all file paths exist.")
