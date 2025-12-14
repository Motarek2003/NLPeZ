from pathlib import Path

cleaned_dir = Path("data/cleaned")

for p in cleaned_dir.iterdir():
    print("RAW NAME:", repr(p.name))
    clean_name = p.name.encode("utf-8").decode("utf-8").strip()
    if p.name != clean_name:
        print(f"Renaming → {clean_name}")
        p.rename(cleaned_dir / clean_name)
