import os
import json
import pandas as pd
from datetime import datetime

def archive_vitals():
    records = []
    for file in os.listdir("data"):
        if file.startswith("stream_") and file.endswith(".json"):
            with open(os.path.join("data", file)) as f:
                data = json.load(f)
                records.append(data)

    if records:
        df = pd.DataFrame(records)
        filename = f"data/archived_vitals_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        df.to_csv(filename, index=False)
        print(f"Archived to {filename}")
    else:
        print("No records to archive")

if __name__ == "__main__":
    archive_vitals()