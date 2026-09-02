import os
import wfdb
from collections import Counter

DATABASES = {
    "MITDB": "databases/mitdb",
    "SVDB": "databases/svdb"
}

for db_name, db_path in DATABASES.items():

    print("\n" + "=" * 70)
    print(f"{db_name} ANNOTATION CHECK")
    print("=" * 70)

    records = wfdb.get_record_list(
        db_path.split("/")[-1]
    )

    total_annotations = Counter()
    f_records = {}

    for record_name in records:

        try:
            annotation = wfdb.rdann(
                os.path.join(db_path, record_name),
                "atr"
            )

            symbols = annotation.symbol

            counts = Counter(symbols)

            total_annotations.update(symbols)

            if counts.get("F", 0) > 0:
                f_records[record_name] = counts.get("F", 0)

        except Exception as e:
            print(f"Could not read {record_name}: {e}")

    print("\nAll annotation symbols:")
    for symbol, count in total_annotations.most_common():
        print(f"{symbol!r}: {count}")

    print("\nF annotations by record:")

    if f_records:
        for record, count in f_records.items():
            print(f"Record {record}: {count}")

    else:
        print("No F annotations found.")

    print("\nTotal F annotations:")
    print(total_annotations.get("F", 0))