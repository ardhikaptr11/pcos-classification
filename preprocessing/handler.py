import argparse
from pathlib import Path
import sys


def handle(args: argparse.Namespace):
    from preprocessing import load_dataset, run_preprocessing

    print(
        f"⏳ Loading dataset from:\n- Primary : {args.primary}\n- Secondary : {args.secondary if args.secondary else ''}"
    )

    try:
        df_raw = load_dataset(args.primary, args.secondary)
        df_preprocessed = run_preprocessing(df_raw)

        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df_preprocessed.to_csv(args.output, index=False)

        print(f"✨ Preprocessing completed! Data saved to: {args.output}")
        print(
            f"📊 Final dataset size: {df_preprocessed.shape[0]} rows x {df_preprocessed.shape[1]} columns"
        )

    except Exception as e:
        print(f"❌ Error while running the pipeline: {e}", file=sys.stderr)
        sys.exit(1)
