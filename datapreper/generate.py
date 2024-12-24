import argparse
from pathlib import Path
from multiprocessing import cpu_count
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm

def preprocess_dataset(args):
    if args.mode == "vad":
        from silero_vad import process_wav
    elif args.mode == "ppw":
        from pseudo_whisper import process_wav
    args.out_dir.mkdir(parents=True, exist_ok=True)

    futures = []
    executor = ProcessPoolExecutor(max_workers=cpu_count())
    print(f"VAD for audio in {args.in_dir}")
    list_list = list(args.in_dir.rglob("*.wav"))
    for i, in_path in enumerate(list_list):
        # if i % 2 == 1:
        relative_path = in_path.relative_to(args.in_dir)
        out_path = args.out_dir / relative_path
        out_path.parent.mkdir(parents=True, exist_ok=True)
        futures.append(
            executor.submit(process_wav, in_path, out_path, args.sample_rate)
        )

    results = [future.result() for future in tqdm(futures)]

    lengths = {path.stem: length for path, length in results}
    seconds = sum(lengths.values())
    hours = seconds / 3600
    print(f"Wrote {len(lengths)} utterances ({hours:.2f} hours)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Resample an audio dataset.")
    parser.add_argument(
        "in_dir", metavar="in-dir", help="path to the dataset directory.", type=Path
    )
    parser.add_argument(
        "out_dir", metavar="out-dir", help="path to the output directory.", type=Path
    )
    parser.add_argument(
        '--mode', choices=['vad', 'ppw'], required=True, help="Select processing mode: 'vad' or 'psuedo_whisper'"
    )
    parser.add_argument(
        "--sample-rate",
        help="target sample rate (default 16kHz)",
        type=int,
        default=16000,
    )
    args = parser.parse_args()
    preprocess_dataset(args)