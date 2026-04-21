from __future__ import annotations

import argparse
from pathlib import Path

from baseline.text_cleanup import postprocess_markdown


def main() -> None:
    p = argparse.ArgumentParser(description="Re-apply baseline postprocessing to existing Markdown.")
    p.add_argument("--input", type=Path, required=True, help="Path to input .md")
    p.add_argument("--output", type=Path, default=None, help="Path to output .md (default: overwrite input)")
    p.add_argument(
        "--no-strip-noise",
        action="store_true",
        help="Disable noise/page-marker stripping (keeps more lines).",
    )
    args = p.parse_args()

    inp: Path = args.input
    if not inp.is_file():
        raise FileNotFoundError(f"Input not found: {inp}")

    outp: Path = args.output if args.output is not None else inp

    text = inp.read_text(encoding="utf-8")
    processed = postprocess_markdown(text, strip_noise=not args.no_strip_noise)

    outp.parent.mkdir(parents=True, exist_ok=True)
    outp.write_text(processed, encoding="utf-8")


if __name__ == "__main__":
    main()

