"""Download all files from a configured Zenodo record into ``data/datasets``."""

from pathlib import Path

import requests

from bp_k_means.utils.logging import logger

# Replace the placeholder with the Zenodo record ID that contains the benchmark datasets.
ZENODO_RECORD_ID = "REPLACE_WITH_RECORD_ID"
DATASET_DIR = Path(__file__).resolve().parents[3] / "data" / "datasets"


def main() -> None:
    """Download each file in the configured Zenodo record if it is not present locally."""
    if ZENODO_RECORD_ID == "REPLACE_WITH_RECORD_ID":
        msg = "Set ZENODO_RECORD_ID in bp_k_means/tools/download_zenodo_dataset.py first."
        raise SystemExit(msg)

    record_response = requests.get(
        f"https://zenodo.org/api/records/{ZENODO_RECORD_ID}",
        timeout=60.0,
    )
    record_response.raise_for_status()
    files = record_response.json().get("files", [])
    if not files:
        msg = f"No files found in Zenodo record {ZENODO_RECORD_ID}."
        raise SystemExit(msg)

    DATASET_DIR.mkdir(parents=True, exist_ok=True)
    for file_info in files:
        filename = Path(str(file_info["key"])).name
        destination = DATASET_DIR / filename
        if destination.exists():
            logger.info("Skipping %s; it already exists.", destination)
            continue

        download_url = file_info["links"]["self"]
        partial = destination.with_name(f"{destination.name}.part")
        logger.info("Downloading %s...", filename)
        try:
            with requests.get(download_url, stream=True, timeout=60.0) as response:
                response.raise_for_status()
                with partial.open("wb") as output_file:
                    for chunk in response.iter_content(chunk_size=1024 * 1024):
                        if chunk:
                            output_file.write(chunk)
            partial.replace(destination)
        except Exception:
            partial.unlink(missing_ok=True)
            raise

        logger.info("Saved %s", destination)


if __name__ == "__main__":
    main()
