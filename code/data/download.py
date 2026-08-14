"""Download annual source files."""

from pathlib import Path
import time
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


BASE_URLS = [
    "http://rank.worldcroquet.org/cgs/data_dir",
    "https://rank.worldcroquet.org/cgs/data_dir",
]

DATA_DIR = Path("data_raw")
DATA_DIR.mkdir(exist_ok=True)


YEARLY_PATTERNS = [
    "game{year}.csv",
    "evnt{year}.csv",
    "data{year}.csv",
    "hidx{year}.csv",
    "out{year}.csv",
]

STATIC_FILES = [
    "names.dat",
    "country.csv",
    "teams.csv",
]


def download_file(filename, output_dir=DATA_DIR, overwrite=False):
    """Download one source file when needed."""

    output_path = output_dir / filename

    if output_path.exists() and output_path.stat().st_size > 0 and not overwrite:
        print("[SKIP] already exists:", filename)
        return True

    last_error = None

    for base_url in BASE_URLS:
        url = base_url + "/" + filename

        try:
            print("[DOWNLOAD]", url)

            request = Request(url, headers={"User-Agent": "Mozilla/5.0"})

            with urlopen(request, timeout=30) as response:
                content = response.read()

            if not content:
                raise RuntimeError("Downloaded content is empty")

            with open(output_path, "wb") as output_file:
                output_file.write(content)

            size_kb = output_path.stat().st_size / 1024
            print("[OK]", filename, "saved,", round(size_kb, 2), "KB")
            return True

        except HTTPError as error:
            last_error = error
            print("[FAILED]", filename, "HTTP error:", error.code)

        except URLError as error:
            last_error = error
            print("[FAILED]", filename, "URL error:", error)

        except Exception as error:
            last_error = error
            print("[FAILED]", filename, "error:", error)

    print("[WARNING] Could not download:", filename)
    print("          Last error:", last_error)
    return False


def download_all_years(
    start_year=1985,
    end_year=2025,
    overwrite=False,
    sleep_seconds=0.2,
):
    """Download expected files for a year range."""

    success_files = []
    failed_files = []

    print("Saving files to:", DATA_DIR.resolve())
    print("Downloading yearly files from", start_year, "to", end_year)
    print("-" * 60)

    for year in range(start_year, end_year + 1):
        print("\nYEAR:", year)

        for pattern in YEARLY_PATTERNS:
            filename = pattern.format(year=year)
            ok = download_file(filename, output_dir=DATA_DIR, overwrite=overwrite)

            if ok:
                success_files.append(filename)
            else:
                failed_files.append(filename)

            time.sleep(sleep_seconds)

    print("\nDownloading static files")
    print("-" * 60)

    for filename in STATIC_FILES:
        ok = download_file(filename, output_dir=DATA_DIR, overwrite=overwrite)

        if ok:
            success_files.append(filename)
        else:
            failed_files.append(filename)

        time.sleep(sleep_seconds)

    print("\n" + "=" * 60)
    print("DOWNLOAD SUMMARY")
    print("=" * 60)
    print("Successful files:", len(success_files))
    print("Failed files:", len(failed_files))

    if failed_files:
        print("\nFailed list:")
        for filename in failed_files:
            print(" -", filename)

    return success_files, failed_files


def check_downloaded_files(start_year=1985, end_year=2025):
    """Report present and missing source files."""

    expected_files = []

    for year in range(start_year, end_year + 1):
        for pattern in YEARLY_PATTERNS:
            expected_files.append(pattern.format(year=year))

    expected_files.extend(STATIC_FILES)

    missing_files = []
    existing_files = []

    for filename in expected_files:
        path = DATA_DIR / filename

        if path.exists() and path.stat().st_size > 0:
            existing_files.append(filename)
        else:
            missing_files.append(filename)

    print("\n" + "=" * 60)
    print("LOCAL FILE CHECK")
    print("=" * 60)
    print("Existing files:", len(existing_files))
    print("Missing files:", len(missing_files))

    if missing_files:
        print("\nMissing list:")
        for filename in missing_files:
            print(" -", filename)

    return existing_files, missing_files


if __name__ == "__main__":
    START_YEAR = 1985
    END_YEAR = 2025

    download_all_years(
        start_year=START_YEAR,
        end_year=END_YEAR,
        overwrite=False,
        sleep_seconds=0.2,
    )

    check_downloaded_files(
        start_year=START_YEAR,
        end_year=END_YEAR,
    )
