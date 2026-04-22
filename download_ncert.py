# download_ncert.py
# Downloads all NCERT PDFs (Classes 6-12) needed for the CSE-GO pipeline.
# Subjects: Science, Geography, History, Economics, Civics/Political Science
#
# ── INSTALL ───────────────────────────────────────────────────────────────────
#   pip install requests
#
# ── RUN ───────────────────────────────────────────────────────────────────────
#   python download_ncert.py
#
# Files are saved to ./source-docs/ with clean descriptive names.
# Already-downloaded files are skipped automatically — safe to re-run.
# ─────────────────────────────────────────────────────────────────────────────

import os
import requests
import time

SAVE_DIR = "./source-docs"
BASE_URL = "https://ncert.nic.in/textbook/pdf/{code}ps.pdf"

# ── BOOK CATALOGUE ────────────────────────────────────────────────────────────
# Format: (ncert_code, save_filename, upsc_priority)
# Priority: HIGH = core UPSC syllabus | MEDIUM = useful | LOW = less tested
#
# The NCERT code maps directly to the PDF URL:
#   code "iesc1" → https://ncert.nic.in/textbook/pdf/iesc1ps.pdf

BOOKS = [

    # ── CLASS 6 ───────────────────────────────────────────────────────────────
    ("fesc1", "Class_06_Science.pdf",                        "HIGH"),
    ("fess1", "Class_06_History_Our_Pasts_I.pdf",            "HIGH"),
    ("fess2", "Class_06_Geography_The_Earth.pdf",            "HIGH"),
    ("fess3", "Class_06_Civics_Social_and_Political_Life_I.pdf", "HIGH"),

    # ── CLASS 7 ───────────────────────────────────────────────────────────────
    ("gesc1", "Class_07_Science.pdf",                        "HIGH"),
    ("gess1", "Class_07_History_Our_Pasts_II.pdf",           "HIGH"),
    ("gess2", "Class_07_Geography_Our_Environment.pdf",      "HIGH"),
    ("gess3", "Class_07_Civics_Social_and_Political_Life_II.pdf", "HIGH"),

    # ── CLASS 8 ───────────────────────────────────────────────────────────────
    ("hesc1", "Class_08_Science.pdf",                        "HIGH"),
    ("hess2", "Class_08_History_Our_Pasts_III.pdf",          "HIGH"),
    ("hess4", "Class_08_Geography_Resources_and_Development.pdf", "HIGH"),
    ("hess3", "Class_08_Civics_Social_and_Political_Life_III.pdf", "HIGH"),

    # ── CLASS 9 ───────────────────────────────────────────────────────────────
    ("iesc1", "Class_09_Science.pdf",                        "HIGH"),
    ("iess3", "Class_09_History_India_and_the_Contemporary_World_I.pdf", "HIGH"),
    ("iess1", "Class_09_Geography_Contemporary_India_I.pdf", "HIGH"),
    ("iess2", "Class_09_Economics_Understanding_Economic_Development.pdf", "HIGH"),
    ("iess4", "Class_09_Civics_Democratic_Politics_I.pdf",   "HIGH"),

    # ── CLASS 10 ──────────────────────────────────────────────────────────────
    ("jesc1", "Class_10_Science.pdf",                        "HIGH"),
    ("jess3", "Class_10_History_India_and_the_Contemporary_World_II.pdf", "HIGH"),
    ("jess1", "Class_10_Geography_Contemporary_India_II.pdf","HIGH"),
    ("jess2", "Class_10_Economics_Understanding_Economic_Development.pdf", "HIGH"),
    ("jess4", "Class_10_Civics_Democratic_Politics_II.pdf",  "HIGH"),

    # ── CLASS 11 ──────────────────────────────────────────────────────────────
    # Science — Biology is HIGH priority for UPSC (ecology, environment)
    # Physics & Chemistry are MEDIUM (occasionally tested)
    ("kebo1", "Class_11_Biology.pdf",                        "HIGH"),
    ("keph1", "Class_11_Physics_Part_I.pdf",                 "MEDIUM"),
    ("keph2", "Class_11_Physics_Part_II.pdf",                "MEDIUM"),
    ("kech1", "Class_11_Chemistry_Part_I.pdf",               "MEDIUM"),
    ("kech2", "Class_11_Chemistry_Part_II.pdf",              "MEDIUM"),

    ("kehs1", "Class_11_History_Themes_in_World_History.pdf","HIGH"),
    ("kegy1", "Class_11_Geography_Fundamentals_of_Physical_Geography.pdf", "HIGH"),
    ("keec1", "Class_11_Economics_Indian_Economic_Development.pdf", "HIGH"),
    ("keps1", "Class_11_Political_Science_Political_Theory.pdf", "HIGH"),
    ("keps2", "Class_11_Political_Science_Indian_Constitution_at_Work.pdf", "HIGH"),

    # ── CLASS 12 ──────────────────────────────────────────────────────────────
    ("lebo1", "Class_12_Biology.pdf",                        "HIGH"),
    ("leph1", "Class_12_Physics_Part_I.pdf",                 "MEDIUM"),
    ("leph2", "Class_12_Physics_Part_II.pdf",                "MEDIUM"),
    ("lech1", "Class_12_Chemistry_Part_I.pdf",               "MEDIUM"),
    ("lech2", "Class_12_Chemistry_Part_II.pdf",              "MEDIUM"),

    ("lehs1", "Class_12_History_Themes_in_Indian_History_I.pdf",   "HIGH"),
    ("lehs2", "Class_12_History_Themes_in_Indian_History_II.pdf",  "HIGH"),
    ("lehs3", "Class_12_History_Themes_in_Indian_History_III.pdf", "HIGH"),
    ("legy1", "Class_12_Geography_Fundamentals_of_Human_Geography.pdf", "HIGH"),
    ("legy2", "Class_12_Geography_India_People_and_Economy.pdf",   "HIGH"),
    ("leec1", "Class_12_Economics_Macroeconomics.pdf",             "HIGH"),
    ("leec2", "Class_12_Economics_Microeconomics.pdf",             "HIGH"),
    ("leps1", "Class_12_Political_Science_Contemporary_World_Politics.pdf", "HIGH"),
    ("leps2", "Class_12_Political_Science_Politics_in_India_since_Independence.pdf", "HIGH"),
]


def download_file(url, save_path, label):
    """
    Downloads a single file from url to save_path.
    Shows a simple progress indicator.
    Returns True on success, False on failure.
    """
    try:
        # stream=True means we download in chunks instead of loading the whole
        # file into memory — important for large PDFs
        response = requests.get(url, stream=True, timeout=30)

        if response.status_code == 404:
            print(f"  NOT FOUND (404) — skipping: {label}")
            return False

        response.raise_for_status()   # raises an error for any other bad status code

        total_bytes = int(response.headers.get("content-length", 0))
        downloaded  = 0

        with open(save_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
                downloaded += len(chunk)

        size_mb = downloaded / (1024 * 1024)
        print(f"  ✓  {label}  ({size_mb:.1f} MB)")
        return True

    except requests.exceptions.ConnectionError:
        print(f"  ✗  CONNECTION ERROR — {label}")
        return False
    except requests.exceptions.Timeout:
        print(f"  ✗  TIMEOUT — {label}")
        return False
    except Exception as e:
        print(f"  ✗  ERROR — {label}: {e}")
        return False


def main():
    os.makedirs(SAVE_DIR, exist_ok=True)

    print(f"Saving PDFs to: {os.path.abspath(SAVE_DIR)}")
    print(f"Total books in catalogue: {len(BOOKS)}\n")

    success  = 0
    skipped  = 0
    failed   = 0
    failed_list = []

    for i, (code, filename, priority) in enumerate(BOOKS, start=1):
        save_path = os.path.join(SAVE_DIR, filename)
        url       = BASE_URL.format(code=code)
        label     = filename.replace(".pdf", "").replace("_", " ")

        # Skip if already downloaded
        if os.path.exists(save_path):
            size_mb = os.path.getsize(save_path) / (1024 * 1024)
            print(f"  —  [{i:02d}/{len(BOOKS)}] Already exists ({size_mb:.1f} MB): {filename}")
            skipped += 1
            continue

        print(f"  ↓  [{i:02d}/{len(BOOKS)}] [{priority}] Downloading: {filename}")

        ok = download_file(url, save_path, label)

        if ok:
            success += 1
        else:
            failed += 1
            failed_list.append(filename)
            # Remove partial file if download failed
            if os.path.exists(save_path):
                os.remove(save_path)

        # Small pause between requests — be polite to the NCERT server
        time.sleep(0.5)

    # ── Summary ───────────────────────────────────────────────────────────────
    print(f"\n{'═'*52}")
    print(f"  DOWNLOAD COMPLETE")
    print(f"  Downloaded  : {success}")
    print(f"  Skipped     : {skipped}  (already existed)")
    print(f"  Failed      : {failed}")
    if failed_list:
        print(f"\n  Failed files:")
        for f in failed_list:
            print(f"    • {f}")
    print(f"{'═'*52}")
    print(f"\nNext step → run ingest.py to build the ChromaDB knowledge base.")


if __name__ == "__main__":
    main()
