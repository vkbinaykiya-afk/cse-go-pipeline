# download_ncert.py
# Downloads complete NCERT textbooks (Classes 6-12) chapter by chapter.
# Each chapter is a separate PDF on ncert.nic.in; this script downloads all
# chapters for each book and merges them into a single PDF using PyMuPDF.
#
# ── INSTALL ───────────────────────────────────────────────────────────────────
#   pip install requests pymupdf
#
# ── RUN ───────────────────────────────────────────────────────────────────────
#   python download_ncert.py
#
# Already-complete books are skipped — safe to re-run after interruptions.
# Chapter temp files are cleaned up after merging.
# ─────────────────────────────────────────────────────────────────────────────

import os
import time
import requests
import urllib3
import fitz   # PyMuPDF — for merging chapter PDFs
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

SAVE_DIR    = "./source-docs"
CHAPTER_URL = "https://ncert.nic.in/textbook/pdf/{code}{ch:02d}.pdf"
MAX_CHAPTERS = 20    # try up to 20 chapters per book; stop at first 404
MAX_RETRIES  = 4     # retry on connection resets
RETRY_DELAY  = 2     # seconds between retries

SESSION = requests.Session()
SESSION.verify = False
SESSION.headers.update({"User-Agent": "Mozilla/5.0 (compatible; NCERT-downloader/2.0)"})

# ── BOOK CATALOGUE ────────────────────────────────────────────────────────────
# Format: (ncert_code, save_filename, upsc_priority)

BOOKS = [

    # ── CLASS 6 ───────────────────────────────────────────────────────────────
    ("fecu1",  "Class_06_Science_Curiosity.pdf",                             "HIGH"),
    ("fees1",  "Class_06_Social_Science_Exploring_Society.pdf",              "HIGH"),

    # ── CLASS 7 ───────────────────────────────────────────────────────────────
    ("gesc1",  "Class_07_Science.pdf",                                       "HIGH"),
    ("gess1",  "Class_07_History_Our_Pasts_II.pdf",                          "HIGH"),
    ("gess2",  "Class_07_Geography_Our_Environment.pdf",                     "HIGH"),
    ("gess3",  "Class_07_Civics_Social_and_Political_Life_II.pdf",           "HIGH"),

    # ── CLASS 8 ───────────────────────────────────────────────────────────────
    ("hesc1",  "Class_08_Science.pdf",                                       "HIGH"),
    ("hess2",  "Class_08_History_Our_Pasts_III.pdf",                         "HIGH"),
    ("hess4",  "Class_08_Geography_Resources_and_Development.pdf",           "HIGH"),
    ("hess3",  "Class_08_Civics_Social_and_Political_Life_III.pdf",          "HIGH"),

    # ── CLASS 9 ───────────────────────────────────────────────────────────────
    ("iesc1",  "Class_09_Science.pdf",                                       "HIGH"),
    ("iess3",  "Class_09_History_India_and_the_Contemporary_World_I.pdf",    "HIGH"),
    ("iess1",  "Class_09_Geography_Contemporary_India_I.pdf",                "HIGH"),
    ("iess2",  "Class_09_Economics_Understanding_Economic_Development.pdf",  "HIGH"),
    ("iess4",  "Class_09_Civics_Democratic_Politics_I.pdf",                  "HIGH"),

    # ── CLASS 10 ──────────────────────────────────────────────────────────────
    ("jesc1",  "Class_10_Science.pdf",                                       "HIGH"),
    ("jess3",  "Class_10_History_India_and_the_Contemporary_World_II.pdf",   "HIGH"),
    ("jess1",  "Class_10_Geography_Contemporary_India_II.pdf",               "HIGH"),
    ("jess2",  "Class_10_Economics_Understanding_Economic_Development.pdf",  "HIGH"),
    ("jess4",  "Class_10_Civics_Democratic_Politics_II.pdf",                 "HIGH"),

    # ── CLASS 11 ──────────────────────────────────────────────────────────────
    ("kebo1",  "Class_11_Biology.pdf",                                       "HIGH"),
    ("keph1",  "Class_11_Physics_Part_I.pdf",                                "MEDIUM"),
    ("keph2",  "Class_11_Physics_Part_II.pdf",                               "MEDIUM"),
    ("kech1",  "Class_11_Chemistry_Part_I.pdf",                              "MEDIUM"),
    ("kech2",  "Class_11_Chemistry_Part_II.pdf",                             "MEDIUM"),
    ("kehs1",  "Class_11_History_Themes_in_World_History.pdf",               "HIGH"),
    ("kegy1",  "Class_11_Geography_Fundamentals_of_Physical_Geography.pdf",  "HIGH"),
    ("kegy2",  "Class_11_Geography_India_Physical_Environment.pdf",          "HIGH"),
    ("keec1",  "Class_11_Economics_Indian_Economic_Development.pdf",         "HIGH"),
    ("keps1",  "Class_11_Political_Science_Political_Theory.pdf",            "HIGH"),
    ("keps2",  "Class_11_Political_Science_Indian_Constitution_at_Work.pdf", "HIGH"),

    # ── CLASS 12 ──────────────────────────────────────────────────────────────
    ("lebo1",  "Class_12_Biology.pdf",                                       "HIGH"),
    ("leph1",  "Class_12_Physics_Part_I.pdf",                                "MEDIUM"),
    ("leph2",  "Class_12_Physics_Part_II.pdf",                               "MEDIUM"),
    ("lech1",  "Class_12_Chemistry_Part_I.pdf",                              "MEDIUM"),
    ("lech2",  "Class_12_Chemistry_Part_II.pdf",                             "MEDIUM"),
    ("lehs1",  "Class_12_History_Themes_in_Indian_History_I.pdf",            "HIGH"),
    ("lehs2",  "Class_12_History_Themes_in_Indian_History_II.pdf",           "HIGH"),
    ("lehs3",  "Class_12_History_Themes_in_Indian_History_III.pdf",          "HIGH"),
    ("legy1",  "Class_12_Geography_Fundamentals_of_Human_Geography.pdf",     "HIGH"),
    ("legy2",  "Class_12_Geography_India_People_and_Economy.pdf",            "HIGH"),
    ("leec1",  "Class_12_Economics_Macroeconomics.pdf",                      "HIGH"),
    ("leec2",  "Class_12_Economics_Microeconomics.pdf",                      "HIGH"),
    ("leps1",  "Class_12_Political_Science_Contemporary_World_Politics.pdf", "HIGH"),
    ("leps2",  "Class_12_Political_Science_Politics_in_India_since_Independence.pdf", "HIGH"),
]


# ── DOWNLOAD HELPERS ──────────────────────────────────────────────────────────

def download_chapter(url, save_path):
    """
    Download a single chapter PDF with retry on connection errors.
    Returns True on success, False on 404, raises on other errors.
    """
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            r = SESSION.get(url, stream=True, timeout=30)

            if r.status_code == 404:
                return False   # chapter doesn't exist — book ends here

            r.raise_for_status()

            with open(save_path, "wb") as f:
                for chunk in r.iter_content(chunk_size=16384):
                    f.write(chunk)
            return True

        except (requests.exceptions.ConnectionError,
                requests.exceptions.Timeout) as e:
            if attempt < MAX_RETRIES:
                time.sleep(RETRY_DELAY * attempt)
            else:
                raise e


def merge_chapters(chapter_paths, output_path):
    """Merge a list of chapter PDFs into one book PDF using PyMuPDF."""
    merged = fitz.open()
    for path in chapter_paths:
        chapter_doc = fitz.open(path)
        merged.insert_pdf(chapter_doc)
        chapter_doc.close()
    merged.save(output_path)
    merged.close()


def download_book(code, filename, priority, index, total):
    save_path  = os.path.join(SAVE_DIR, filename)
    label      = filename.replace(".pdf", "").replace("_", " ")
    temp_dir   = os.path.join(SAVE_DIR, f"_tmp_{code}")

    # Skip only if existing PDF has enough pages to be a full book (>=30pp)
    if os.path.exists(save_path):
        pages = _count_pages(save_path)
        if pages >= 30:
            size_mb = os.path.getsize(save_path) / (1024 * 1024)
            print(f"  —  [{index:02d}/{total}] Already complete ({pages}pp, {size_mb:.1f}MB): {filename}")
            return "skipped"
        else:
            print(f"  ↻  [{index:02d}/{total}] Preview detected ({pages}pp) — re-downloading full book")

    print(f"  ↓  [{index:02d}/{total}] [{priority}] {label}")

    os.makedirs(temp_dir, exist_ok=True)
    chapter_paths = []

    for ch in range(1, MAX_CHAPTERS + 1):
        url       = CHAPTER_URL.format(code=code, ch=ch)
        ch_path   = os.path.join(temp_dir, f"ch{ch:02d}.pdf")

        try:
            ok = download_chapter(url, ch_path)
        except Exception as e:
            print(f"       Ch{ch:02d}: error — {e}")
            break

        if not ok:
            break   # 404 = no more chapters

        pages = _count_pages(ch_path)
        size  = os.path.getsize(ch_path) / 1024
        print(f"       Ch{ch:02d}: {pages}pp  ({size:.0f}KB)")
        chapter_paths.append(ch_path)
        time.sleep(0.3)

    if not chapter_paths:
        print(f"       No chapters found for {code} — skipping.")
        _cleanup(temp_dir)
        return "failed"

    # Merge all chapters into one book PDF
    print(f"       Merging {len(chapter_paths)} chapters ...", end="", flush=True)
    merge_chapters(chapter_paths, save_path)
    total_pages = _count_pages(save_path)
    size_mb     = os.path.getsize(save_path) / (1024 * 1024)
    print(f" → {total_pages}pp  {size_mb:.1f}MB  ✓")

    _cleanup(temp_dir)
    return "ok"


def _count_pages(pdf_path):
    try:
        doc = fitz.open(pdf_path)
        n   = len(doc)
        doc.close()
        return n
    except Exception:
        return 0


def _cleanup(temp_dir):
    """Remove temp chapter files."""
    import shutil
    if os.path.exists(temp_dir):
        shutil.rmtree(temp_dir)


# ── MAIN ──────────────────────────────────────────────────────────────────────

def main():
    os.makedirs(SAVE_DIR, exist_ok=True)
    print(f"Saving to: {os.path.abspath(SAVE_DIR)}")
    print(f"Books in catalogue: {len(BOOKS)}\n")

    success = skipped = failed = 0
    failed_list = []

    for i, (code, filename, priority) in enumerate(BOOKS, start=1):
        result = download_book(code, filename, priority, i, len(BOOKS))
        if result == "ok":
            success += 1
        elif result == "skipped":
            skipped += 1
        else:
            failed += 1
            failed_list.append(filename)
        time.sleep(0.5)

    print(f"\n{'═'*52}")
    print(f"  DOWNLOAD COMPLETE")
    print(f"  Downloaded  : {success}  (new)")
    print(f"  Skipped     : {skipped}  (already complete)")
    print(f"  Failed      : {failed}")
    if failed_list:
        print(f"\n  Failed:")
        for f in failed_list:
            print(f"    • {f}")
    print(f"{'═'*52}")
    print(f"\nNext step → run ingest.py to rebuild the ChromaDB knowledge base.")


if __name__ == "__main__":
    main()
