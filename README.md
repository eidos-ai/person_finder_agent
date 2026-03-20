# js_person_finder

Find recent-graduate outcomes (employed, studying, military, or job-seeking) by researching LinkedIn profiles for people listed in a Google Sheet, and write results to CSV/JSON.

## What it does

For each row in your Google Sheet, the app:

1. Searches for the most likely LinkedIn profile for the person.
2. Fetches and reads the profile (or uses Coresignal to retrieve structured candidates).
3. Asks an LLM to output a structured employment/study status JSON.
4. Writes:
   - `results.csv` (incremental)
   - per-person JSON and a per-person log file in `--outdir`

## Modes: `coresignal` vs `local`

Use `--source` to choose how LinkedIn profiles are fetched:

- `coresignal` (default): Uses Coresignal MCP tools (remote structured search/details).
- `local`: Uses a local headless Chromium (Selenium) to scrape LinkedIn search + profile pages.

The agent output schema is the same in both modes.

## Requirements

### 1) Python dependencies

```bash
pip install -r requirements.txt
```

### 2) Google service account credentials

This project expects a VertexAI/Gemini service account JSON file to exist in the repository root with this exact filename:

`gcp-sa Manuel_Correa JS.json`

The code sets `GOOGLE_APPLICATION_CREDENTIALS` to `./gcp-sa Manuel_Correa JS.json`.

### 3) Google Sheet

`main.py` reads from the worksheet named `DataImport` and expects these column headers:

- `SchoolID - Internal Use Only`
- `FirstName`
- `LastName`
- `City`
- `State`
- `SchoolName`
- `Department`
- `Graduation`

## Run the app

### Coresignal mode

Set your Coresignal API key:

```bash
export CORESIGNAL_API_KEY="..."
```

Run:

```bash
python main.py YOUR_SPREADSHEET_ID --source coresignal --start 0 --end 10
```

Useful options:

- `--output results.csv` (default `results.csv`)
- `--outdir outs` (default `outs`)
- `--target-ids path/to/ids.json` (filters by School IDs; disables `--start/--end`)
- `--debug` (verbose logging)

### Local mode (headless browser)

Local mode does not use Coresignal.

1) Install/ensure headless Chromium + Selenium can launch it.
   - The code uses `/Applications/Chromium.app/Contents/MacOS/Chromium`.
   - If you have a different Chromium path, you may need to update `skills/fetch_linkedin.py`.

2) Log in to LinkedIn once (so the scraper can reuse the session).

```bash
python skills/fetch_linkedin.py login
```

Then:
- complete the LinkedIn login in the opened Chromium window
- close the browser tab/window when you’re done

3) Run:

```bash
python main.py YOUR_SPREADSHEET_ID --source local --start 0 --end 10
```

## Output files

- `results.csv`
  - Written incrementally as each person is processed.
  - Includes input fields plus extracted profile/status fields, along with `error`, `cost_usd`, and `elapsed_seconds`.
- `--outdir/<school_id>.json` (or `<First>_<Last>.json` if no `school_id`)
  - Full captured agent response, structured `profile`, and token/cost information when available.
- `--outdir/<school_id>.log`
  - Loguru log file for that specific person run.

## Notes / gotchas

- LinkedIn scraping is brittle by nature (UI changes, bot detection, and rate limits). If local runs fail frequently, prefer `coresignal` mode.
- The app can skip rows already present in the output CSV:
  - it reads existing `school_id` values from `--output` and avoids re-processing them.

