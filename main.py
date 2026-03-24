#!/usr/bin/env python3
"""
Main entry point — loads persons from a Google Sheets spreadsheet,
processes each through the LinkedIn profile agent, and writes results
to a CSV file or a "Results" worksheet incrementally.
"""

import argparse
import asyncio
import csv
import json
import os
import sys
import time
from collections.abc import Callable
from typing import Optional

from loguru import logger

from agent import PersonInitialDescription, PersonProfile, run_agent
from coresignal import get_credits_remaining
from google_sheets import (
    append_sheet_row,
    get_or_create_results_sheet,
    get_processed_ids_from_sheet,
    load_persons,
)


# ── CSV columns ────────────────────────────────────────────

INPUT_FIELDS = [
    "school_id", "first_name", "last_name", "city", "state",
    "school_name", "school_department", "graduation_date",
]

PROFILE_FIELDS = [
    "confidence",
    "has_job", "job_type", "job_related", "job_schedule",
    "self_employed", "employer_name", "employer_city", "employer_state",
    "continuing_education", "current_education_institution", "current_education_field",
    "sources",
]

CSV_COLUMNS = INPUT_FIELDS + PROFILE_FIELDS + ["error", "cost_usd", "elapsed_seconds"]


# ── CSV helpers ────────────────────────────────────────────

def _ensure_csv_header(path: str) -> None:
    """Write the CSV header row if the file does not exist yet."""
    if not os.path.exists(path):
        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
            writer.writeheader()


def _append_csv_row(path: str, row: dict) -> None:
    """Append a single row to the CSV file and close immediately."""
    with open(path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        writer.writerow(row)


def _load_existing_output_ids(path: str) -> set[str]:
    """Load already-processed school_id values from an output CSV."""
    if not os.path.exists(path):
        return set()

    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        if "school_id" not in (reader.fieldnames or []):
            return set()

        existing_ids: set[str] = set()
        for row in reader:
            school_id = (row.get("school_id") or "").strip()
            if school_id:
                existing_ids.add(school_id)
        return existing_ids


def _build_csv_row(
    school_id: str,
    person: PersonInitialDescription,
    profile: Optional[PersonProfile],
    confidence: str,
    error: str,
    cost_usd: float,
    elapsed: float,
) -> dict:
    """Build a CSV row dict from person input and agent result."""
    row: dict = {
        "school_id": school_id,
        "first_name": person.first_name,
        "last_name": person.last_name,
        "city": person.city,
        "state": person.state,
        "school_name": person.school_name,
        "school_department": person.school_department,
        "graduation_date": person.graduation_date,
    }

    if profile is not None:
        row.update({
            "confidence": confidence,
            "has_job": profile.has_job,
            "job_type": profile.job_type,
            "job_related": profile.job_related,
            "job_schedule": profile.job_schedule,
            "self_employed": profile.self_employed,
            "employer_name": profile.employer_name,
            "employer_city": profile.employer_city,
            "employer_state": profile.employer_state,
            "continuing_education": profile.continuing_education,
            "current_education_institution": profile.current_education_institution,
            "current_education_field": profile.current_education_field,
            "sources": "; ".join(profile.sources),
        })
    else:
        for field in PROFILE_FIELDS:
            row[field] = ""

    row["error"] = error
    row["cost_usd"] = f"{cost_usd:.6f}"
    row["elapsed_seconds"] = f"{elapsed:.1f}"
    return row


def _save_person_output(
    school_id: str,
    person: PersonInitialDescription,
    profile: Optional[PersonProfile],
    confidence: str,
    response: str,
    consumption,
    error: str,
    cost_usd: float,
    elapsed: float,
    outdir: str,
) -> None:
    """Save full output for a person to a JSON file named by school_id."""
    os.makedirs(outdir, exist_ok=True)
    filename = f"{school_id}.json" if school_id else f"{person.first_name}_{person.last_name}.json"
    filepath = os.path.join(outdir, filename)

    data = {
        "input": {
            "school_id": school_id,
            "first_name": person.first_name,
            "last_name": person.last_name,
            "city": person.city,
            "state": person.state,
            "school_name": person.school_name,
            "school_department": person.school_department,
            "graduation_date": person.graduation_date,
        },
        "confidence": confidence,
        "profile": {
            "has_job": profile.has_job,
            "job_type": profile.job_type,
            "job_related": profile.job_related,
            "job_schedule": profile.job_schedule,
            "self_employed": profile.self_employed,
            "employer_name": profile.employer_name,
            "employer_city": profile.employer_city,
            "employer_state": profile.employer_state,
            "continuing_education": profile.continuing_education,
            "current_education_institution": profile.current_education_institution,
            "current_education_field": profile.current_education_field,
            "sources": profile.sources,
        } if profile else None,
        "response": response,
        "error": error,
        "cost_usd": cost_usd,
        "elapsed_seconds": elapsed,
    }

    if consumption:
        data["consumption"] = {
            "tool_calls_used": consumption.tool_calls_used,
            "max_llm_calls": consumption.max_llm_calls,
            "models": {
                model: {
                    "input_tokens": usage.input_tokens,
                    "output_tokens": usage.output_tokens,
                    "thinking_tokens": usage.thinking_tokens,
                    "cached_tokens": usage.cached_tokens,
                    "cost_usd": usage.cost_usd,
                }
                for model, usage in consumption.models.items()
            },
        }

    with open(filepath, "w") as f:
        json.dump(data, f, indent=2)
    logger.info("Saved output to {}", filepath)


# ── per-person processing ─────────────────────────────────

async def process_person(
    school_id: str,
    person: PersonInitialDescription,
    row_writer: Callable[[dict], None],
    outdir: str,
    source: str,
) -> tuple[float, float, bool, int]:
    """Process one person, write the result row via row_writer.

    Returns: (cost, elapsed, found, coresignal_calls).
    """
    name = f"{person.first_name} {person.last_name}"

    # Add a per-person log file handler (captures everything shown on terminal)
    os.makedirs(outdir, exist_ok=True)
    log_name = school_id if school_id else f"{person.first_name}_{person.last_name}"
    log_path = os.path.join(outdir, f"{log_name}.log")
    log_id = logger.add(log_path, level="DEBUG", mode="w")

    start = time.time()
    total_cost = 0.0
    coresignal_count = 0
    error = ""
    found = False
    confidence = ""
    profile: Optional[PersonProfile] = None
    response_text = ""
    consumption_data = None

    try:
        result = await run_agent(person, source=source)
        found = result.found
        confidence = result.confidence.value
        profile = result.profile
        response_text = result.response
        consumption_data = result.consumption

        logger.info("RESULT: {}\n{}", name, result.profile)
        logger.debug("Final Response: {}\n{}", name, result.response)

        for model, usage in consumption_data.models.items():
            cost = usage.cost_usd
            total_cost += cost
            logger.info(
                "{} | in={} | out={} | thinking={} | cost=${:.6f}",
                model, usage.input_tokens, usage.output_tokens,
                usage.thinking_tokens, cost,
            )
        logger.info("Tool calls: {}", consumption_data.tool_calls_used)
        if consumption_data.coresignal_calls:
            for tool_name, count in consumption_data.coresignal_calls.items():
                logger.info("Coresignal | {} = {}", tool_name, count)
            coresignal_count = sum(consumption_data.coresignal_calls.values())

    except Exception as e:
        logger.exception(e)
        logger.error("{} -> ERROR: {}", name, e)
        error = str(e)

    elapsed = time.time() - start
    logger.info("{} completed in {:.1f}s", name, elapsed)

    # Write result row immediately (CSV or spreadsheet)
    row = _build_csv_row(school_id, person, profile, confidence, error, total_cost, elapsed)
    row_writer(row)

    # Save per-person output file
    _save_person_output(school_id, person, profile, confidence, response_text, consumption_data, error, total_cost, elapsed, outdir)

    # Remove per-person log handler
    logger.remove(log_id)

    return total_cost, elapsed, found, coresignal_count


# ── main ───────────────────────────────────────────────────

async def main() -> None:
    parser = argparse.ArgumentParser(
        description="Process persons from a Google Sheet and write LinkedIn results to CSV or the Results sheet.",
    )
    parser.add_argument("spreadsheet_id", help="Google Sheets spreadsheet ID")
    parser.add_argument("--start", type=int, default=None, help="Start row index (0-based)")
    parser.add_argument("--end", type=int, default=None, help="End row index (exclusive)")
    parser.add_argument(
        "--target-ids",
        default=None,
        help="Path to JSON file with School IDs to process (disables --start/--end).",
    )
    parser.add_argument("--output", default="results.csv", help="Output CSV path (default: results.csv); only used with --no-write-to-sheet")
    parser.add_argument(
        "--no-write-to-sheet",
        action="store_true",
        dest="no_write_to_sheet",
        help="Write results to a local CSV instead of the 'Results' worksheet in the spreadsheet.",
    )
    parser.add_argument("--outdir", default="outs", help="Directory for per-person output files (default: outs)")
    parser.add_argument(
        "--source",
        choices=["coresignal", "local"],
        default=os.environ.get("PERSON_FINDER_SOURCE", "local"),
        help="Where to fetch LinkedIn profiles from: `local` (default, headless browser) or `coresignal`.",
    )
    parser.add_argument("--debug", action="store_true", help="Enable DEBUG-level logging")

    args = parser.parse_args()

    # ── configure loguru ──────────────────────────────────
    logger.remove()  # remove default handler
    log_level = "DEBUG" if args.debug else "INFO"
    logger.add(sys.stderr, level=log_level)

    if args.target_ids and (args.start is not None or args.end is not None):
        raise ValueError("When --target-ids is provided, --start and --end cannot be used.")

    target_ids: set[str] | None = None
    if args.target_ids:
        with open(args.target_ids) as f:
            payload = json.load(f)
        if not isinstance(payload, list):
            raise ValueError(f"--target-ids file must contain a JSON array: {args.target_ids}")
        target_ids = {str(item).strip() for item in payload if str(item).strip()}
        logger.info("Loaded {} target School IDs from {}", len(target_ids), args.target_ids)

    # ── load persons from spreadsheet ─────────────────────
    logger.info("Loading persons from spreadsheet: {}", args.spreadsheet_id)
    load_start = None if target_ids is not None else args.start
    load_end = None if target_ids is not None else args.end
    persons = load_persons(args.spreadsheet_id, start=load_start, end=load_end)

    if target_ids is not None:
        persons = [(school_id, person) for school_id, person in persons if str(school_id).strip() in target_ids]
        logger.info(
            "Loaded {} persons after applying target IDs filter (full sheet loaded first)",
            len(persons),
        )
    else:
        logger.info("Loaded {} persons (start={}, end={})", len(persons), args.start, args.end)

    # ── resume: skip already-processed school_ids ─────────
    if not args.no_write_to_sheet:
        if args.start is not None or args.end is not None:
            existing_ids: set[str] = set()
            logger.info("Sheet output: --start/--end set; resume from Results sheet disabled.")
        else:
            existing_ids = get_processed_ids_from_sheet(args.spreadsheet_id)
            if existing_ids:
                logger.info("Sheet output: resuming; {} school_id(s) already in Results sheet.", len(existing_ids))
        output_dest = f"spreadsheet {args.spreadsheet_id} (Results)"
        get_or_create_results_sheet(args.spreadsheet_id, CSV_COLUMNS)

        def row_writer(row: dict) -> None:
            append_sheet_row(args.spreadsheet_id, row, CSV_COLUMNS)

    else:
        existing_ids = _load_existing_output_ids(args.output)
        output_dest = args.output
        _ensure_csv_header(args.output)

        def row_writer(row: dict) -> None:
            _append_csv_row(args.output, row)

    if existing_ids:
        before = len(persons)
        persons = [(school_id, person) for school_id, person in persons if str(school_id).strip() not in existing_ids]
        logger.info(
            "Skipped {} persons already present in {}. Remaining: {}",
            before - len(persons),
            output_dest,
            len(persons),
        )

    logger.info("Writing results to: {}", output_dest)

    if not persons:
        logger.warning("No persons to process.")
        return

    # ── process each person sequentially ──────────────────
    total_cost = 0.0
    total_elapsed = 0.0
    found_count = 0
    total_coresignal = 0
    total_persons = len(persons)
    source_is_coresignal = args.source == "coresignal"
    cs_credits: int | None = None
    cs_credits_checked_at = 0  # profile index when credits were last checked

    for i, (school_id, person) in enumerate(persons, 1):
        logger.info(
            "[{}/{}] Processing: {} {}",
            i, total_persons, person.first_name, person.last_name,
        )
        cost, elapsed, found, cs_calls = await process_person(
            school_id,
            person,
            row_writer,
            args.outdir,
            source=args.source,
        )
        total_cost += cost
        total_elapsed += elapsed
        total_coresignal += cs_calls
        if found:
            found_count += 1

        avg_time = total_elapsed / i

        # Check Coresignal credits every 20 profiles (and on the first one).
        # In `local` mode, we do not query Coresignal.
        credits_str = "n/a"
        if source_is_coresignal:
            if cs_credits is None or (i - cs_credits_checked_at) >= 20:
                cs_credits = get_credits_remaining()
                cs_credits_checked_at = i

            profiles_ago = i - cs_credits_checked_at
            if cs_credits is not None:
                credits_str = (
                    f"{cs_credits} ({profiles_ago} profiles ago)" if profiles_ago > 0 else str(cs_credits)
                )
            else:
                credits_str = "?"

        logger.info(
            "\033[47;30m PROGRESS: {}/{} | Cost: ${:.4f} | Avg time: {:.1f}s | Found: {}/{} | CS calls: {} | CS credits: {} \033[0m",
            i, total_persons, total_cost, avg_time, found_count, i, total_coresignal, credits_str,
        )

    # ── totals summary ────────────────────────────────────
    logger.info("=================================================")
    logger.info(
        "TOTAL COST USD: ${:.6f} | TIME: {:.1f}s | PERSONS: {} | FOUND: {} | CS calls: {}",
        total_cost, total_elapsed, total_persons, found_count, total_coresignal,
    )


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Interrupted.")
    except Exception as e:
        logger.exception("Error: {}", e)
        raise
