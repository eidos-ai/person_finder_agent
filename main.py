#!/usr/bin/env python3
"""
Main entry point — loads persons from a Google Sheets spreadsheet,
processes each through the LinkedIn profile agent, and writes results
to a CSV file incrementally.
"""

import argparse
import asyncio
import csv
import os
import sys
import time
from typing import Optional

import pandas as pd
from loguru import logger

from agent import PersonInitialDescription, PersonProfile, run_agent
from google_api import _gs


# ── CSV columns ────────────────────────────────────────────

INPUT_FIELDS = [
    "first_name", "last_name", "city", "state",
    "school_name", "school_department", "graduation_date",
]

PROFILE_FIELDS = [
    "has_job", "job_type", "job_related", "job_schedule",
    "self_employed", "employer_name", "employer_city", "employer_state",
    "continuing_education", "education_institution", "education_field",
    "sources",
]

CSV_COLUMNS = INPUT_FIELDS + PROFILE_FIELDS + ["error", "cost_usd", "elapsed_seconds"]


# ── spreadsheet loader ─────────────────────────────────────

def load_persons(
    spreadsheet_id: str,
    start: Optional[int] = None,
    end: Optional[int] = None,
) -> list[PersonInitialDescription]:
    """
    Load persons from Google Sheets.

    Args:
        spreadsheet_id: Google Sheets spreadsheet ID
        start: Optional start row index (0-based)
        end: Optional end row index (exclusive)

    Returns:
        List of PersonInitialDescription instances
    """
    sheet = _gs.open_by_key(spreadsheet_id).worksheet("DataImport")
    data = sheet.get_all_values()
    if not data:
        return []

    headers = data[0]
    rows = data[1:]

    # Apply row range if specified
    if start is not None:
        rows = rows[start:]
    if end is not None:
        rows = rows[:end]

    df = pd.DataFrame(rows, columns=headers)

    persons = []
    for _, row in df.iterrows():
        persons.append(PersonInitialDescription(
            first_name=row["FirstName"],
            last_name=row["LastName"],
            city=row["City"],
            state=row["State"],
            school_name=row["SchoolName"],
            school_department=row["Department"],
            graduation_date=row["Graduation"],
        ))
    return persons


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


def _build_csv_row(
    person: PersonInitialDescription,
    profile: Optional[PersonProfile],
    error: str,
    cost_usd: float,
    elapsed: float,
) -> dict:
    """Build a CSV row dict from person input and agent result."""
    row: dict = {
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
            "has_job": profile.has_job,
            "job_type": profile.job_type,
            "job_related": profile.job_related,
            "job_schedule": profile.job_schedule,
            "self_employed": profile.self_employed,
            "employer_name": profile.employer_name,
            "employer_city": profile.employer_city,
            "employer_state": profile.employer_state,
            "continuing_education": profile.continuing_education,
            "education_institution": profile.education_institution,
            "education_field": profile.education_field,
            "sources": "; ".join(profile.sources),
        })
    else:
        for field in PROFILE_FIELDS:
            row[field] = ""

    row["error"] = error
    row["cost_usd"] = f"{cost_usd:.6f}"
    row["elapsed_seconds"] = f"{elapsed:.1f}"
    return row


# ── per-person processing ─────────────────────────────────

async def process_person(
    person: PersonInitialDescription,
    csv_path: str,
) -> tuple[float, float]:
    """Process one person, write the result row to the CSV, return (cost, elapsed)."""
    name = f"{person.first_name} {person.last_name}"
    start = time.time()
    total_cost = 0.0
    error = ""
    profile: Optional[PersonProfile] = None

    try:
        result = await run_agent(person)
        profile = result.profile

        logger.info("RESULT: {}\n{}", name, result.profile)
        logger.debug("Final Response: {}\n{}", name, result.response)

        c = result.consumption
        for model, usage in c.models.items():
            cost = usage.cost_usd
            total_cost += cost
            logger.info(
                "{} | in={} | out={} | thinking={} | cost=${:.6f}",
                model, usage.input_tokens, usage.output_tokens,
                usage.thinking_tokens, cost,
            )
        logger.info("Tool calls: {}", c.tool_calls_used)

    except Exception as e:
        logger.exception(e)
        logger.error("{} -> ERROR: {}", name, e)
        error = str(e)

    elapsed = time.time() - start
    logger.info("{} completed in {:.1f}s", name, elapsed)

    # Write result to CSV immediately
    row = _build_csv_row(person, profile, error, total_cost, elapsed)
    _append_csv_row(csv_path, row)

    return total_cost, elapsed


# ── main ───────────────────────────────────────────────────

async def main() -> None:
    parser = argparse.ArgumentParser(
        description="Process persons from a Google Sheet and write LinkedIn results to CSV.",
    )
    parser.add_argument("spreadsheet_id", help="Google Sheets spreadsheet ID")
    parser.add_argument("--start", type=int, default=None, help="Start row index (0-based)")
    parser.add_argument("--end", type=int, default=None, help="End row index (exclusive)")
    parser.add_argument("--output", default="results.csv", help="Output CSV path (default: results.csv)")
    parser.add_argument("--debug", action="store_true", help="Enable DEBUG-level logging")

    args = parser.parse_args()

    # ── configure loguru ──────────────────────────────────
    logger.remove()  # remove default handler
    log_level = "DEBUG" if args.debug else "INFO"
    logger.add(sys.stderr, level=log_level)

    # ── load persons from spreadsheet ─────────────────────
    logger.info("Loading persons from spreadsheet: {}", args.spreadsheet_id)
    persons = load_persons(args.spreadsheet_id, start=args.start, end=args.end)
    logger.info("Loaded {} persons (start={}, end={})", len(persons), args.start, args.end)

    if not persons:
        logger.warning("No persons to process.")
        return

    # ── ensure CSV header exists ──────────────────────────
    _ensure_csv_header(args.output)
    logger.info("Writing results to: {}", args.output)

    # ── process each person sequentially ──────────────────
    total_cost = 0.0
    total_elapsed = 0.0

    for i, person in enumerate(persons, 1):
        logger.info(
            "[{}/{}] Processing: {} {}",
            i, len(persons), person.first_name, person.last_name,
        )
        cost, elapsed = await process_person(person, args.output)
        total_cost += cost
        total_elapsed += elapsed

    # ── totals summary ────────────────────────────────────
    logger.info(
        "TOTAL COST USD: ${:.6f} | TIME: {:.1f}s | PERSONS: {}",
        total_cost, total_elapsed, len(persons),
    )


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Interrupted.")
    except Exception as e:
        logger.exception("Error: {}", e)
        raise

