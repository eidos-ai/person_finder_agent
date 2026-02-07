#!/usr/bin/env python3
"""
Main entry point — loads persons from a Google Sheets spreadsheet,
processes each through the LinkedIn profile agent, and writes results
to a CSV file incrementally.
"""

import argparse
import asyncio
import csv
import json
import os
import sys
import time
from typing import Optional

from loguru import logger

from agent import PersonInitialDescription, PersonProfile, run_agent
from google_sheets import load_persons


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
    csv_path: str,
    outdir: str,
) -> tuple[float, float, bool, int]:
    """Process one person, write the result row to the CSV, return (cost, elapsed, found, coresignal_calls)."""
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
        result = await run_agent(person)
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

    # Write result to CSV immediately
    row = _build_csv_row(school_id, person, profile, confidence, error, total_cost, elapsed)
    _append_csv_row(csv_path, row)

    # Save per-person output file
    _save_person_output(school_id, person, profile, confidence, response_text, consumption_data, error, total_cost, elapsed, outdir)

    # Remove per-person log handler
    logger.remove(log_id)

    return total_cost, elapsed, found, coresignal_count


# ── main ───────────────────────────────────────────────────

async def main() -> None:
    parser = argparse.ArgumentParser(
        description="Process persons from a Google Sheet and write LinkedIn results to CSV.",
    )
    parser.add_argument("spreadsheet_id", help="Google Sheets spreadsheet ID")
    parser.add_argument("--start", type=int, default=None, help="Start row index (0-based)")
    parser.add_argument("--end", type=int, default=None, help="End row index (exclusive)")
    parser.add_argument("--output", default="results.csv", help="Output CSV path (default: results.csv)")
    parser.add_argument("--outdir", default="outs", help="Directory for per-person output files (default: outs)")
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
    found_count = 0
    total_coresignal = 0
    total_persons = len(persons)

    for i, (school_id, person) in enumerate(persons, 1):
        logger.info(
            "[{}/{}] Processing: {} {}",
            i, total_persons, person.first_name, person.last_name,
        )
        cost, elapsed, found, cs_calls = await process_person(school_id, person, args.output, args.outdir)
        total_cost += cost
        total_elapsed += elapsed
        total_coresignal += cs_calls
        if found:
            found_count += 1

        avg_time = total_elapsed / i
        logger.info(
            "\033[47;30m PROGRESS: {}/{} | Cost: ${:.4f} | Avg time: {:.1f}s | Found: {}/{} | CS calls: {} \033[0m",
            i, total_persons, total_cost, avg_time, found_count, i, total_coresignal,
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

