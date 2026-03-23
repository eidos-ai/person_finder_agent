"""
Google Sheets helper — loads persons from a spreadsheet.
"""

from typing import Optional

import pandas as pd
import gspread

from agent import PersonInitialDescription
from google_api import _gs

RESULTS_SHEET_TITLE = "Results"


def load_persons(
    spreadsheet_id: str,
    start: Optional[int] = None,
    end: Optional[int] = None,
) -> list[tuple[str, PersonInitialDescription]]:
    """
    Load persons from Google Sheets.

    Args:
        spreadsheet_id: Google Sheets spreadsheet ID
        start: Optional start row index (0-based)
        end: Optional end row index (exclusive)

    Returns:
        List of (school_id, PersonInitialDescription) tuples.
    """
    sheet = _gs.open_by_key(spreadsheet_id).worksheet("DataImport")
    data = sheet.get_all_values()
    if not data:
        return []

    headers = data[0]
    rows = data[1:]

    # Apply row range if specified (both use 0-based data-row indices)
    rows = rows[start:end]

    df = pd.DataFrame(rows, columns=headers)

    persons = []
    for _, row in df.iterrows():
        school_id = row["SchoolID - Internal Use Only"]
        person = PersonInitialDescription(
            first_name=row["FirstName"],
            last_name=row["LastName"],
            city=row["City"],
            state=row["State"],
            school_name=row["SchoolName"],
            school_department=row["Department"],
            graduation_date=row["Graduation"],
        )
        persons.append((school_id, person))
    return persons


def get_or_create_results_sheet(
    spreadsheet_id: str,
    fieldnames: list[str],
) -> gspread.Worksheet:
    """
    Return the Results worksheet, creating it with a header row if missing.
    """
    sh = _gs.open_by_key(spreadsheet_id)
    try:
        ws = sh.worksheet(RESULTS_SHEET_TITLE)
    except gspread.exceptions.WorksheetNotFound:
        ws = sh.add_worksheet(
            title=RESULTS_SHEET_TITLE,
            rows=10000,
            cols=max(len(fieldnames), 1),
        )
        ws.append_row(fieldnames)
        return ws

    values = ws.get_all_values()
    if not values:
        ws.append_row(fieldnames)
    return ws


def get_processed_ids_from_sheet(spreadsheet_id: str) -> set[str]:
    """
    Load already-processed school_id values from the Results sheet.
    Returns empty set if the sheet does not exist.
    """
    sh = _gs.open_by_key(spreadsheet_id)
    try:
        ws = sh.worksheet(RESULTS_SHEET_TITLE)
    except gspread.exceptions.WorksheetNotFound:
        return set()

    data = ws.get_all_values()
    if len(data) < 2:
        return set()

    headers = data[0]
    try:
        school_id_idx = headers.index("school_id")
    except ValueError:
        return set()

    existing_ids: set[str] = set()
    for row in data[1:]:
        if school_id_idx < len(row):
            sid = (row[school_id_idx] or "").strip()
            if sid:
                existing_ids.add(sid)
    return existing_ids


def append_sheet_row(
    spreadsheet_id: str,
    row: dict,
    fieldnames: list[str],
) -> None:
    """Append one result row to the Results sheet (values in fieldnames order)."""
    sh = _gs.open_by_key(spreadsheet_id)
    ws = sh.worksheet(RESULTS_SHEET_TITLE)
    values = [str(row.get(key, "") if row.get(key, "") is not None else "") for key in fieldnames]
    ws.append_row(values)
