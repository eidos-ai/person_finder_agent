"""
Google Sheets helper — loads persons from a spreadsheet.
"""

from typing import Optional

import pandas as pd

from agent import PersonInitialDescription
from google_api import _gs


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
