import gspread
from google.oauth2 import service_account

from googleapiclient.discovery import build
from googleapiclient.errors import HttpError

_SCOPES = [
    # if modifying these scopes, delete the file token.json.
    'https://www.googleapis.com/auth/drive',
    'https://www.googleapis.com/auth/drive.file',
    'https://www.googleapis.com/auth/drive.appdata',
    'https://www.googleapis.com/auth/drive.metadata',
    'https://www.googleapis.com/auth/spreadsheets',
]

creds = service_account.Credentials.from_service_account_file(
    './gcp-sa Manuel_Correa JS.json', scopes=_SCOPES
)

# Export credentials and project_id for use in other modules
credentials = creds
# Extract project_id from credentials
project_id = creds.project_id if hasattr(creds, 'project_id') else None
if not project_id:
    # Fallback: read from JSON file
    import json
    try:
        with open('./gcp-sa Manuel_Correa JS.json', 'r') as f:
            sa_info = json.load(f)
            project_id = sa_info.get('project_id')
    except Exception:
        project_id = None

try:
    _drive_service = build('drive', 'v3', credentials=creds)
    _gs = gspread.authorize(credentials=creds)
except HttpError as error:
    # TODO(developer) - Handle errors from drive API.
    print(f'APIG - Init : {error}')