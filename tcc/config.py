from fastapi.templating import Jinja2Templates
from pathlib import Path

TEMPLATES_PATH = Path(__file__).resolve().parent.parent / "templates"
templates = Jinja2Templates(directory=str(TEMPLATES_PATH))
