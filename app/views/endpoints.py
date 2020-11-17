from typing import Any
from fastapi import APIRouter, Request

from starlette.responses import HTMLResponse

from app.core.config import SERVER_HOST
from app.views.templates import templates

router = APIRouter()


@router.get("/home", response_class=HTMLResponse, name="get_home")
def get_home(
    request: Request, 
) -> Any:
    return templates.TemplateResponse(
        "home.html",
        {"request": request, "SERVER_HOST": SERVER_HOST},
    )