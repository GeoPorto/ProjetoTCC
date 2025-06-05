from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from tcc.config import templates
from fastapi.staticfiles import StaticFiles

app = FastAPI()

app.mount("/static", StaticFiles(directory="static"), name="static")
@app.get("/", response_class=HTMLResponse)
async def exibir_formulario(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


