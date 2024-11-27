# -*- coding: utf-8 -*-
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from config import settings
from api import router  # Importamos el router definido en api.py

# Crear la instancia de la aplicación
app = FastAPI(
    title=settings.PROJECT_NAME,
    openapi_url=f"{settings.API_V1_STR}/openapi.json",
    description="API para el proyecto final con endpoints para predicción y estado de salud.",
    version="0.1.0"
)

# Ruta raíz con HTML
@app.get("/", response_class=HTMLResponse)
def read_root():
    """
    Página principal con el enlace a la documentación.
    """
    body = (
        "<html>"
        "<body style='padding: 10px;'>"
        "<h1>Bienvenido a la API de Proyecto Final</h1>"
        "<div>"
        "Accede a la documentación en: <a href='/docs'>Swagger UI</a><br>"
        "Revisa el estado de salud en: <a href='/health'>/health</a>"
        "</div>"
        "</body>"
        "</html>"
    )
    return body

# Ruta de estado de salud
@app.get("/health")
def health_check():
    """
    Verifica si la API está corriendo correctamente.
    """
    return {"status": "ok"}

# Incluir el router de `api.py`
app.include_router(router, prefix=settings.API_V1_STR)


