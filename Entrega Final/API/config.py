# -*- coding: utf-8 -*-
from pydantic_settings import BaseSettings
from pydantic import AnyHttpUrl
from typing import List

class Settings(BaseSettings):
    API_V1_STR: str = "/api/v1"
    BACKEND_CORS_ORIGINS: List[AnyHttpUrl] = [
        "http://localhost",
        "http://127.0.0.1",
    ]
    PROJECT_NAME: str = "API de Proyecto Final"

    class Config:
        case_sensitive = True


settings = Settings()



