"""
FastAPI API server for the RAG question answering system.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import List

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from query import RAGQueryEngine


class QueryRequest(BaseModel):
    question: str = Field(..., min_length=1, description="User question")


class SourceResponse(BaseModel):
    source: str
    section: str
    score: float
    content: str | None = None


class QueryResponse(BaseModel):
    answer: str
    sources: List[SourceResponse]


@lru_cache(maxsize=1)
def get_engine() -> RAGQueryEngine:
    return RAGQueryEngine(
        show_sources=True,
        use_hybrid_search=True,
        use_rerank=True,
    )


app = FastAPI(title="规章制度智能问答 API", version="1.0.0")
BASE_DIR = Path(__file__).resolve().parent
FRONTEND_DIST_DIR = BASE_DIR / "frontend" / "dist"
FRONTEND_INDEX_FILE = FRONTEND_DIST_DIR / "index.html"

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/health")
def health_check() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/api/query", response_model=QueryResponse)
def query_api(payload: QueryRequest) -> QueryResponse:
    question = payload.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="question cannot be empty")

    try:
        engine = get_engine()
        answer = engine.query(question)
        sources = [SourceResponse(**source) for source in engine.get_sources()]
        return QueryResponse(answer=answer, sources=sources)
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


if FRONTEND_DIST_DIR.exists():
    app.mount(
        "/assets",
        StaticFiles(directory=str(FRONTEND_DIST_DIR / "assets")),
        name="frontend-assets",
    )

    favicon_file = FRONTEND_DIST_DIR / "favicon.svg"

    @app.get("/favicon.svg", include_in_schema=False)
    def frontend_favicon() -> FileResponse:
        if favicon_file.exists():
            return FileResponse(favicon_file)
        raise HTTPException(status_code=404, detail="favicon not found")

    @app.get("/{full_path:path}", include_in_schema=False)
    def frontend_app(full_path: str) -> FileResponse:
        if full_path.startswith("api/"):
            raise HTTPException(status_code=404, detail="API route not found")

        requested = FRONTEND_DIST_DIR / full_path
        if full_path and requested.exists() and requested.is_file():
            return FileResponse(requested)

        if FRONTEND_INDEX_FILE.exists():
            return FileResponse(FRONTEND_INDEX_FILE)

        raise HTTPException(status_code=404, detail="frontend build not found")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("api_server:app", host="0.0.0.0", port=8000, reload=False)
