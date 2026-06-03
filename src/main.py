"""
/src/main.py

Author: Jared Moore
Date: July, 2025

General FastAPI app to run the API and serve the frontend.
"""

import logging
import os

from fastapi import FastAPI, Request, Response
from fastapi.exceptions import HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from starlette.background import BackgroundTask

from api.api import app as api_app
from experiment.llm_utils import disable_litellm_logging

# Silence LiteLLM logging
disable_litellm_logging()

# By default, don't show the docs. This can be overridden by the imported app.
app = FastAPI(openapi_url=None, docs_url=None, redoc_url=None)

# Mount the API routes
app.include_router(api_app.router)

## API Call Logging

# Ensure the log directory exists
if not os.path.exists("logs"):
    os.makedirs("logs")

# Check if handlers are already set (to prevent duplicate handlers during reloads)
root_logger = logging.getLogger()
root_logger.setLevel(logging.INFO)
if root_logger.handlers:
    root_logger.handlers.clear()

# Create the file handler
file_handler = logging.FileHandler("logs/main.log")
file_handler.setLevel(logging.DEBUG)

# Create the stream handler for stdout
stream_handler = logging.StreamHandler()
stream_handler.setLevel(logging.INFO)

# Define the formatter
formatter = logging.Formatter(
    "%(asctime)s - %(name)s - %(levelname)s - %(funcName)s - %(message)s"
)
file_handler.setFormatter(formatter)
stream_handler.setFormatter(formatter)

# Configure the root logger with both handlers
root_logger.addHandler(file_handler)
root_logger.addHandler(stream_handler)

# Use the module-level logger
logger = logging.getLogger(__name__)


def log_info(req_body, res_body):
    """Log the request and response body."""
    logger.debug(f"Request Body: {req_body}")
    logger.debug(f"Response Body: {res_body}")


@app.middleware("http")
async def log_request_and_response(request: Request, call_next):
    """
    Logs all requests and responses.
    """
    req_body = await request.body()
    response = await call_next(request)

    res_body = b""
    async for chunk in response.body_iterator:
        res_body += chunk

    task = BackgroundTask(log_info, req_body, res_body)
    return Response(
        content=res_body,
        status_code=response.status_code,
        headers=dict(response.headers),
        media_type=response.media_type,
        background=task,
    )


@app.middleware("http")
async def no_cache_middleware(request: Request, call_next):
    """
    Don't allow any caching of pages.
    """
    response = await call_next(request)
    # Overwrite/add the no‐cache headers
    response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
    response.headers["Pragma"] = "no-cache"
    response.headers["Expires"] = "0"
    return response


# Serve static files (HTML, JS, CSS)
frontend_dir = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "frontend", "dist")
)
app.mount(
    "/",
    StaticFiles(directory=frontend_dir, html=True, check_dir=False),
    name="frontend",
)


@app.exception_handler(404)
async def spa_404_handler(request: Request, exc: HTTPException):
    # If this was really an API call, or a request for *.js/*.css/etc, don't override
    path = request.url.path
    if path.startswith("/api") or os.path.splitext(path)[1]:
        return JSONResponse(status_code=exc.status_code, content={"detail": exc.detail})

    # Otherwise, serve index.html
    index_path = os.path.join(frontend_dir, "index.html")
    if not os.path.exists(index_path):
        # this likely means your build output directory is wrong
        return JSONResponse(status_code=500, content={"detail": "index.html not found"})
    return FileResponse(index_path, media_type="text/html")


# CORS middleware for development

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # Vite server
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
