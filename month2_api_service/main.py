import logging
import os

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from dotenv import load_dotenv

from .config import allowed_origins
from .routes.api import router
from .services.security import new_request_id

load_dotenv()

logger = logging.getLogger("month2_api_service")
logging.basicConfig(level=logging.INFO)

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins(),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.include_router(router)


@app.middleware("http")
async def attach_request_id(request: Request, call_next):
    request_id = request.headers.get("X-Request-ID") or new_request_id()
    request.state.request_id = request_id
    response = await call_next(request)
    response.headers["X-Request-ID"] = request_id
    return response


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception):
    """Never send raw exception text to clients for unexpected errors."""
    from fastapi import HTTPException

    if isinstance(exc, HTTPException):
        # Should normally be handled by FastAPI; keep as safety net.
        return JSONResponse(
            status_code=exc.status_code,
            content={"detail": exc.detail},
            headers=getattr(exc, "headers", None),
        )

    request_id = getattr(request.state, "request_id", None) or new_request_id()
    logger.exception(
        "Unhandled error request_id=%s path=%s", request_id, request.url.path
    )
    return JSONResponse(
        status_code=500,
        content={
            "detail": (
                f"An unexpected error occurred. Please try again. "
                f"Request id: {request_id}"
            )
        },
        headers={"X-Request-ID": request_id},
    )


if __name__ == "__main__":
    import uvicorn

    port = int(os.getenv("PORT", "8000"))
    uvicorn.run(app, host="0.0.0.0", port=port)
