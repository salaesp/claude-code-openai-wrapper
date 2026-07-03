"""Entry point: `python -m app.main` or `uvicorn app.server:app`."""
import uvicorn

if __name__ == "__main__":
    uvicorn.run("app.server:app", host="0.0.0.0", port=8000, reload=False)
