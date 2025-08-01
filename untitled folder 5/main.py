import uvicorn
from fastapi import FastAPI
from contextlib import asynccontextmanager
from code_monitor.api.routes import api_router
from code_monitor.db.database import init_db
from code_monitor.db.migrations import run_migrations
from code_monitor.config import settings

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: Run before the application starts
    await init_db()
    await run_migrations()  # Run migrations after database initialization
    yield
    # Shutdown: Run when the application is shutting down
    # Add shutdown logic here if needed
    pass

app = FastAPI(
    title="GitHub Code Monitor",
    description="Monitor GitHub code changes with AI-powered insights",
    version="1.0.0",
    lifespan=lifespan
)

# Mount the API router at /api prefix
app.include_router(api_router, prefix="/api")

if __name__ == "__main__":
    uvicorn.run("main:app", host=settings.HOST, port=settings.PORT, reload=True)