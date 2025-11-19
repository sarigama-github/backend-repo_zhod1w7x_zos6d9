import os
from typing import List, Literal, Optional, Dict, Any
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import requests

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class ChatMessage(BaseModel):
    role: Literal["system", "user", "assistant"]
    content: str


class ChatRequest(BaseModel):
    provider: Literal["openai", "anthropic"] = Field(..., description="Model provider to use")
    model: str = Field(..., description="Model name to use")
    api_key: str = Field(..., description="API key for the selected provider")
    messages: List[ChatMessage] = Field(..., description="Chat history in OpenAI-style format")
    temperature: Optional[float] = Field(0.7, ge=0, le=2)
    max_tokens: Optional[int] = Field(None, ge=1)


class ChatResponse(BaseModel):
    provider: str
    model: str
    content: str
    usage: Optional[Dict[str, Any]] = None


@app.get("/")
def read_root():
    return {"message": "Hello from FastAPI Backend!"}


@app.get("/api/hello")
def hello():
    return {"message": "Hello from the backend API!"}


@app.get("/test")
def test_database():
    """Test endpoint to check if backend is running (DB fields retained for compatibility)."""
    response = {
        "backend": "✅ Running",
        "database": "❌ Not Available",
        "database_url": None,
        "database_name": None,
        "connection_status": "Not Connected",
        "collections": []
    }

    try:
        from database import db
        if db is not None:
            response["database"] = "✅ Available"
            response["database_url"] = "✅ Set" if os.getenv("DATABASE_URL") else "❌ Not Set"
            response["database_name"] = db.name if hasattr(db, 'name') else "✅ Connected"
            response["connection_status"] = "Connected"
            try:
                collections = db.list_collection_names()
                response["collections"] = collections[:10]
                response["database"] = "✅ Connected & Working"
            except Exception as e:
                response["database"] = f"⚠️  Connected but Error: {str(e)[:50]}"
        else:
            response["database"] = "⚠️  Available but not initialized"
    except ImportError:
        response["database"] = "❌ Database module not found (run enable-database first)"
    except Exception as e:
        response["database"] = f"❌ Error: {str(e)[:50]}"

    response["database_url"] = "✅ Set" if os.getenv("DATABASE_URL") else "❌ Not Set"
    response["database_name"] = "✅ Set" if os.getenv("DATABASE_NAME") else "❌ Not Set"

    return response


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    """Unified chat endpoint supporting multiple providers.

    Security note: For demo purposes, api_key is accepted per-request. In production, store API keys securely server-side.
    """
    try:
        if req.provider == "openai":
            return _chat_openai(req)
        elif req.provider == "anthropic":
            return _chat_anthropic(req)
        else:
            raise HTTPException(status_code=400, detail="Unsupported provider")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


def _chat_openai(req: ChatRequest) -> ChatResponse:
    url = "https://api.openai.com/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {req.api_key}",
        "Content-Type": "application/json",
    }
    payload: Dict[str, Any] = {
        "model": req.model,
        "messages": [m.model_dump() for m in req.messages],
        "temperature": req.temperature,
    }
    if req.max_tokens is not None:
        payload["max_tokens"] = req.max_tokens

    r = requests.post(url, headers=headers, json=payload, timeout=60)
    if r.status_code >= 400:
        try:
            detail = r.json()
        except Exception:
            detail = r.text
        raise HTTPException(status_code=r.status_code, detail=detail)
    data = r.json()
    content = data["choices"][0]["message"]["content"]
    return ChatResponse(provider="openai", model=req.model, content=content, usage=data.get("usage"))


def _chat_anthropic(req: ChatRequest) -> ChatResponse:
    url = "https://api.anthropic.com/v1/messages"
    headers = {
        "x-api-key": req.api_key,
        "anthropic-version": "2023-06-01",
        "content-type": "application/json",
    }

    # Convert OpenAI-style messages to Anthropic format
    system_parts: List[str] = [m.content for m in req.messages if m.role == "system"]
    system_str = "\n\n".join(system_parts) if system_parts else None
    convo: List[Dict[str, Any]] = []
    for m in req.messages:
        if m.role == "system":
            continue
        if m.role in ("user", "assistant"):
            convo.append({"role": m.role, "content": m.content})

    payload: Dict[str, Any] = {
        "model": req.model,
        "messages": convo,
        "temperature": req.temperature,
    }
    if system_str:
        payload["system"] = system_str
    if req.max_tokens is not None:
        payload["max_tokens"] = req.max_tokens
    else:
        payload["max_tokens"] = 1024

    r = requests.post(url, headers=headers, json=payload, timeout=60)
    if r.status_code >= 400:
        try:
            detail = r.json()
        except Exception:
            detail = r.text
        raise HTTPException(status_code=r.status_code, detail=detail)
    data = r.json()
    content = "".join(
        [block.get("text", "") for msg in data.get("content", []) for block in ([msg] if isinstance(msg, dict) else [])]
    )
    if not content:
        # Fallback for structured response
        try:
            first = data.get("content", [])[0]
            content = first.get("text", "") if isinstance(first, dict) else str(first)
        except Exception:
            content = ""

    return ChatResponse(provider="anthropic", model=req.model, content=content, usage=data.get("usage"))


if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
