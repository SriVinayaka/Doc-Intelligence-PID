







#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from engine.policy_search_engine import PolicySearchEngine as PolicySearchEngine
import asyncio
import asyncio
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel





app = FastAPI(title="Policy Search Agent UI")


from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# app = FastAPI()

# ✅ ADD THIS BLOCK HERE
origins = [
    "http://localhost:3000",   # React UI
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ✅ Your existing endpoints below
@app.get("/")
def root():
    return {"message": "Policy Search AI Agent API is Working"}

# Load engine once (fast + efficient)
engine = PolicySearchEngine()


# ============================================================
# Request schema
# ============================================================

class QueryRequest(BaseModel):
    query: str = ""
    field: str = ""
    value: str = ""
    top_k: int = 1


# ============================================================
# Async query endpoint
# ============================================================

@app.post("/query")
async def query_api(req: QueryRequest):
    loop = asyncio.get_event_loop()

    # Run blocking FAISS + embedding in thread pool
    result = await loop.run_in_executor(
        None,
        lambda: engine.hybrid_query(
            query=req.query,
            field=req.field,
            value=req.value,
            top_k=req.top_k
        )
    )

    return JSONResponse(result)


# ============================================================
# Simple Web GUI
# ============================================================

@app.get("/home", response_class=HTMLResponse)
async def home():
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Policy Search AI Agent</title>
        <style>
            body {
                font-family: Arial;
                margin: 40px;
                background: #f5f5f5;
            }
            input, button {
                padding: 10px;
                margin: 5px;
                width: 300px;
            }
            textarea {
                width: 100%;
                height: 300px;
                margin-top: 20px;
            }
        </style>
    </head>
    <body>

    <h2>🧠 Policy Search AI Agent</h2>

    <input id="query" placeholder="Natural language query"><br>
    <input id="field" placeholder="Field (optional)"><br>
    <input id="value" placeholder="Value (optional)"><br>
    <button onclick="sendQuery()">Search</button>

    <textarea id="result"></textarea>

    <script>
        async function sendQuery() {
            const query = document.getElementById("query").value;
            const field = document.getElementById("field").value;
            const value = document.getElementById("value").value;

            const res = await fetch("/query", {
                method: "POST",
                headers: {"Content-Type": "application/json"},
                body: JSON.stringify({
                    query: query,
                    field: field,
                    value: value,
                    top_k: 5
                })
            });

            const data = await res.json();
            document.getElementById("result").value =
                JSON.stringify(data, null, 2);
        }
    </script>

    </body>
    </html>
    """

