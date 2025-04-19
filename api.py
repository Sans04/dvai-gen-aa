from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from services.chatbot import ChatBot, FastAPICallbackHandler

import asyncio
from typing import AsyncGenerator

app = FastAPI()
chatbot = ChatBot()

class QuestionRequest(BaseModel):
    question: str

@app.post("/invoke")
async def nejm_stream(request: QuestionRequest):
    async def token_stream() -> AsyncGenerator[str, None]:
        loop = asyncio.get_running_loop()
        queue = asyncio.Queue()

        handler = FastAPICallbackHandler(queue, loop)
        chatbot.llm.callbacks = [handler]

        def run_sync():
            chatbot.answer_question(request.question)
            asyncio.run_coroutine_threadsafe(queue.put(None), loop)

        asyncio.get_running_loop().run_in_executor(None, run_sync)

        while True:
            token = await queue.get()
            if token is None:
                break
            yield token

    return StreamingResponse(token_stream(), media_type="text/plain")

