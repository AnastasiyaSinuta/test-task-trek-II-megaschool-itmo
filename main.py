import time

from fastapi import FastAPI, HTTPException, Request, Response
from schemas.request import PredictionRequest, PredictionResponse
from utils.logger import setup_logger
import json
import requests

# Initialize
app = FastAPI()
logger = None


@app.on_event("startup")
async def startup_event():
    global logger
    logger = await setup_logger()


@app.middleware("http")
async def log_requests(request: Request, call_next):
    start_time = time.time()

    body = await request.body()
    await logger.info(
        f"Incoming request: {request.method} {request.url}\n"
        f"Request body: {body.decode()}"
    )

    response = await call_next(request)
    process_time = time.time() - start_time

    response_body = b""
    async for chunk in response.body_iterator:
        response_body += chunk

    await logger.info(
        f"Request completed: {request.method} {request.url}\n"
        f"Status: {response.status_code}\n"
        f"Response body: {response_body.decode()}\n"
        f"Duration: {process_time:.3f}s"
    )

    return Response(
        content=response_body,
        status_code=response.status_code,
        headers=dict(response.headers),
        media_type=response.media_type,
    )


@app.post("/api/request", response_model=PredictionResponse)
async def predict(body: PredictionRequest):
    try:
        await logger.info(f"Processing prediction request with id: {body.id}")
        prompt = "Предоставь ответ на вопрос в формате JSON {\"answer\": числовое значение (от 1 до 10), содержащее правильный вариант ответа на вопрос,\"reasoning\": объяснение или дополнительная информация по запросу в кавычках,\"sources\":  список ссылок на источники информации (не более трех), перечисленных через запятую в квадратных скобках}. Вопрос: " + body.query
        token = "***"
        url = "https://api.fireworks.ai/inference/v1/chat/completions"
        payload = {
            "model": "accounts/fireworks/models/llama-v3p1-405b-instruct",
            "max_tokens": 16384,
            "top_p": 1,
            "top_k": 40,
            "presence_penalty": 0,
            "frequency_penalty": 0,
            "temperature": 0.6,
            "messages": [
                {
                    "role": "user",
                    "content": f"{prompt}"
                }
            ]
        }
        headers = {
            "Accept": "application/json",
            "Content-Type": "application/json",
            "Authorization": f"Bearer {token}"
        }
        response_model = requests.request("POST", url, headers=headers, data=json.dumps(payload))
        response_content = str(response_model.json()['choices'][0]['message']['content'])
        json_string = response_content[response_content.find('{'):response_content.rfind('}') + 1]
        data = json.loads(json_string)
        answer = data.get("answer")
        reasoning = data.get("reasoning")
        sources = data.get("sources", [])

        response = PredictionResponse(
            id=body.id,
            answer=answer,
            reasoning=reasoning,
            sources=sources,
        )
        await logger.info(f"Successfully processed request {body.id}")
        return response
    except ValueError as e:
        error_msg = str(e)
        await logger.error(f"Validation error for request {body.id}: {error_msg}")
        raise HTTPException(status_code=400, detail=error_msg)
    except Exception as e:
        await logger.error(f"Internal error processing request {body.id}: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")