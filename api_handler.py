import os
from openai import AsyncOpenAI
from aiolimiter import AsyncLimiter

def get_openrouter_client():
    client = AsyncOpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=os.getenv("OPENROUTER_API_KEY"),
    )
    return client

class RateLimitedClient:
    def __init__(self, client, rate_limit):
        self.client = client
        self.limiter = AsyncLimiter(rate_limit, 1)  # rate_limit requests per second

    async def query_model(self, model, prompt):
        async with self.limiter:
            completion = await self.client.chat.completions.create(
                model=model,
                messages=[
                    {
                        "role": "user",
                        "content": prompt,
                    },
                ],
            )
            return completion.choices[0].message.content