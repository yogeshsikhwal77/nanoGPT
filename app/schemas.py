from pydantic import BaseModel, Field

class ChatRequest(BaseModel):
    story: str = Field(..., description="Story context")
    question: str = Field(..., description="Question regarding the story")
    temperature: float = Field(0.2, ge=0.0, le=2.0)
    max_tokens: int = Field(50, ge=1, le=128)
    model_name: str = Field("fiko", description="Model to use: 'fiko' or 'fire'")

class ChatResponse(BaseModel):
    answer: str
    tokens_generated: int
    inference_time_ms: float

class HealthResponse(BaseModel):
    status: str
    device: str
    vram_used_mb: float