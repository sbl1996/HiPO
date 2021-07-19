from typing import Dict, Any, Optional
from datetime import datetime
from pydantic import BaseModel

class Trial(BaseModel):
    id: int
    params: Dict[str, Any]
    budget: float
    value: float
    start: datetime
    end: Optional[datetime]