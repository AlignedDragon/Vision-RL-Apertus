"""Apertus display_answers tool for verl rollouts.

Functionally a no-op: returns an empty ToolResponse synchronously and signals
`is_terminal=True` in the metrics dict. The custom agent loop in
agent_loops/cof_tool_agent_loop.py reads this signal and stops the rollout
immediately, so the SFT-OOD case of the model receiving a tool message *after*
display_answers never occurs.
"""

from typing import Any, Optional
from uuid import uuid4

from verl.tools.base_tool import BaseTool
from verl.tools.schemas import OpenAIFunctionToolSchema, ToolResponse


class DisplayAnswersTool(BaseTool):
    """Terminal tool. Final-answer extraction is done downstream by the reward fn."""

    def __init__(self, config: dict, tool_schema: OpenAIFunctionToolSchema):
        super().__init__(config, tool_schema)

    def get_openai_tool_schema(self) -> OpenAIFunctionToolSchema:
        return self.tool_schema

    async def create(self, instance_id: Optional[str] = None, **kwargs) -> tuple[str, ToolResponse]:
        if instance_id is None:
            instance_id = str(uuid4())
        return instance_id, ToolResponse()

    async def execute(
        self, instance_id: str, parameters: dict[str, Any], **kwargs
    ) -> tuple[ToolResponse, float, dict]:
        answer = str(parameters.get("answer", ""))
        return (
            ToolResponse(text=""),
            0.0,
            {"success": True, "answer": answer, "is_terminal": True},
        )

    async def release(self, instance_id: str, **kwargs) -> None:
        return
