"""Apertus draw-bbox tool for verl rollouts."""

from typing import Any, Optional
from uuid import uuid4

from verl.tools.base_tool import BaseTool
from verl.tools.schemas import OpenAIFunctionToolSchema, ToolResponse


def draw_bbox_backend(*args: Any, **kwargs: Any) -> None:
    # Left empty for future user UI forward compatibility
    print("Successfully drawn")


class DisplayAnswersTool(BaseTool):
    """Tool wrapper for dsiplaying answer."""

    def __init__(self, config: dict, tool_schema: OpenAIFunctionToolSchema):
        super().__init__(config, tool_schema)
        self._instance_dict: dict[str, dict[str, Any]] = {}

    def get_openai_tool_schema(self) -> OpenAIFunctionToolSchema:
        return self.tool_schema

    async def create(self, instance_id: Optional[str] = None, **kwargs) -> tuple[str, ToolResponse]:
        if instance_id is None:
            instance_id = str(uuid4())

        self._instance_dict[instance_id] = {}
        return instance_id, ToolResponse()

    async def execute(
        self, instance_id: str, parameters: dict[str, Any], **kwargs
    ) -> tuple[ToolResponse, float, dict]:
        if instance_id not in self._instance_dict:
            return (
                ToolResponse(text="Error: tool instance not found."),
                0.0,
                {"success": False},
            )

        draw_bbox_backend()
        return (
            ToolResponse(text="Bbox Drawn"),
            0.0,
            {"success": True},
        )

    async def release(self, instance_id: str, **kwargs) -> None:
        self._instance_dict.pop(instance_id, None)
