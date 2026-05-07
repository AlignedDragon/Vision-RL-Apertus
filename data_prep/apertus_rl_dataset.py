"""Apertus-flavoured RLHFDataset and ToolAgentLoop.

The Apertus chat template iterates `tools` and reads `tool.description` directly
(no OpenAI `{"type":"function","function":{...}}` envelope). verl's tool registry
produces the wrapped form, so we unwrap once after super().__init__ at both
sites that pass tools to `apply_chat_template`:
  - dataset-side prompt-length filtering (RLHFDataset.doc2len)
  - rollout-side agent loop (ToolAgentLoop)

Tool dispatch is unaffected — `self.tools` keys (used to route emitted tool
calls to Python implementations) are not touched.
"""

from verl.experimental.agent_loop.agent_loop import register
from verl.experimental.agent_loop.tool_agent_loop import ToolAgentLoop
from verl.utils.dataset.rl_dataset import RLHFDataset


def _unwrap_openai_function(schema: dict) -> dict:
    if isinstance(schema, dict) and schema.get("type") == "function" and "function" in schema:
        return schema["function"]
    return schema


class ApertusRLHFDataset(RLHFDataset):
    def _read_files_and_tokenize(self):
        # RLHFDataset.__init__ runs prompt-length filtering inside
        # _read_files_and_tokenize, so we must unwrap before super() — otherwise
        # the filter applies the chat template with wrapped tool schemas and crashes.
        # Some verl versions call this before tool_schemas is set on self, so
        # access defensively via getattr.
        tool_schemas = getattr(self, "tool_schemas", None)
        if tool_schemas:
            self.tool_schemas = [_unwrap_openai_function(s) for s in tool_schemas]
        super()._read_files_and_tokenize()


@register("cof_tool_agent")
class ApertusToolAgentLoop(ToolAgentLoop):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self.tool_schemas:
            self.tool_schemas = [_unwrap_openai_function(s) for s in self.tool_schemas]
