"""Custom verl agent loop for the CoF-RL pipeline.

Behaviour: identical to verl's stock ToolAgentLoop except that any tool whose
execute() metrics include `is_terminal=True` (i.e. display_answers) ends the
rollout immediately after that tool's response is appended -- no further model
generation. This keeps Apertus from ever seeing a tool message that follows
display_answers, which is OOD relative to the SFT corpus.

Registered under the name "cof_tool_agent". The dataset row's `agent_name`
column selects this loop at rollout time.
"""

from verl.experimental.agent_loop.agent_loop import register
from verl.experimental.agent_loop.tool_agent_loop import AgentState, ToolAgentLoop


@register("cof_tool_agent")
class CofToolAgentLoop(ToolAgentLoop):
    async def _call_tool(self, tool_call, tools_kwargs, agent_data):
        resp, reward, meta = await super()._call_tool(tool_call, tools_kwargs, agent_data)
        if isinstance(meta, dict) and meta.get("is_terminal"):
            agent_data._cof_terminal_seen = True
        return resp, reward, meta

    async def _handle_processing_tools_state(self, agent_data) -> AgentState:
        next_state = await super()._handle_processing_tools_state(agent_data)
        if getattr(agent_data, "_cof_terminal_seen", False):
            return AgentState.TERMINATED
        return next_state
