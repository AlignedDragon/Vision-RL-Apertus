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

import asyncio
import json
import logging

import regex

from verl.experimental.agent_loop.agent_loop import register
from verl.experimental.agent_loop.tool_agent_loop import AgentState, ToolAgentLoop
from verl.experimental.agent_loop.tool_parser import FunctionCall, ToolParser
from verl.utils.dataset.rl_dataset import RLHFDataset
from verl.utils.profiler import simple_timer
from verl.utils.ray_utils import get_event_loop

logger = logging.getLogger(__file__)


def _unwrap_openai_function(schema: dict) -> dict:
    if isinstance(schema, dict) and schema.get("type") == "function" and "function" in schema:
        return schema["function"]
    return schema


@ToolParser.register("apertus")
class ApertusToolParser(ToolParser):
    """Parse Apertus native tool calls: <|tools_prefix|>[{tool: args}]<|tools_suffix|>."""

    tool_call_regex = regex.compile(r"<\|tools_prefix\|>(.*?)<\|tools_suffix\|>", regex.DOTALL)

    async def extract_tool_calls(self, responses_ids: list[int]) -> tuple[str, list[FunctionCall]]:
        loop = get_event_loop()
        text = await loop.run_in_executor(
            None, lambda: self.tokenizer.decode(responses_ids, skip_special_tokens=False)
        )
        matches = self.tool_call_regex.findall(text)
        if not matches:
            return text, []

        function_calls = []
        for match in matches:
            try:
                calls = json.loads(match)
                if not isinstance(calls, list):
                    calls = [calls]
                for call in calls:
                    if not isinstance(call, dict):
                        continue
                    if "name" in call and "arguments" in call:
                        name = call["name"]
                        arguments = call["arguments"]
                    elif len(call) == 1:
                        name, arguments = next(iter(call.items()))
                    else:
                        continue
                    function_calls.append(
                        FunctionCall(name=str(name), arguments=json.dumps(arguments, ensure_ascii=False))
                    )
            except Exception as e:
                logger.warning(f"Failed to decode Apertus tool call: {e}")

        content = self.tool_call_regex.sub("", text)
        return content, function_calls


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

    async def _handle_generating_state(
        self, agent_data, sampling_params: dict, ignore_termination: bool = False
    ) -> AgentState:
        with simple_timer("generate_sequences", agent_data.metrics):
            output = await self.server_manager.generate(
                request_id=agent_data.request_id,
                prompt_ids=agent_data.prompt_ids,
                sampling_params=sampling_params,
                image_data=agent_data.image_data,
                video_data=agent_data.video_data,
            )

        if agent_data.metrics.get("num_preempted") is None:
            agent_data.metrics["num_preempted"] = output.num_preempted if output.num_preempted is not None else -1
        else:
            agent_data.metrics["num_preempted"] += output.num_preempted if output.num_preempted is not None else 0

        agent_data.assistant_turns += 1
        agent_data.response_ids = output.token_ids
        agent_data.prompt_ids += agent_data.response_ids
        agent_data.response_mask += [1] * len(agent_data.response_ids)
        if output.log_probs:
            agent_data.response_logprobs += output.log_probs
        if output.routed_experts is not None:
            agent_data.routed_experts = output.routed_experts

        if not ignore_termination and len(agent_data.response_mask) >= self.response_length:
            return AgentState.TERMINATED
        if self.max_assistant_turns and agent_data.assistant_turns >= self.max_assistant_turns:
            return AgentState.TERMINATED
        if self.max_user_turns and agent_data.user_turns >= self.max_user_turns:
            return AgentState.TERMINATED

        text, agent_data.tool_calls = await self.tool_parser.extract_tool_calls(agent_data.response_ids)
        if agent_data.tool_calls:
            return AgentState.PROCESSING_TOOLS

        # The model sometimes emits a bare <|tools_suffix|> without a parseable
        # tool call. Treat it as an empty tool observation so generation can
        # continue instead of ending the rollout at the suffix token.
        raw_text = text
        if "<|tools_suffix|>" not in raw_text:
            raw_text = await self.loop.run_in_executor(
                None, lambda: self.tokenizer.decode(agent_data.response_ids, skip_special_tokens=False)
            )
        if "<|tools_suffix|>" in raw_text:
            empty_tool_output_ids = await self.loop.run_in_executor(
                None, lambda: self.tokenizer.encode("[]", add_special_tokens=False)
            )
            if len(agent_data.response_mask) + len(empty_tool_output_ids) >= self.response_length:
                return AgentState.TERMINATED
            agent_data.prompt_ids += empty_tool_output_ids
            agent_data.response_mask += [0] * len(empty_tool_output_ids)
            if agent_data.response_logprobs:
                agent_data.response_logprobs += [0.0] * len(empty_tool_output_ids)
            agent_data.user_turns += 1
            return AgentState.GENERATING

        return AgentState.TERMINATED

    async def _handle_processing_tools_state(self, agent_data) -> AgentState:
        tasks = []
        tool_call_names = []
        for tool_call in agent_data.tool_calls[: self.max_parallel_calls]:
            tasks.append(self._call_tool(tool_call, agent_data.tools_kwargs, agent_data))
            tool_call_names.append(tool_call.name)

        with simple_timer("tool_calls", agent_data.metrics):
            responses = await asyncio.gather(*tasks)

        tool_texts = []
        for tool_response, tool_reward, _ in responses:
            tool_texts.append(tool_response.text or "")
            if tool_reward is not None:
                agent_data.tool_rewards.append(tool_reward)

        response_text = "[" + ",".join(tool_texts) + "]"
        response_ids = await self.loop.run_in_executor(
            None, lambda: self.tokenizer.encode(response_text, add_special_tokens=False)
        )

        if len(agent_data.response_mask) + len(response_ids) >= self.response_length:
            return AgentState.TERMINATED

        agent_data.prompt_ids += response_ids
        agent_data.response_mask += [0] * len(response_ids)
        if agent_data.response_logprobs:
            agent_data.response_logprobs += [0.0] * len(response_ids)
        agent_data.user_turns += 1

        return AgentState.GENERATING
