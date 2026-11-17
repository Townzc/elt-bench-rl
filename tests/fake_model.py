"""An offline stand-in for a Tinker model, for token-level tests.

``byte_tokenizer`` is a real HuggingFace fast tokenizer built in memory (one
token per byte plus the ChatML special tokens), so the cookbook's real
``qwen3_instruct`` renderer runs on it unchanged: tool declarations, tool-call
rendering and tool-call parsing are the production code paths.

``ScriptedTokenCompleter`` plays the role of the sampling client: it returns
the tokens of the next scripted assistant turn of whichever rollout the prompt
belongs to.
"""

from __future__ import annotations

from collections.abc import Callable

import tinker
from tinker_cookbook.completers import StopCondition, TokenCompleter, TokensWithLogprobs
from tinker_cookbook.renderers import get_renderer
from tinker_cookbook.renderers.base import RenderContext, Renderer
from tokenizers import Tokenizer, decoders, models, pre_tokenizers
from transformers import PreTrainedTokenizerFast

from elt_rl.env import ELTEnvGroupBuilder
from elt_rl.runtime import RolloutContext
from elt_rl.scripted import Turn, assistant_message

RENDERER_NAME = "qwen3_instruct"
ASSISTANT_HEADER = "<|im_start|>assistant"


def byte_tokenizer() -> PreTrainedTokenizerFast:
    alphabet = sorted(pre_tokenizers.ByteLevel.alphabet())
    backend = Tokenizer(models.BPE(vocab={ch: i for i, ch in enumerate(alphabet)}, merges=[]))
    backend.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=False)
    backend.decoder = decoders.ByteLevel()
    backend.add_special_tokens(["<|endoftext|>", "<|im_start|>", "<|im_end|>"])
    return PreTrainedTokenizerFast(tokenizer_object=backend, eos_token="<|endoftext|>")


class ScriptedTokenCompleter(TokenCompleter):
    """Replays a scripted policy as sampled tokens.

    Args:
        builder: The group being rolled out (to find its live rollouts).
        script_for: Maps a rollout (by index in the group) to its scripted turns.
    """

    def __init__(
        self,
        builder: ELTEnvGroupBuilder,
        script_for: Callable[[int, RolloutContext], list[Turn]],
    ) -> None:
        self.builder = builder
        self.script_for = script_for
        self.tokenizer = byte_tokenizer()
        self.renderer: Renderer = get_renderer(RENDERER_NAME, self.tokenizer)
        self.calls = 0
        self.max_prompt_tokens = 0

    async def __call__(
        self,
        model_input: tinker.ModelInput,
        stop: StopCondition,
        *,
        max_tokens: int | None = None,
    ) -> TokensWithLogprobs:
        self.calls += 1
        self.max_prompt_tokens = max(self.max_prompt_tokens, model_input.length)
        prompt = self.tokenizer.decode(model_input.to_ints())
        assert stop == self.renderer.get_stop_sequences()
        matches = [
            (index, rollout.ctx)
            for index, rollout in enumerate(self.builder._rollouts)
            if rollout.ctx.workspace in prompt
        ]
        assert len(matches) == 1, "prompt does not identify exactly one rollout"
        index, ctx = matches[0]
        turns = self.script_for(index, ctx)
        turn_index = prompt.count(ASSISTANT_HEADER) - 1  # the last one opens this turn
        if turn_index < len(turns):
            message = assistant_message(turns[turn_index], turn_index)
        else:  # script exhausted: answer without a tool call, which ends the episode
            message = {"role": "assistant", "content": "Nothing more to do."}
        rendered = self.renderer.render_message(
            message, RenderContext(idx=turn_index, is_last=True, prev_message=None)
        )
        tokens = [t for chunk in rendered.output for t in chunk.tokens]
        return TokensWithLogprobs(tokens=tokens, maybe_logprobs=[-0.25] * len(tokens))
