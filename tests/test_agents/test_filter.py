from __future__ import annotations

import pytest

from planora.agents.filter import StreamFilter
from planora.agents.registry import StreamFormat


@pytest.mark.parametrize(
    ("stream_format", "line", "expected"),
    [
        (StreamFormat.CLAUDE, "", False),
        (StreamFormat.CLAUDE, "   ", False),
        (
            StreamFormat.CLAUDE,
            '{"type":"content_block_delta","delta":{"type":"text_delta"}}',
            False,
        ),
        (
            StreamFormat.CLAUDE,
            '{"type": "content_block_stop", "content_block": {}}',
            False,
        ),
        (
            StreamFormat.CLAUDE,
            '{"type":"assistant","message":{"content":[{"type":"text","text":"ok"}]}}',
            True,
        ),
        (StreamFormat.CLAUDE, '{"message":"no type field"}', True),
        (StreamFormat.CODEX, '{"type":"response.output_item.delta"}', False),
        (StreamFormat.CODEX, '{"type":"item.completed"}', True),
        (StreamFormat.COPILOT, '{"toolName":"Read"}', True),
    ],
)
def test_should_keep_filters_only_known_noise(
    stream_format: StreamFormat,
    line: str,
    expected: bool,
) -> None:
    assert StreamFilter(stream_format).should_keep(line) is expected


@pytest.mark.asyncio
async def test_filter_stream_preserves_actionable_lines_in_order() -> None:
    async def raw_lines():
        for line in (
            '{"type":"content_block_delta"}',
            '{"type":"assistant"}',
            '{"type":"content_block_stop"}',
            '{"type":"result"}',
        ):
            yield line

    filtered = [
        line async for line in StreamFilter(StreamFormat.CLAUDE).filter_stream(raw_lines())
    ]

    assert filtered == ['{"type":"assistant"}', '{"type":"result"}']
