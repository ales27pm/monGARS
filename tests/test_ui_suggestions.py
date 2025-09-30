from __future__ import annotations

import pytest

from monGARS.core.aui import AUISuggester


@pytest.mark.asyncio
async def test_order_keyword_reflects_intent() -> None:
    suggester = AUISuggester()
    order, _ = await suggester.order("please refactor this python function")
    assert order[0] == "code"

    order2, _ = await suggester.order("tl;dr the following conversation")
    assert order2[0] == "summarize"

    order3, _ = await suggester.order("explain transformers like I'm five")
    assert order3[0] == "explain"
