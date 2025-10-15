from monGARS.core.llm_integration import _sanitize_slot_generation_options


def test_sanitize_slot_generation_options_maps_sampling_values() -> None:
    options = {"temperature": 0.7, "top_p": 0.85, "num_predict": 256}

    result = _sanitize_slot_generation_options(options)

    assert result["temperature"] == 0.7
    assert result["top_p"] == 0.85
    assert result["max_new_tokens"] == 256
    assert result["do_sample"] is True


def test_sanitize_slot_generation_options_defaults_when_missing() -> None:
    result = _sanitize_slot_generation_options({})

    assert result["max_new_tokens"] == 512
    assert result["do_sample"] is False


def test_sanitize_slot_generation_options_respects_repeat_penalty() -> None:
    options = {"repeat_penalty": 1.1, "temperature": 0.0}

    result = _sanitize_slot_generation_options(options)

    assert result["repetition_penalty"] == 1.1
    assert result["do_sample"] is False
