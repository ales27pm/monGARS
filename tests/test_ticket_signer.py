"""Unit tests for the internal TicketSigner implementation."""

from __future__ import annotations

import time

import pytest

from monGARS.api.ticket_signer import BadSignature, SignatureExpired, TicketSigner


class FixedClock:
    def __init__(self, start: float) -> None:
        self._now = start

    def __call__(self) -> float:
        return self._now

    def advance(self, seconds: float) -> None:
        self._now += seconds


def test_sign_and_unsign_roundtrip() -> None:
    clock = FixedClock(start=time.time())
    signer = TicketSigner("secret", clock=clock.__call__)

    token = signer.sign(b"user-123")
    assert signer.unsign(token, max_age=60) == b"user-123"


def test_token_expiration() -> None:
    clock = FixedClock(start=1000)
    signer = TicketSigner("secret", clock=clock.__call__, clock_skew_tolerance=0)
    token = signer.sign(b"payload")

    clock.advance(5)
    assert signer.unsign(token, max_age=10) == b"payload"

    clock.advance(6)
    with pytest.raises(SignatureExpired):
        signer.unsign(token, max_age=10)


def test_sign_and_unsign_non_ascii_payload() -> None:
    clock = FixedClock(start=time.time())
    signer = TicketSigner("secret", clock=clock.__call__)

    payload = "üñîçødë-测试-🚀".encode("utf-8")
    token = signer.sign(payload)

    assert signer.unsign(token, max_age=60) == payload


def test_detects_signature_tampering() -> None:
    signer = TicketSigner("secret")
    token = signer.sign(b"payload")

    payload, timestamp, signature = token.split(".")
    altered_signature = signature[:-1] + ("A" if signature[-1] != "A" else "B")
    tampered = ".".join((payload, timestamp, altered_signature))

    with pytest.raises(BadSignature):
        signer.unsign(tampered, max_age=10)


def test_detects_payload_tampering() -> None:
    signer = TicketSigner("secret")
    token = signer.sign(b"payload")

    payload, timestamp, signature = token.split(".")
    altered_payload = payload[:-1] + ("A" if payload[-1] != "A" else "B")
    tampered = ".".join((altered_payload, timestamp, signature))

    with pytest.raises(BadSignature):
        signer.unsign(tampered, max_age=10)


def test_detects_timestamp_tampering() -> None:
    signer = TicketSigner("secret")
    token = signer.sign(b"payload")

    payload, timestamp, signature = token.split(".")
    altered_timestamp = timestamp[:-1] + ("A" if timestamp[-1] != "A" else "B")
    tampered = ".".join((payload, altered_timestamp, signature))

    with pytest.raises(BadSignature):
        signer.unsign(tampered, max_age=10)


def test_rejects_invalid_structure() -> None:
    signer = TicketSigner("secret")

    with pytest.raises(BadSignature):
        signer.unsign("missing-parts", max_age=5)


def test_rejects_invalid_base64_payload() -> None:
    signer = TicketSigner("secret")
    timestamp = str(int(signer._clock()))
    payload = "abc$"
    signature = signer._signature(payload, timestamp)
    token = ".".join((payload, timestamp, signature))

    with pytest.raises(BadSignature):
        signer.unsign(token, max_age=10)


def test_rejects_non_integer_timestamp() -> None:
    signer = TicketSigner("secret")
    timestamp = "not-a-timestamp"
    payload = signer._b64encode(b"payload")
    signature = signer._signature(payload, timestamp)
    token = ".".join((payload, timestamp, signature))

    with pytest.raises(BadSignature):
        signer.unsign(token, max_age=10)


def test_rejects_future_timestamp_beyond_skew() -> None:
    clock = FixedClock(start=1000)
    signer = TicketSigner(
        "secret",
        clock=clock.__call__,
        clock_skew_tolerance=0,
    )
    token = signer.sign(b"payload")

    clock.advance(1)
    future_timestamp = str(int(clock() + 10))
    payload, _, _ = token.split(".")
    signature = signer._signature(payload, future_timestamp)
    tampered = ".".join((payload, future_timestamp, signature))

    with pytest.raises(BadSignature):
        signer.unsign(tampered, max_age=10)
