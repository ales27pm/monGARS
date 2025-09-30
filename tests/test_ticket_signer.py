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
    signer = TicketSigner("secret", clock=clock.__call__)
    token = signer.sign(b"payload")

    clock.advance(5)
    assert signer.unsign(token, max_age=10) == b"payload"

    clock.advance(6)
    with pytest.raises(SignatureExpired):
        signer.unsign(token, max_age=10)


def test_detects_tampering() -> None:
    signer = TicketSigner("secret")
    token = signer.sign(b"payload")

    payload, timestamp, signature = token.split(".")
    altered_signature = signature[:-1] + ("A" if signature[-1] != "A" else "B")
    tampered = ".".join((payload, timestamp, altered_signature))

    with pytest.raises(BadSignature):
        signer.unsign(tampered, max_age=10)


def test_rejects_invalid_structure() -> None:
    signer = TicketSigner("secret")

    with pytest.raises(BadSignature):
        signer.unsign("missing-parts", max_age=5)
