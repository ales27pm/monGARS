"""Shared pytest configuration and fixtures."""

import importlib.machinery
import sys
import types

fake_spacy = types.ModuleType("spacy")
fake_spacy.load = lambda n: object()
fake_spacy.__spec__ = importlib.machinery.ModuleSpec("spacy", loader=None)
sys.modules.setdefault("spacy", fake_spacy)
