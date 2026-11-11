"""Test setup: run the app with no Google Cloud access.

Gemini is replaced with a mock before app.main is imported, and data is
written to a temporary folder, so tests are fast, free and offline.
"""

import os
import tempfile
from unittest import mock

os.environ.setdefault("GOOGLE_CLOUD_PROJECT", "test-project")
os.environ["DATA_DIR"] = tempfile.mkdtemp(prefix="llmops-test-")
os.environ["ENABLE_TEST_ROUTES"] = "false"

import google.genai  # noqa: E402

google.genai.Client = mock.MagicMock()
