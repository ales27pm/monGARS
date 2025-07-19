import importlib
from unittest import mock

import monGARS.utils.hardware as hw


def test_detect_embedded_device_arm():
    with mock.patch("platform.machine", return_value="armv7l"):
        assert hw.detect_embedded_device() == "armv7l"


def test_detect_embedded_device_non_arm():
    with mock.patch("platform.machine", return_value="x86_64"):
        assert hw.detect_embedded_device() is None


def test_recommended_worker_count_defaults():
    with mock.patch("platform.machine", return_value="x86_64"):
        assert hw.recommended_worker_count(default=3) == 3


def test_recommended_worker_count_arm():
    with mock.patch("platform.machine", return_value="armv7l"), mock.patch(
        "psutil.cpu_count", return_value=4
    ):
        assert hw.recommended_worker_count() == 1


def test_recommended_worker_count_aarch64():
    with mock.patch("platform.machine", return_value="aarch64"), mock.patch(
        "psutil.cpu_count", return_value=8
    ):
        assert hw.recommended_worker_count() == 2
