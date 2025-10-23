from unittest import mock

import monGARS.utils.hardware as hw


def test_detect_embedded_device_arm():
    with mock.patch("platform.machine", return_value="armv7l"):
        assert hw.detect_embedded_device() == "armv7l"


def test_detect_embedded_device_aarch64():
    with mock.patch("platform.machine", return_value="aarch64"):
        assert hw.detect_embedded_device() == "aarch64"


def test_detect_embedded_device_arm64():
    with mock.patch("platform.machine", return_value="arm64"):
        assert hw.detect_embedded_device() == "arm64"


def test_detect_embedded_device_non_arm():
    with mock.patch("platform.machine", return_value="x86_64"):
        assert hw.detect_embedded_device() is None

    with mock.patch("platform.machine", return_value="amd64"):
        assert hw.detect_embedded_device() is None

    with mock.patch("platform.machine", return_value="i386"):
        assert hw.detect_embedded_device() is None

    with mock.patch("platform.machine", return_value="unknown_arch"):
        assert hw.detect_embedded_device() is None


def test_recommended_worker_count_defaults():
    with mock.patch("platform.machine", return_value="x86_64"):
        assert hw.recommended_worker_count(default=3) == 3


def test_recommended_worker_count_arm():
    with mock.patch("platform.machine", return_value="armv7l"), mock.patch(
        "psutil.cpu_count", return_value=4
    ):
        assert hw.recommended_worker_count() == 1


def test_recommended_worker_count_cpu_count_none():
    with mock.patch("platform.machine", return_value="armv7l"), mock.patch(
        "psutil.cpu_count", return_value=None
    ):
        assert hw.recommended_worker_count() == 1


def test_recommended_worker_count_cpu_count_zero():
    with mock.patch("platform.machine", return_value="armv7l"), mock.patch(
        "psutil.cpu_count", return_value=0
    ):
        assert hw.recommended_worker_count() == 1


def test_recommended_worker_count_aarch64():
    with mock.patch("platform.machine", return_value="aarch64"), mock.patch(
        "psutil.cpu_count", return_value=8
    ):
        assert hw.recommended_worker_count() == 2

    with mock.patch("platform.machine", return_value="aarch64"), mock.patch(
        "psutil.cpu_count", return_value=1
    ):
        assert hw.recommended_worker_count() == 1

    with mock.patch("platform.machine", return_value="aarch64"), mock.patch(
        "psutil.cpu_count", return_value=2
    ):
        assert hw.recommended_worker_count() == 2
