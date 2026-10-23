import pytest

from conftest import api
from constants import settings


def test_root_lists_all_modules(client):
    response = client.get(settings.ROOT_ROUTE)
    assert response.status_code == 200
    assert set(response.json()["modules"]) == {"stt", "cv", "llm", "tts"}


@pytest.mark.parametrize(
    ("module", "prefix"),
    [
        ("stt", settings.STT_ROUTE_PREFIX),
        ("cv", settings.CV_ROUTE_PREFIX),
        ("llm", settings.LLM_ROUTE_PREFIX),
        ("tts", settings.TTS_ROUTE_PREFIX),
    ],
)
def test_module_health_and_config(client, module, prefix):
    health = client.get(api(prefix, "/health"))
    assert health.status_code == 200
    assert health.json()["module"] == module

    config = client.get(api(prefix, "/config"))
    assert config.status_code == 200
    assert config.json()["route_prefix"] == prefix
