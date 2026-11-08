import httpx
import pytest

from backend.app.mcp.schemas import ToolContext
from backend.app.mcp.tools.weather import WeatherTool
from backend.app.services.privacy import EgressGuard, EgressRequest


PLACE = {"name": "北京", "admin1": "北京市", "country": "中国", "country_code": "CN",
         "latitude": 39.9, "longitude": 116.4, "feature_code": "PPLC"}
FORECAST = {"timezone": "Asia/Shanghai", "current": {"time": "2026-11-08T12:45", "temperature_2m": 10.5,
            "relative_humidity_2m": 50, "apparent_temperature": 8, "weather_code": 0, "wind_speed_10m": 7},
            "daily": {"time": ["2026-11-08"], "weather_code": [3], "temperature_2m_max": [12],
                      "temperature_2m_min": [3], "precipitation_probability_max": [20]}}


@pytest.fixture
def anyio_backend():
    return "asyncio"


@pytest.mark.anyio
async def test_city_weather_survives_privacy_checks_and_ignores_namesake_villages():
    calls = []
    def handler(request):
        calls.append(request)
        if request.url.host == "geocoding-api.open-meteo.com":
            assert request.url.params["name"] == "北京"
            assert request.url.params["countryCode"] == "CN"
            return httpx.Response(200, json={"results": [PLACE, {**PLACE, "feature_code": "PPL", "admin1": "四川"}]})
        assert request.url.params["latitude"] == "39.9"
        assert "case_id" not in request.url.params
        return httpx.Response(200, json=FORECAST)
    result = await WeatherTool(transport=httpx.MockTransport(handler)).invoke(
        {"city": "北京", "country_code": "CN"}, ToolContext(case_id="CASE_PRIVATE"))
    assert len(calls) == 2
    assert result["status"] == "ok"
    assert result["location"]["city"] == "北京"
    assert result["current"]["temperature_c"] == 10.5
    assert result["current"]["condition"] == "晴"
    assert result["current"]["time"] == "2026-11-08T12:45"
    assert result["daily"][0]["temperature_min_c"] == 3
    assert result["results"][0]["url"].startswith("https://api.open-meteo.com/")
    assert EgressGuard().check(EgressRequest(channel="llm", destination="agent_tool_result", payload=result)).allowed
    assert not EgressGuard().check(EgressRequest(channel="http", destination="example.com", payload={"name": "张三"})).allowed


@pytest.mark.anyio
async def test_ambiguous_cities_are_not_silently_chosen():
    def handler(request):
        assert request.url.host == "geocoding-api.open-meteo.com"
        return httpx.Response(200, json={"results": [PLACE, {**PLACE, "admin1": "其他地区", "latitude": 1}]})
    result = await WeatherTool(transport=httpx.MockTransport(handler)).invoke({"city": "北京"}, ToolContext())
    assert result["status"] == "ambiguous"
    assert len(result["locations"]) == 2


@pytest.mark.anyio
async def test_unknown_city_does_not_request_forecast():
    def handler(request):
        assert request.url.host == "geocoding-api.open-meteo.com"
        return httpx.Response(200, json={"results": []})
    result = await WeatherTool(transport=httpx.MockTransport(handler)).invoke({"city": "不存在的城市"}, ToolContext())
    assert result["status"] == "not_found"


@pytest.mark.anyio
@pytest.mark.parametrize("failure", ["timeout", "upstream_401", "missing_current", "invalid_json_shape"])
async def test_failed_weather_never_invents_values_or_expires_site_login(failure):
    def handler(request):
        if failure == "timeout":
            raise httpx.ReadTimeout("timeout", request=request)
        if failure == "upstream_401":
            return httpx.Response(401, json={})
        if request.url.host == "geocoding-api.open-meteo.com":
            return httpx.Response(200, json={"results": [PLACE]})
        return httpx.Response(200, json=[] if failure == "invalid_json_shape" else {"current": None})
    result = await WeatherTool(transport=httpx.MockTransport(handler)).invoke({"city": "北京"}, ToolContext())
    assert result["status"] == "unavailable"
    assert "current" not in result
    assert "登录" not in result["message"]


@pytest.mark.anyio
async def test_missing_city_does_not_make_external_request():
    def handler(request):
        raise AssertionError("must ask for a city first")
    result = await WeatherTool(transport=httpx.MockTransport(handler)).invoke({"city": "  "}, ToolContext())
    assert result["status"] == "needs_location"


@pytest.mark.anyio
async def test_transient_weather_failure_recovers_inside_one_tool_call():
    attempts = 0
    def handler(request):
        nonlocal attempts
        if request.url.host == "geocoding-api.open-meteo.com":
            return httpx.Response(200, json={"results": [PLACE]})
        attempts += 1
        return httpx.Response(503, json={}) if attempts == 1 else httpx.Response(200, json=FORECAST)
    result = await WeatherTool(transport=httpx.MockTransport(handler)).invoke({"city": "北京"}, ToolContext())
    assert result["status"] == "ok"
    assert attempts == 2
