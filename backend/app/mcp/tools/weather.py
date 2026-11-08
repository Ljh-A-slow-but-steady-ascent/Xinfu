from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timezone
from typing import Any

import httpx
from pydantic import BaseModel, ConfigDict, Field

from backend.app.mcp.schemas import ToolContext
from backend.app.services.privacy import EgressGuard, EgressRequest


class WeatherInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    city: str = Field(min_length=2, max_length=100, description="城市名；同名城市可加省份或国家，如 Paris, France。不要填街道或患者信息。")
    country_code: str | None = Field(default=None, pattern=r"^[A-Z]{2}$")
    days: int = Field(default=1, ge=1, le=7)


CONDITIONS = {
    0: "晴", 1: "晴间多云", 2: "少云", 3: "多云", 45: "雾", 48: "雾凇",
    51: "小毛毛雨", 53: "中等毛毛雨", 55: "强毛毛雨", 56: "冻毛毛雨", 57: "强冻毛毛雨",
    61: "小雨", 63: "中雨", 65: "大雨", 66: "冻雨", 67: "强冻雨",
    71: "小雪", 73: "中雪", 75: "大雪", 77: "雪粒", 80: "小阵雨", 81: "阵雨", 82: "强阵雨",
    85: "阵雪", 86: "强阵雪", 95: "雷暴", 96: "雷暴伴冰雹", 99: "强雷暴伴冰雹",
}
logger = logging.getLogger(__name__)


async def _get(client: httpx.AsyncClient, url: str, params: dict[str, Any]) -> httpx.Response:
    for attempt in range(2):
        try:
            response = await client.get(url, params=params)
            if attempt == 0 and response.status_code in {429, 500, 502, 503, 504}:
                await asyncio.sleep(0.2)
                continue
            response.raise_for_status()
            return response
        except httpx.TransportError:
            if attempt:
                raise
            await asyncio.sleep(0.2)
    raise RuntimeError("unreachable")


class WeatherTool:
    name = "get_weather"
    description = "查询指定城市的当前天气及未来七日预报。城市未知时先询问用户；同名地点根据返回候选确认。"

    def __init__(self, guard: EgressGuard | None = None, *, transport: httpx.AsyncBaseTransport | None = None) -> None:
        self.guard = guard or EgressGuard()
        self.transport = transport

    def input_schema(self) -> dict[str, Any]:
        return WeatherInput.model_json_schema()

    def _checked(self, host: str, params: dict[str, Any], context: ToolContext) -> dict[str, Any]:
        return self.guard.enforce(EgressRequest(
            channel="http", destination=host, payload=params,
            case_id=context.case_id, tool_name=self.name,
        )).sanitized_payload

    async def invoke(self, arguments: dict[str, Any], context: ToolContext) -> dict[str, Any]:
        query = WeatherInput.model_validate(arguments)
        city = query.city.strip()
        if not city:
            return {"status": "needs_location", "question": "你想查询哪个城市的天气？"}
        params: dict[str, Any] = {"city": city, "count": 5, "language": "zh", "format": "json"}
        if query.country_code:
            params["countryCode"] = query.country_code
        params = self._checked("geocoding-api.open-meteo.com", params, context)
        params["name"] = params.pop("city")
        try:
            async with httpx.AsyncClient(
                timeout=httpx.Timeout(8, connect=3), transport=self.transport,
                headers={"User-Agent": "Xinfu/1.0 (weather lookup)"},
            ) as client:
                response = await _get(client, "https://geocoding-api.open-meteo.com/v1/search", params)
                geocoding = response.json()
                if not isinstance(geocoding, dict):
                    raise ValueError("invalid location response")
                locations = geocoding.get("results", [])
                if not isinstance(locations, list) or any(not isinstance(item, dict) for item in locations):
                    raise ValueError("invalid locations")
                if not locations:
                    return {"status": "not_found", "question": "没有找到这个城市，请补充城市名和所在省份或国家。"}
                # Exact names resolve prefix matches, but do not guess between namesakes.
                name = city.split(",", 1)[0].strip().casefold().removesuffix("市")
                exact = [item for item in locations if str(item.get("name", "")).casefold().removesuffix("市") == name]
                candidates = exact or locations
                cities = [item for item in candidates if item.get("feature_code") in {"PPLC", "PPLA", "PPLA2", "PPLA3"}]
                if cities:
                    candidates = cities
                if len(candidates) != 1:
                    return {"status": "ambiguous", "question": "你指的是哪个地区的城市？", "locations": [
                        {"city": item.get("name"), "region": item.get("admin1"), "country": item.get("country"), "country_code": item.get("country_code")}
                        for item in candidates[:5]
                    ]}
                place = candidates[0]
                params = self._checked("api.open-meteo.com", {
                    "latitude": place["latitude"], "longitude": place["longitude"],
                    "current": "temperature_2m,relative_humidity_2m,apparent_temperature,weather_code,wind_speed_10m",
                    "daily": "weather_code,temperature_2m_max,temperature_2m_min,precipitation_probability_max",
                    "forecast_days": query.days, "timezone": "auto",
                    "temperature_unit": "celsius", "wind_speed_unit": "kmh",
                }, context)
                response = await _get(client, "https://api.open-meteo.com/v1/forecast", params)
                forecast = response.json()
                if not isinstance(forecast, dict):
                    raise ValueError("invalid weather response")
                current = forecast.get("current", {})
                if not isinstance(current, dict) or not current.get("time") or type(current.get("temperature_2m")) not in (int, float):
                    raise ValueError("incomplete current weather")
                daily = forecast.get("daily", {})
                if not isinstance(daily, dict):
                    raise ValueError("invalid forecast")
                days = []
                for index, date in enumerate(daily.get("time", [])[:query.days]):
                    def value(key: str) -> Any:
                        values = daily.get(key, [])
                        return values[index] if index < len(values) else None
                    days.append({"date": date, "condition": CONDITIONS.get(value("weather_code"), "未知"),
                                 "temperature_max_c": value("temperature_2m_max"), "temperature_min_c": value("temperature_2m_min"),
                                 "precipitation_probability_percent": value("precipitation_probability_max")})
                return {
                    "status": "ok", "source": "Open-Meteo", "source_scope": "外部天气服务",
                    "retrieved_at": datetime.now(timezone.utc).isoformat(), "timezone": forecast.get("timezone"),
                    "location": {"city": place.get("name"), "region": place.get("admin1"), "country": place.get("country"), "country_code": place.get("country_code")},
                    "current": {"time": current["time"], "temperature_c": current["temperature_2m"],
                                "feels_like_c": current.get("apparent_temperature"), "humidity_percent": current.get("relative_humidity_2m"),
                                "condition": CONDITIONS.get(current.get("weather_code"), "未知"), "wind_speed_kmh": current.get("wind_speed_10m")},
                    "daily": days,
                    "results": [{"title": f"{place['name']}天气 · Open-Meteo", "url": str(response.url)}],
                }
        except (httpx.HTTPError, ValueError, KeyError, TypeError) as error:
            code = error.response.status_code if isinstance(error, httpx.HTTPStatusError) else type(error).__name__
            logger.warning("Weather source unavailable: %s", code)
            return {"status": "unavailable", "source": "Open-Meteo", "message": "暂时无法获取天气，请稍后重试。", "retryable": True}
