# ruff: noqa: E501

import json
import logging
import os
import time
import uuid
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from itertools import chain
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlsplit

import httpx
from dotenv import dotenv_values
from pydantic import SecretStr, ValidationError

from .config import WorkerSettings
from .exceptions import ProviderConfigurationError, ProviderError, WorkerError
from .inference_queue import InferenceLease, InferenceQueue
from .models import JsonMode, ProviderName
from .providers import create_provider
from .providers.adapter import CanonicalProviderAdapter
from .providers.base import (
    ProviderFunctionCall,
    ProviderMessage,
    ProviderRequest,
    ProviderToolCallsEvent,
)
from .providers.ollama import OllamaProvider
from .request_limits import RequestLimits, load_request_limits
from .responses.adapter import adapt_response_request
from .responses.builder import build_response
from .responses.schemas import (
    ResponseCreateRequest,
    ResponseErrorDetail,
    ResponseFunctionCall,
    ResponseObject,
    ResponseOutputMessage,
    ResponseOutputText,
)
from .responses.state import ResponseStateStore
from .responses.streaming import ResponseStreamEvent, encode_sse, map_provider_events
from .routing.gateway import resolve_gateway_fallback, resolve_gateway_route
from .routing.leases import create_route_lease, escalate_route_lease, escalation_reason_for
from .routing.logging import log_escalation, log_routing_decision
from .routing.models import RoutingMode
from .routing.routellm_adapter import ROUTELLM_BACKENDS
from .system_metrics import read_system_metrics
from .tools.executor import ToolExecutor
from .usage_statistics import (
    get_routing_plan,
    record_escalation,
    record_model_call,
    record_route_lease,
    record_routing_plan,
    summarize_model_calls,
    summarize_routing,
    summarize_v2_statistics,
)
from .virtual_models import VIRTUAL_MODEL_REGISTRY, ModelTier
from .web_config import (
    TIER_API_KEY_ENV,
    initialize_container_settings,
    load_gateway_routing_settings,
    load_public_settings,
    load_web_worker_settings,
    public_gateway_settings,
    save_gateway_settings,
    save_provider_settings,
)
from .web_models import (
    GatewaySettingsInput,
    ProviderSettingsInput,
    TierModelDiscoveryInput,
    validate_model_name,
)

INFERENCE_QUEUE = InferenceQueue()

RESPONSE_STATE = ResponseStateStore()
REQUEST_LIMITS = RequestLimits()
HTTP_LOGGER = logging.getLogger("local_code_worker.http")
if not HTTP_LOGGER.handlers:
    _http_handler = logging.StreamHandler()
    _http_handler.setFormatter(logging.Formatter("%(message)s"))
    HTTP_LOGGER.addHandler(_http_handler)
HTTP_LOGGER.setLevel(logging.INFO)
LOCAL_HOSTS = {
    "localhost",
    "127.0.0.1",
    "::1",
    "host.docker.internal",
    "local-code-worker-web",
}

# --- Responses API tool debug logging ---
_responses_debug = os.environ.get("LCW_DEBUG_RESPONSES_TOOLS", "").lower() in ("1", "true", "yes")
_responses_debug_log = logging.getLogger("local_code_worker.responses_tools")


def _td(event: str, **kw: object) -> None:
    """Log tool pipeline debug info when LCW_DEBUG_RESPONSES_TOOLS=true."""
    if not _responses_debug:
        return
    safe: dict[str, object] = {}
    for k, v in kw.items():
        if k in ("api_key", "authorization", "cookie", "token"):
            safe[k] = "[REDACTED]"
        elif isinstance(v, str) and len(v) > 500:
            safe[k] = v[:500] + f"...[{len(v)} chars]"
        else:
            safe[k] = v
    _responses_debug_log.warning(
        "[TOOLS] %s %s", event, json.dumps(safe, default=str, ensure_ascii=False)
    )


def _summarize_tools(tools: list[object]) -> list[dict[str, object]]:
    """Redacted tool summary for debug logging."""
    out: list[dict[str, object]] = []
    for t in tools:
        d = t.model_dump(exclude_none=True) if hasattr(t, "model_dump") else {}
        s: dict[str, object] = {"type": d.get("type", "?")}
        if "name" in d:
            s["name"] = d["name"]
        fn = d.get("function")
        if isinstance(fn, dict):
            s["fn_name"] = fn.get("name", "?")
            params = fn.get("parameters", {})
            if isinstance(params, dict):
                s["fn_params"] = list(params.get("properties", {}).keys())
        out.append(s)
    return out


def _resolve_tier_stored_key(tier: ModelTier, env_path: Path) -> str | None:
    """Resolve the API key stored for one routing tier (env var or .env file)."""
    env_name = TIER_API_KEY_ENV[tier]
    raw = os.environ.get(env_name)
    if raw is None:
        raw = dotenv_values(env_path).get(env_name)
    if raw is None:
        return None
    value = str(raw).strip()
    return value or None


def discover_tier_models(
    provider: ProviderName,
    base_url: Any,
    api_key: str | None,
    env_path: Path,
    transport: httpx.BaseTransport | None = None,
) -> list[str]:
    """Discover models for one routing tier using only that tier's settings.

    The request goes only to the given provider/base_url. ``api_key`` is the key
    typed in the card; when it is absent no key is used (neither the key of
    another tier nor unrelated environment credentials).
    """
    settings = WorkerSettings(
        _env_file=env_path,
        llm_provider=provider,
        llm_base_url=base_url,
        llm_model="",
        llm_api_key=SecretStr(api_key) if api_key else None,
        llm_api_key_env="",
    )
    return create_provider(settings, transport).list_models()


def _discovery_error_message(error: Exception) -> str:
    """Map provider discovery failures to short, user-facing messages."""
    if isinstance(error, ProviderConfigurationError):
        return "Для этого провайдера автоматический поиск моделей не поддерживается."
    if isinstance(error, ProviderError):
        category = error.category
        if category == "connection":
            return "Не удалось подключиться к серверу моделей"
        if category == "timeout":
            return "Таймаут подключения: сервер моделей не ответил"
        if category == "transport_error":
            return "Ошибка сети: не удалось связаться с сервером моделей"
        if category in {"http_401", "http_403"}:
            return "Ошибка авторизации: проверьте API-ключ"
        if category == "http_404":
            return "Endpoint получения моделей не найден (HTTP 404)"
        if category.startswith("http_"):
            return f"Не удалось получить список моделей (HTTP {category[5:]})"
        if category in {"invalid_json", "models_unsupported"}:
            return "Модели не найдены или сервер вернул некорректный ответ"
        return str(error)
    return "Не удалось получить список моделей"


class RequestBodyTooLarge(ValueError):
    def __init__(self, *, max_bytes: int, received_bytes: int) -> None:
        super().__init__("Request body exceeds maximum allowed size")
        self.max_bytes = max_bytes
        self.received_bytes = received_bytes


INDEX_HTML = r"""<!doctype html>
<html lang="ru"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Local Code Worker</title><style>
:root{color-scheme:dark;--bg:#0b1020;--panel:#131b31;--line:#2b3859;--text:#e8edff;--muted:#93a4ca;--accent:#7c9cff;--ok:#55d6a3;--bad:#ff7d91}*{box-sizing:border-box}
body{margin:0;background:radial-gradient(circle at 10% 0,#1c2850 0,transparent 38%),var(--bg);font:15px/1.5 system-ui;color:var(--text)}
main{max-width:920px;margin:42px auto;padding:0 20px}.hero{display:flex;justify-content:space-between;gap:24px;align-items:end;margin-bottom:22px}h1{font-size:32px;margin:0}p{color:var(--muted)}
.card{background:color-mix(in srgb,var(--panel) 94%,transparent);border:1px solid var(--line);border-radius:18px;padding:22px;box-shadow:0 22px 70px #0005}[hidden]{display:none!important}.grid{display:grid;grid-template-columns:1fr 1fr;gap:16px}.wide{grid-column:1/-1}
label{display:block;color:var(--muted);font-size:13px;margin-bottom:6px}input,select,button,textarea{width:100%;border:1px solid var(--line);border-radius:10px;background:#0d1428;color:var(--text);padding:11px 12px;font:inherit}button{cursor:pointer;background:var(--accent);color:#081126;border:0;font-weight:750}button.secondary{background:#243252;color:var(--text)}button:disabled{opacity:.45;cursor:not-allowed}.actions{display:flex;gap:12px;margin-top:18px}.status{min-height:24px;margin:12px 0 0;color:var(--muted)}.ok{color:var(--ok)}.bad{color:var(--bad)}textarea{height:150px;resize:vertical;font-family:ui-monospace,monospace}.keyrow{display:flex;gap:9px}.keyrow input{flex:1}.keyrow button{width:auto}.pill{padding:5px 9px;border:1px solid var(--line);border-radius:99px;color:var(--muted);font-size:12px}@media(max-width:650px){.grid{grid-template-columns:1fr}.wide{grid-column:auto}.hero{display:block}.actions{flex-direction:column}}
.tiers{display:grid;grid-template-columns:repeat(3,1fr);gap:12px}.tier{padding:16px;border:1px solid var(--line);border-radius:14px;background:#0d1428}.tier h3{margin:0 0 2px}.tier p{font-size:12px;margin:0 0 12px}.tier label{margin-top:9px}.check{display:flex;align-items:center;gap:8px}.check input{width:auto}.route-head{display:grid;grid-template-columns:2fr 1fr 1fr;gap:12px;margin-bottom:16px}@media(max-width:850px){.tiers{grid-template-columns:1fr}.route-head{grid-template-columns:1fr}}
.table-wrap{overflow-x:auto;border:1px solid var(--line);border-radius:12px}.usage-table{width:100%;border-collapse:collapse;min-width:760px}.usage-table th,.usage-table td{padding:11px 12px;text-align:right;border-bottom:1px solid var(--line);white-space:nowrap}.usage-table th{color:var(--muted);font-size:12px;font-weight:650;background:#0d1428}.usage-table th:first-child,.usage-table td:first-child{text-align:left}.usage-table tbody tr:last-child td{border-bottom:0}.usage-table tbody tr:hover{background:#18213a}.model-cell{font-weight:650;color:#b9c8ff}.provider-cell{color:var(--muted);font-size:12px}
</style></head><body><main><div class="hero"><div><h1>Local Code Worker</h1><p>Провайдеры, ключи и локальные модели — в одном локальном интерфейсе.</p></div><span class="pill" id="health">проверка…</span></div>
<section class="card" id="routingSettings"><h2 style="margin-top:0">Маршрутизация моделей</h2><p>Запрос начинается с локальных уровней. STRONG используйте как облачную страховку, если локальные модели не справились.</p><div class="route-head"><div><label for="routeMode">Режим</label><select id="routeMode"><option value="router">Router — применять выбор</option><option value="route_llm">RouteLLM policy</option><option value="shadow">Shadow — только сравнивать</option><option value="canary">Canary — стабильная выборка</option><option value="observe_only">Observe only — совместимость</option><option value="legacy">Legacy — одна модель ниже</option></select></div><div class="check"><input id="routeLlm" type="checkbox"><label for="routeLlm">RouteLLM</label></div><div><label for="routeThreshold">Старый порог RouteLLM</label><input id="routeThreshold" type="number" min="0" max="1" step="0.05"></div><div><label for="localThreshold">LOCAL threshold</label><input id="localThreshold" type="number" min="0" max="1" step="0.05"></div><div><label for="strongThreshold">STRONG threshold</label><input id="strongThreshold" type="number" min="0" max="1" step="0.05"></div><div><label for="canaryPercent">Canary, %</label><input id="canaryPercent" type="number" min="0" max="100" step="1"></div><div><label for="maxEscalations">Максимум эскалаций</label><input id="maxEscalations" type="number" min="0" max="10" step="1"></div></div><div class="tiers" id="tierCards"></div><div class="actions"><button id="saveRouting">Сохранить маршрутизацию</button></div><div class="status" id="routingStatus"></div></section>
<section class="card"><div class="grid"><div><label for="provider">Провайдер</label><select id="provider"><option value="ollama">Ollama (локально)</option><option value="openai-compatible">OpenAI-compatible API</option></select></div><div><label for="baseUrl">Base URL</label><input id="baseUrl"></div><div><label for="model">Модель</label><select id="model"></select></div><div><label for="contextLength">Контекст (токены)</label><input id="contextLength" type="number" min="512" max="131072" step="512"></div><div id="pullBlock"><label for="pullModel">Имя модели для скачивания (необязательно)</label><input id="pullModel" placeholder="qwen2.5-coder:7b-instruct-q5_K_M"></div><div id="keyBlock"><label for="apiKey">API-ключ (пусто = сохранить текущий)</label><div class="keyrow"><input id="apiKey" type="password" autocomplete="new-password" placeholder="••••••••"><button class="secondary" id="clearKey" type="button">Удалить</button></div></div></div>
<div class="actions"><button id="save">Сохранить и проверить</button><button class="secondary" id="refresh">Обновить модели</button><button class="secondary" id="pull">Скачать модель</button></div><div class="status" id="status"></div><textarea id="progress" readonly placeholder="Прогресс загрузки Ollama…"></textarea></section><section class="card" style="margin-top:18px"><h2 style="margin:0">Память модели</h2><p>Выгрузка модели из VRAM при простое запросов.</p><div class="grid"><div><label for="unloadPolicy">Выгрузка модели при простое</label><select id="unloadPolicy"><option value="immediate">Сразу (по умолчанию)</option><option value="5">5 мин</option><option value="10">10 мин</option><option value="30">30 мин</option><option value="never">Никогда</option></select></div></div><div class="actions"><button id="saveUnload">Сохранить</button></div><div class="status" id="unloadStatus"></div></section><section class="card" style="margin-top:18px"><div class="hero" style="margin-bottom:12px"><div><h2 style="margin:0">Мониторинг системы и моделей</h2><p style="margin:4px 0 0">Обновляется каждые 15 секунд.</p></div><span class="pill" id="runtimeUpdated">проверка…</span></div><div class="grid" id="metrics"></div><div class="status" id="runtime">Проверка состояния Ollama…</div></section><section class="card" style="margin-top:18px"><h2 style="margin:0">Маршрутизация</h2><p>Распределение, эскалации, latency и экономия cloud tokens.</p><div class="grid" id="routerMetrics"></div></section><section class="card" style="margin-top:18px"><div class="hero" style="margin-bottom:12px"><div><h2 style="margin:0">Статистика обращений</h2><p style="margin:4px 0 0">Накопительные данные; API-вызовы и legacy proposal учитываются раздельно.</p></div><span class="pill" id="usageUpdated">проверка…</span></div><div class="table-wrap"><table class="usage-table"><thead><tr><th>Модель</th><th>Запросы</th><th>Входные</th><th>Выходные</th><th>Ток/с</th><th>API успешно</th><th>API с ошибкой</th><th>Proposal валиден</th><th>Proposal невалиден</th></tr></thead><tbody id="usageStats"></tbody></table></div></section></main>
<script>
const $=id=>document.getElementById(id), provider=$('provider'), baseUrl=$('baseUrl'), model=$('model'), contextLength=$('contextLength'), pullModel=$('pullModel'), apiKey=$('apiKey'), status=$('status'), progress=$('progress'), runtime=$('runtime'), runtimeUpdated=$('runtimeUpdated'), metrics=$('metrics'), routerMetrics=$('routerMetrics'), usageStats=$('usageStats');unloadPolicy=$('unloadPolicy');let clearKey=false;
const tierNames=['local','mid','strong'],tierLabels={local:'LOCAL',mid:'MID',strong:'STRONG'},tierHelp={local:'Первая локальная модель: быстрые и простые задачи.',mid:'Локальная модель для рассуждений и сложного исполнения.',strong:'Последний уровень; здесь можно указать облачную модель.'},clearedTierKeys=new Set();const tierFindLoading={local:false,mid:false,strong:false},tierModelLists={local:[],mid:[],strong:[]};
function routingMessage(text,ok=true){const node=$('routingStatus');node.textContent=text;node.className='status '+(ok?'ok':'bad')}
function tierCard(name){const card=document.createElement('div');card.className='tier';card.dataset.tier=name;card.innerHTML=`<h3>${tierLabels[name]}</h3><p>${tierHelp[name]}</p><div class="check"><input id="${name}Enabled" type="checkbox"><label for="${name}Enabled">Уровень включён</label></div><label for="${name}Provider">Провайдер</label><select id="${name}Provider"><option value="ollama">Ollama</option><option value="openai-compatible">OpenAI-compatible</option></select><label for="${name}BaseUrl">Base URL</label><input id="${name}BaseUrl"><label for="${name}Model">Модель</label><select id="${name}Model"></select><button class="secondary" type="button" data-find-models="${name}">Найти модели</button><small id="${name}FindStatus"></small><label for="${name}Context">Контекст</label><input id="${name}Context" type="number" min="512" max="131072" step="512"><label for="${name}Key">API-ключ (пусто = оставить)</label><div class="keyrow"><input id="${name}Key" type="password" autocomplete="new-password"><button class="secondary" type="button" data-clear-key="${name}">×</button></div><small id="${name}KeyState"></small>`;return card}
for(const name of tierNames)$('tierCards').appendChild(tierCard(name));
const legacySettings=provider.closest('section.card'),legacyModes=new Set(['legacy','observe_only','shadow','canary']);
function syncLegacySettings(){legacySettings.hidden=!legacyModes.has($('routeMode').value)}
syncLegacySettings();
$('routeMode').addEventListener('change',syncLegacySettings);
function setTierModel(name,value){const select=$(name+'Model');if(value&&![...select.options].some(option=>option.value===value)){const option=document.createElement('option');option.value=value;option.textContent=value;select.appendChild(option)}select.value=value||''}
function fillTier(name,data){$(name+'Enabled').checked=data.enabled;$(name+'Provider').value=data.provider;$(name+'BaseUrl').value=data.base_url||'';setTierModel(name,data.model);$(name+'Context').value=data.context_length||32768;$(name+'Key').value='';$(name+'KeyState').textContent=data.api_key_configured?'Ключ сохранён':'Ключ не задан'}
function tierPayload(name){const key=$(name+'Key').value,clear=clearedTierKeys.has(name),action=clear?'clear':key?'replace':'keep';return {enabled:$(name+'Enabled').checked,provider:$(name+'Provider').value,base_url:$(name+'BaseUrl').value,model:$(name+'Model').value,context_length:Number($(name+'Context').value),api_key_action:action,api_key:action==='replace'?key:null}}
async function loadRouting(){try{const data=await jsonFetch('/api/v2/settings');$('routeMode').value=data.mode;syncLegacySettings();$('routeLlm').checked=data.routellm_enabled;$('routeThreshold').value=data.routellm_threshold;$('localThreshold').value=data.local_threshold;$('strongThreshold').value=data.strong_threshold;$('canaryPercent').value=data.canary_percent;$('maxEscalations').value=data.max_escalations_per_lease;for(const name of tierNames)fillTier(name,data.tiers[name]);routingMessage('Маршрутизация загружена')}catch(e){routingMessage(e.message,false)}}
async function saveRouting(){try{const tiers=Object.fromEntries(tierNames.map(name=>[name,tierPayload(name)])),data=await jsonFetch('/api/v2/settings',{method:'PUT',body:JSON.stringify({mode:$('routeMode').value,tiers,routellm_enabled:$('routeLlm').checked,routellm_threshold:Number($('routeThreshold').value),local_threshold:Number($('localThreshold').value),strong_threshold:Number($('strongThreshold').value),canary_percent:Number($('canaryPercent').value),max_escalations_per_lease:Number($('maxEscalations').value)})});clearedTierKeys.clear();for(const name of tierNames)fillTier(name,data.tiers[name]);routingMessage('Маршрутизация сохранена',true)}catch(e){routingMessage(e.message,false)}}
async function findModels(name){if(tierFindLoading[name])return;const button=$(name+'FindModels'),statusNode=$(name+'FindStatus'),baseUrl=$(name+'BaseUrl').value.trim();if(!baseUrl){statusNode.textContent='Укажите Base URL';statusNode.className='bad';return}statusNode.textContent='';statusNode.className='';button.disabled=true;button.textContent='Поиск…';tierFindLoading[name]=true;try{const data=await jsonFetch('/api/v2/discover-models',{method:'POST',body:JSON.stringify({tier:name,provider:$(name+'Provider').value,base_url:baseUrl,api_key:$(name+'Key').value||null})});const models=[...new Set((Array.isArray(data.models)?data.models:[]).filter(model=>typeof model==='string'&&model.trim()).map(model=>model.trim()))];tierModelLists[name]=models;const select=$(name+'Model'),current=select.value;select.replaceChildren();for(const model of models){const option=document.createElement('option');option.value=model;option.textContent=model;select.appendChild(option)}setTierModel(name,current);statusNode.textContent=`Найдено моделей: ${models.length}`;statusNode.className='ok'}catch(e){statusNode.textContent=e.message;statusNode.className='bad'}finally{button.disabled=false;button.textContent='Найти модели';tierFindLoading[name]=false}}
function message(text,ok=true){status.textContent=text;status.className='status '+(ok?'ok':'bad')}
function sync(){const local=provider.value==='ollama';$('keyBlock').style.display=local?'none':'block';$('pullBlock').style.display=local?'block':'none';$('pull').disabled=!local;if(local&&(!baseUrl.value||baseUrl.value.includes('example')))baseUrl.value='http://localhost:11434'}
async function jsonFetch(url,options={}){const response=await fetch(url,{headers:{'Content-Type':'application/json'},...options});const data=await response.json();if(!response.ok)throw new Error(data.error||response.statusText);return data}
function bytes(value){return typeof value==='number'?(value/1024/1024/1024).toFixed(1)+' ГБ':'—'}
function meter(title,value,caption,percent=0){const card=document.createElement('div');card.style.cssText='padding:14px;border:1px solid var(--line);border-radius:12px;background:#0d1428';const heading=document.createElement('div');heading.style.color='var(--muted)';heading.textContent=title;const number=document.createElement('div');number.style.cssText='font-size:25px;font-weight:750;margin-top:3px;color:var(--accent)';number.textContent=value;const bar=document.createElement('div');bar.style.cssText='height:7px;background:#253252;border-radius:99px;margin:10px 0 7px;overflow:hidden';const fill=document.createElement('div');fill.style.cssText=`height:100%;width:${Math.max(0,Math.min(100,percent))}%;background:linear-gradient(90deg,var(--accent),var(--ok));border-radius:inherit;transition:width .35s ease`;bar.appendChild(fill);const text=document.createElement('div');text.style.color='var(--muted)';text.textContent=caption;card.append(heading,number,bar,text);return card}
function renderMetrics(data){metrics.replaceChildren();const gpu=data.gpus&&data.gpus[0];if(gpu){metrics.append(meter(gpu.name,`${gpu.usage_percent}% GPU`,`${gpu.temperature_celsius} °C · ${bytes(gpu.memory_used_bytes)} / ${bytes(gpu.memory_total_bytes)} VRAM`,gpu.usage_percent),meter('Питание и частота',`${gpu.power_watts} Вт`,`лимит ${gpu.power_limit_watts} Вт · ${gpu.clock_mhz} МГц`,gpu.power_limit_watts?gpu.power_watts/gpu.power_limit_watts*100:0))}else{metrics.append(meter('GPU','Недоступна','Docker не передал NVIDIA NVML в контейнер'))}metrics.append(meter('CPU системы',`${data.cpu.usage_percent}% CPU`,'агрегированная загрузка',data.cpu.usage_percent),meter('RAM системы',`${data.memory.usage_percent}% RAM`,`${bytes(data.memory.used_bytes)} / ${bytes(data.memory.total_bytes)}`,data.memory.usage_percent))}
function renderUsage(data){usageStats.replaceChildren();if(!data.models.length){const row=document.createElement('tr'),cell=document.createElement('td');cell.colSpan=9;cell.textContent='Пока нет завершённых обращений.';row.appendChild(cell);usageStats.appendChild(row);return}for(const item of data.models){const row=document.createElement('tr'),modelCell=document.createElement('td'),modelName=document.createElement('div'),providerName=document.createElement('div');modelName.className='model-cell';modelName.textContent=item.model;providerName.className='provider-cell';providerName.textContent=item.provider;modelCell.append(modelName,providerName);row.appendChild(modelCell);for(const value of [item.requests,item.prompt_tokens,item.completion_tokens,item.tokens_per_second,item.api_completed,item.api_failed,item.code_valid,item.code_invalid]){const cell=document.createElement('td');cell.textContent=Number(value||0).toLocaleString();row.appendChild(cell)}usageStats.appendChild(row)}$('usageUpdated').textContent='обновлено '+new Date().toLocaleTimeString()}
function renderRouting(data,inference){routerMetrics.replaceChildren();const m=data.metrics,routes=Object.entries(m.requests_by_route||{}).map(([k,v])=>`${k.toUpperCase()}: ${v}`).join(' · ')||'нет решений',rates=Object.entries(m.success_rate_by_tier||{}).map(([k,v])=>`${k.toUpperCase()}: ${Math.round(v*100)}%`).join(' · ')||'нет данных',latency=Object.entries(m.average_latency_ms_by_tier||{}).map(([k,v])=>`${k.toUpperCase()}: ${Math.round(v)} ms`).join(' · ')||'нет данных';routerMetrics.append(meter('Inference',inference.active?'Выполняется':'Ожидание',`в очереди: ${inference.waiting} · ${inference.model||'модель не активна'}`,inference.active?100:0),meter(`Режим: ${data.mode}`,`${m.router_decisions_total} решений`,routes),meter('Эскалации',String(m.escalations_total),Object.entries(m.escalations_by_reason||{}).map(([k,v])=>`${k}: ${v}`).join(' · ')||'нет'),meter('Success rate',rates,latency),meter('Cloud tokens',String(m.cloud_tokens||0),`оценочно сохранено: ${m.cloud_tokens_saved||0}`))}
function runtimeCard(item){const total=item.size||0,vram=item.size_vram||0,gpu=total?Math.round(vram/total*100):0,card=document.createElement('div');card.style.cssText='padding:12px;border:1px solid var(--line);border-radius:10px;margin-top:8px';const title=document.createElement('strong');title.textContent=item.name;const details=document.createElement('div');details.style.color='var(--muted)';details.textContent=`GPU: ${gpu}% · CPU: ${100-gpu}% · VRAM модели: ${bytes(vram)} из ${bytes(total)} · Контекст: ${item.context_length||'—'}`;card.append(title,details);return card}
async function runtimeStatus(){try{const [data,system,usage,routing,inference]=await Promise.all([jsonFetch('/api/runtime'),jsonFetch('/api/system'),jsonFetch('/api/statistics'),jsonFetch('/api/v2/router/status'),jsonFetch('/api/inference')]);renderMetrics(system);renderUsage(usage);renderRouting(routing,inference);runtime.replaceChildren();if(data.provider!=='ollama'){runtime.textContent='Статус размещения доступен только для Ollama.'}else if(!data.models.length){runtime.textContent='Сейчас ни одна модель не загружена в Ollama.'}else{for(const item of data.models)runtime.appendChild(runtimeCard(item));}runtimeUpdated.textContent='обновлено '+new Date().toLocaleTimeString()}catch(e){runtime.textContent='Не удалось получить статус: '+e.message;runtimeUpdated.textContent='ошибка'}}
async function load(){try{const data=await jsonFetch('/api/settings');provider.value=data.provider;baseUrl.value=data.base_url;contextLength.value=data.context_length;sync();await models(data.model||'');await runtimeStatus();try{const up=await jsonFetch('/api/unload-policy');unloadPolicy.value=up.policy}catch(_){}$('health').textContent=data.api_key_configured?'ключ сохранён':'локальная конфигурация'}catch(e){message(e.message,false)}}
async function save(){try{const action=clearKey?'clear':apiKey.value?'replace':'keep';const data=await jsonFetch('/api/settings',{method:'PUT',body:JSON.stringify({provider:provider.value,base_url:baseUrl.value,model:model.value,context_length:Number(contextLength.value),api_key_action:action,api_key:action==='replace'?apiKey.value:null})});apiKey.value='';clearKey=false;message('Настройки сохранены. Выгрузите модель через ollama stop, чтобы применить новый контекст.',true);$('health').textContent=data.api_key_configured?'ключ сохранён':'готово'}catch(e){message(e.message,false)}}
async function models(preferredModel=model.value){preferredModel=typeof preferredModel==='string'?preferredModel:'';try{const data=await jsonFetch('/api/models');const names=[...new Set((Array.isArray(data.models)?data.models:[]).filter(name=>typeof name==='string'&&name.trim()).map(name=>name.trim()))];if(preferredModel&&!names.includes(preferredModel))names.unshift(preferredModel);model.replaceChildren();for(const name of names){const option=document.createElement('option');option.value=name;option.textContent=name;model.appendChild(option)}if(preferredModel)model.value=preferredModel;message(`Найдено моделей: ${data.models.length}`)}catch(e){message(e.message,false)}}
async function pull(){progress.value='';message('Загрузка модели…');const requestedModel=pullModel.value.trim()||model.value;try{const response=await fetch('/api/ollama/pull',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({model:requestedModel})});if(!response.ok){const d=await response.json();throw new Error(d.error)}const reader=response.body.getReader(),decoder=new TextDecoder();let pending='';while(true){const {value,done}=await reader.read();pending+=decoder.decode(value||new Uint8Array(),{stream:!done});const lines=pending.split('\n');pending=lines.pop();for(const line of lines)if(line){const item=JSON.parse(line);progress.value+=JSON.stringify(item)+'\n';progress.scrollTop=progress.scrollHeight}if(done)break}pullModel.value='';await models(requestedModel);message('Модель установлена',true)}catch(e){message(e.message,false)}}
async function saveUnloadPolicy(){try{await jsonFetch('/api/unload-policy',{method:'PUT',body:JSON.stringify({policy:unloadPolicy.value})})}catch(e){console.error('Unload policy save failed:',e)}}async function loadUnloadPolicy(){try{const d=await jsonFetch('/api/unload-policy');unloadPolicy.value=d.policy}catch(e){console.error('Unload policy load failed:',e)}}provider.addEventListener('change',sync);$('save').onclick=save;$('saveUnload').onclick=async()=>{await saveUnloadPolicy();$('unloadStatus').textContent='Сохранено';$('unloadStatus').className='status ok'};$('refresh').onclick=()=>models();$('pull').onclick=pull;$('clearKey').onclick=()=>{clearKey=true;apiKey.value='';message('Ключ будет удалён после сохранения')};$('saveRouting').onclick=saveRouting;$('tierCards').onclick=event=>{const name=event.target.dataset.clearKey;if(name){clearedTierKeys.add(name);$(name+'Key').value='';$(name+'KeyState').textContent='Ключ будет удалён после сохранения'}const find=event.target.dataset.findModels;if(find){findModels(find)}};load();loadRouting();loadUnloadPolicy();window.setInterval(runtimeStatus,15000);
</script></body></html>"""


class WorkerWebHandler(BaseHTTPRequestHandler):
    env_path = Path(".env")
    server_version = "LocalCodeWorkerWeb/0.1"

    def log_message(self, format: str, *args: object) -> None:
        # Avoid logging request payloads, query strings, and credentials.
        return

    def _local_request(self) -> bool:
        host = self.headers.get("Host", "").split(":", 1)[0].strip("[]").lower()
        origin = self.headers.get("Origin")
        origin_host = urlsplit(origin).hostname if origin else None
        return host in LOCAL_HOSTS and (origin_host is None or origin_host in LOCAL_HOSTS)

    def _send_json(self, status: HTTPStatus, payload: dict[str, object]) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _send_request_too_large(self, error: RequestBodyTooLarge) -> None:
        self._send_json(
            HTTPStatus.REQUEST_ENTITY_TOO_LARGE,
            {
                "error": {
                    "message": str(error),
                    "type": "invalid_request_error",
                    "code": "request_too_large",
                    "details": {
                        "max_bytes": error.max_bytes,
                        "received_bytes": error.received_bytes,
                    },
                }
            },
        )

    def _read_json(self, *, max_bytes: int) -> dict[str, Any]:
        transfer_encoding = self.headers.get("Transfer-Encoding")
        if transfer_encoding and transfer_encoding.lower().strip() != "identity":
            raise ValueError("Chunked request bodies are not supported")
        raw_length = self.headers.get("Content-Length")
        if raw_length is None:
            raise ValueError("Content-Length header is required")
        try:
            length = int(raw_length)
        except ValueError as error:
            raise ValueError("Invalid Content-Length") from error
        if length < 0:
            raise ValueError("Invalid Content-Length")
        if length > max_bytes:
            raise RequestBodyTooLarge(max_bytes=max_bytes, received_bytes=length)
        body = self.rfile.read(length)
        if len(body) != length:
            raise ValueError("Request body ended before Content-Length bytes were received")
        payload = json.loads(body)
        if not isinstance(payload, dict):
            raise ValueError("JSON body must be an object")
        return payload

    def _settings(self) -> WorkerSettings:
        return load_web_worker_settings(self.env_path)

    def _settings_with_tier_secret(self, settings: WorkerSettings) -> WorkerSettings:
        if not settings.llm_api_key_env:
            return settings
        secret = dotenv_values(self.env_path).get(settings.llm_api_key_env)
        if not secret:
            return settings
        return settings.model_copy(update={"llm_api_key": SecretStr(str(secret))})

    def _openai_models(self) -> None:
        standard_models = [
            {
                "id": model.id,
                "object": "model",
                "created": 0,
                "owned_by": "local-code-worker",
            }
            for model in VIRTUAL_MODEL_REGISTRY.list_models()
        ]
        _LCW_BASE_INSTRUCTIONS = (
            "You are a helpful coding assistant powered by a local model. "
            "You share a workspace with the user. Your job is to collaborate "
            "with them until their goal is genuinely handled.\n\n"
            "# Communication\n"
            "- Use the `commentary` channel for progress updates while you work.\n"
            "- End your turn with a final message in the `final` channel.\n"
            "- Be concise and direct. Lead with the outcome.\n\n"
            "# Tools\n"
            "- Use shell_command for running commands.\n"
            "- Use apply_patch for file edits.\n"
            "- Prefer the narrowest command that accomplishes the task."
        )
        _LCW_MODEL_MESSAGES = {
            "instructions_template": _LCW_BASE_INSTRUCTIONS,
            "instructions_variables": None,
            "approvals": None,
            "auto_review": None,
            "permissions": None,
        }
        codex_models = [
            {
                "slug": model.id,
                "display_name": model.id.rsplit("/", 1)[-1].upper(),
                "description": model.description,
                "default_reasoning_level": "low",
                "supported_reasoning_levels": [
                    {"effort": "low", "description": "Local inference"},
                    {"effort": "medium", "description": "Local inference"},
                    {"effort": "high", "description": "Local inference"},
                ],
                "shell_type": "shell_command",
                "visibility": "list",
                "supported_in_api": True,
                "priority": 10,
                "additional_speed_tiers": [],
                "service_tiers": [],
                "availability_nux": None,
                "upgrade": None,
                "base_instructions": _LCW_BASE_INSTRUCTIONS,
                "model_messages": _LCW_MODEL_MESSAGES,
                "include_skills_usage_instructions": False,
                "default_reasoning_summary": "none",
                "support_verbosity": True,
                "default_verbosity": "low",
                "apply_patch_tool_type": "freeform",
                "web_search_tool_type": "text_and_image",
                "truncation_policy": {"mode": "tokens", "limit": 10_000},
                "supports_parallel_tool_calls": True,
                "supports_image_detail_original": False,
                "context_window": 16_000,
                "max_context_window": 16_000,
                "comp_hash": "0",
                "effective_context_window_percent": 95,
                "experimental_supported_tools": [],
                "input_modalities": ["text"],
                "supports_search_tool": True,
                "use_responses_lite": True,
                "tool_mode": "default",
                "multi_agent_version": "v1",
            }
            for model in VIRTUAL_MODEL_REGISTRY.list_models()
        ]
        self._send_json(
            HTTPStatus.OK,
            {
                "object": "list",
                "data": standard_models,
                "models": codex_models,
            },
        )

    def _chat_request(
        self,
    ) -> tuple[
        WorkerSettings,
        list[dict[str, str]],
        bool,
        str,
        int | None,
        dict[str, object] | None,
    ]:
        payload = self._read_json(max_bytes=REQUEST_LIMITS.max_ui_request_bytes)
        raw_messages = payload.get("messages")
        if not isinstance(raw_messages, list) or not raw_messages:
            raise ValueError("messages must be a non-empty array")
        messages: list[dict[str, str]] = []
        for raw_message in raw_messages:
            if not isinstance(raw_message, dict):
                raise ValueError("each message must be an object")
            role = raw_message.get("role")
            content = raw_message.get("content")
            if role not in {"system", "user", "assistant", "tool"}:
                raise ValueError("message role is unsupported")
            if not isinstance(content, str):
                raise ValueError("message content must be a string")
            messages.append({"role": role, "content": content})

        settings = self._settings()
        model = validate_model_name(str(payload.get("model") or "local-code-worker/auto"))
        virtual_model = VIRTUAL_MODEL_REGISTRY.resolve(model)
        updates: dict[str, object] = {"llm_stream": False}
        temperature = payload.get("temperature")
        if temperature is not None:
            if not isinstance(temperature, int | float) or isinstance(temperature, bool):
                raise ValueError("temperature must be a non-negative number")
            if temperature < 0:
                raise ValueError("temperature must be a non-negative number")
            updates["llm_temperature"] = float(temperature)
        max_tokens = payload.get("max_tokens")
        if max_tokens is not None:
            if not isinstance(max_tokens, int) or isinstance(max_tokens, bool) or max_tokens <= 0:
                raise ValueError("max_tokens must be a positive integer")
            updates["llm_max_output_characters"] = min(
                settings.llm_max_output_characters, max_tokens * 8
            )
            # Honor larger client budgets (e.g. matching extraction needs far
            # more than the 4096-token default) up to a safety ceiling.
            updates["llm_max_output_tokens"] = min(
                max(settings.llm_max_output_tokens, max_tokens), 32_768
            )
        response_format = payload.get("response_format")
        response_schema: dict[str, object] | None = None
        if response_format is not None:
            if not isinstance(response_format, dict):
                raise ValueError("response_format must be an object")
            response_type = response_format.get("type")
            if response_type == "json_object":
                updates["llm_json_mode"] = JsonMode.JSON_OBJECT
            elif response_type == "json_schema":
                json_schema = response_format.get("json_schema")
                if not isinstance(json_schema, dict):
                    raise ValueError("json_schema response_format requires a json_schema object")
                schema = json_schema.get("schema")
                if not isinstance(schema, dict):
                    raise ValueError("json_schema response_format requires a schema object")
                updates["llm_json_mode"] = JsonMode.JSON_SCHEMA
                response_schema = schema
            elif response_type not in {None, "text"}:
                raise ValueError(
                    "only text, json_object and json_schema response formats are supported"
                )
        # Client-provided context override: wins over the routed tier's default
        # (the routing decision is applied afterwards, so _chat_completion
        # re-applies this value once the tier is known).
        context_length = payload.get("context_length")
        client_context_length: int | None = None
        if context_length is not None:
            if (
                not isinstance(context_length, int)
                or isinstance(context_length, bool)
                or context_length <= 0
            ):
                raise ValueError("context_length must be a positive integer")
            updates["llm_num_ctx"] = context_length
            client_context_length = context_length
        # Per-request "don't think" for reasoning models; absent = model default.
        think = payload.get("think")
        if think is not None:
            if not isinstance(think, bool):
                raise ValueError("think must be a boolean")
            updates["llm_think"] = think
        stream = payload.get("stream", False)
        if not isinstance(stream, bool):
            raise ValueError("stream must be a boolean")
        return (
            settings.model_copy(update=updates),
            messages,
            stream,
            virtual_model.id,
            client_context_length,
            response_schema,
        )

    def _chat_completion(self, lease: InferenceLease) -> None:
        (
            settings,
            messages,
            stream,
            public_model,
            client_context_length,
            response_schema,
        ) = self._chat_request()
        provider_request = ProviderRequest(
            messages=[ProviderMessage.model_validate(message) for message in messages],
            response_schema=response_schema,
            max_output_characters=settings.llm_max_output_characters,
            max_output_tokens=settings.llm_max_output_tokens,
            json_mode=settings.llm_json_mode,
            stream=False,
        )
        routing_settings = load_gateway_routing_settings(self.env_path)
        routellm_backend = (
            ROUTELLM_BACKENDS.get(routing_settings.routellm_checkpoint_path)
            if routing_settings.routellm_enabled
            else None
        )
        settings, routing_plan = resolve_gateway_route(
            provider_request,
            public_model,
            settings,
            routing_settings,
            routellm_backend,
        )
        if client_context_length is not None:
            # The routed tier sets its own num_ctx; a client-provided
            # context_length wins (matching passes its tuned budgets).
            settings = settings.model_copy(update={"llm_num_ctx": client_context_length})
        response_id = f"chat_{uuid.uuid4().hex}"
        if routing_settings.mode is not RoutingMode.LEGACY:
            route_lease = create_route_lease(response_id, routing_plan.actual)
            record_route_lease(route_lease)
            routing_plan = routing_plan.model_copy(
                update={
                    "actual": routing_plan.actual.model_copy(
                        update={"lease_id": route_lease.lease_id}
                    )
                }
            )
        settings = self._settings_with_tier_secret(settings)
        provider = create_provider(settings)
        lease.model = settings.llm_model
        lease.route = routing_plan.actual.tier.value
        if isinstance(provider, OllamaProvider):
            lease.idle_cleanup = provider.unload_model
        try:
            content = provider.chat(
                messages,
                response_schema,
                settings.llm_max_output_characters,
                settings.llm_max_output_tokens,
            )
        except (ProviderError, WorkerError, OSError):
            record_model_call(
                provider.last_generation_metadata,
                kind="chat",
                outcome="failed",
                tier=routing_plan.actual.tier.value,
            )
            fallback = resolve_gateway_fallback(
                settings,
                routing_settings,
                provider_request,
                routing_plan.actual.tier,
            )
            if fallback is None:
                record_routing_plan(response_id, routing_plan)
                raise
            settings, routing_plan = fallback
            if client_context_length is not None:
                settings = settings.model_copy(update={"llm_num_ctx": client_context_length})
            settings = self._settings_with_tier_secret(settings)
            provider = create_provider(settings)
            lease.model = settings.llm_model
            lease.route = routing_plan.actual.tier.value
            if isinstance(provider, OllamaProvider):
                lease.idle_cleanup = provider.unload_model
            content = provider.chat(
                messages,
                response_schema,
                settings.llm_max_output_characters,
                settings.llm_max_output_tokens,
            )
        metadata = provider.last_generation_metadata
        record_model_call(metadata, kind="chat", outcome="completed")
        completion_id = f"chatcmpl-{uuid.uuid4().hex}"
        created = int(time.time())
        finish_reason = metadata.finish_reason if metadata and metadata.finish_reason else "stop"
        usage = metadata.usage if metadata else {}
        normalized_usage = {
            "prompt_tokens": int(usage.get("prompt_tokens", 0)),
            "completion_tokens": int(usage.get("completion_tokens", 0)),
            "total_tokens": int(usage.get("total_tokens", 0)),
        }
        if stream:
            chunks = [
                {
                    "id": completion_id,
                    "object": "chat.completion.chunk",
                    "created": created,
                    "model": public_model,
                    "choices": [
                        {
                            "index": 0,
                            "delta": {"role": "assistant", "content": content},
                            "finish_reason": None,
                        }
                    ],
                },
                {
                    "id": completion_id,
                    "object": "chat.completion.chunk",
                    "created": created,
                    "model": public_model,
                    "choices": [{"index": 0, "delta": {}, "finish_reason": finish_reason}],
                },
            ]
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "text/event-stream; charset=utf-8")
            self.send_header("Cache-Control", "no-store")
            self.send_header("Connection", "close")
            self.end_headers()
            for chunk in chunks:
                self.wfile.write(f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n".encode())
            self.wfile.write(b"data: [DONE]\n\n")
            self.wfile.flush()
            return
        self._send_json(
            HTTPStatus.OK,
            {
                "id": completion_id,
                "object": "chat.completion",
                "created": created,
                "model": public_model,
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": content},
                        "finish_reason": finish_reason,
                    }
                ],
                "usage": normalized_usage,
            },
        )

    def _response(self, lease: InferenceLease, request_id: str) -> None:
        request = ResponseCreateRequest.model_validate(
            self._read_json(max_bytes=REQUEST_LIMITS.max_responses_request_bytes)
        )
        virtual_model = VIRTUAL_MODEL_REGISTRY.resolve(request.model)
        settings = self._settings().model_copy(
            update={"llm_stream": request.stream, "llm_json_mode": JsonMode.NONE}
        )
        adapted = adapt_response_request(
            request,
            max_output_characters=settings.llm_max_output_characters,
            json_mode=JsonMode.NONE,
        )
        provider_request = adapted.request
        hosted_tool_names = adapted.hosted_tool_names
        tool_executor = ToolExecutor() if hosted_tool_names else None

        # Debug: log Codex request details
        _td(
            "codex_request",
            request_id=request_id,
            model=request.model,
            stream=request.stream,
            tools_count=len(request.tools),
            tool_types=[t.type for t in request.tools],
            tool_names=[getattr(t, "name", None) for t in request.tools],
            tool_choice=str(request.tool_choice),
            parallel_tool_calls=request.parallel_tool_calls,
            previous_response_id=request.previous_response_id,
            has_instructions=request.instructions is not None,
            input_types=[
                getattr(i, "type", "?")
                for i in (request.input if isinstance(request.input, list) else [])
            ],
            input_count=len(request.input) if isinstance(request.input, list) else 1,
            hosted_tool_names=list(hosted_tool_names),
            passthrough_tool_count=len([t for t in adapted.all_normalized if t.is_function]),
            provider_tools_count=len(provider_request.tools),
            provider_tool_names=[t.name for t in provider_request.tools],
        )

        route_lease = None
        if request.previous_response_id is not None:
            previous = RESPONSE_STATE.get_stored(request.previous_response_id)
            previous_messages = list(previous.messages)
            route_lease = previous.route_lease
            provider_request = provider_request.model_copy(
                update={"messages": previous_messages + provider_request.messages}
            )
        response_id = f"resp_{uuid.uuid4().hex}"
        routing_settings = load_gateway_routing_settings(self.env_path)
        routellm_backend = (
            ROUTELLM_BACKENDS.get(routing_settings.routellm_checkpoint_path)
            if routing_settings.routellm_enabled
            else None
        )
        settings, routing_plan = resolve_gateway_route(
            provider_request,
            virtual_model.id,
            settings,
            routing_settings,
            routellm_backend,
            route_lease,
            response_id,
        )
        if routing_settings.mode is not RoutingMode.LEGACY:
            if route_lease is None:
                route_lease = create_route_lease(response_id, routing_plan.actual)
                record_route_lease(route_lease)
            routing_plan = routing_plan.model_copy(
                update={
                    "actual": routing_plan.actual.model_copy(
                        update={"lease_id": route_lease.lease_id}
                    )
                }
            )
        settings = self._settings_with_tier_secret(settings)
        provider = create_provider(settings)
        lease.model = settings.llm_model
        lease.route = routing_plan.actual.tier.value
        if isinstance(provider, OllamaProvider):
            lease.idle_cleanup = provider.unload_model
        message_id = f"msg_{uuid.uuid4().hex}"
        created_at = int(time.time())
        if not request.stream:
            import json as _json

            from .providers.base import ProviderFunctionCall as _PFC

            accumulated_calls: list[_PFC] = []
            max_tool_rounds = 5
            for _tool_round in range(max_tool_rounds):
                while True:
                    try:
                        result = CanonicalProviderAdapter(provider).complete(provider_request)
                        _td(
                            "model_result",
                            round=_tool_round,
                            content_len=len(result.content),
                            function_calls=[
                                {"name": fc.name, "call_id": fc.call_id}
                                for fc in result.function_calls
                            ],
                            finish_reason=result.finish_reason,
                        )
                        break
                    except (ProviderError, WorkerError, OSError) as error:
                        record_model_call(
                            provider.last_generation_metadata,
                            kind="response",
                            outcome="failed",
                            tier=routing_plan.actual.tier.value,
                        )
                        fallback = resolve_gateway_fallback(
                            settings,
                            routing_settings,
                            provider_request,
                            routing_plan.actual.tier,
                        )
                        if fallback is None:
                            record_routing_plan(response_id, routing_plan)
                            raise
                        settings, routing_plan = fallback
                        if route_lease is not None:
                            route_lease, escalation = escalate_route_lease(
                                route_lease,
                                routing_plan.actual,
                                escalation_reason_for(error),
                                request_id=request_id,
                                response_id=response_id,
                                max_escalations=routing_settings.max_escalations_per_lease,
                            )
                            record_escalation(escalation)
                            record_route_lease(route_lease)
                            log_escalation(escalation)
                            routing_plan = routing_plan.model_copy(
                                update={
                                    "actual": routing_plan.actual.model_copy(
                                        update={"lease_id": route_lease.lease_id}
                                    )
                                }
                            )
                        settings = self._settings_with_tier_secret(settings)
                        provider = create_provider(settings)
                        lease.model = settings.llm_model
                        lease.route = routing_plan.actual.tier.value
                        lease.idle_cleanup = (
                            provider.unload_model if isinstance(provider, OllamaProvider) else None
                        )
                # Check for hosted tool calls
                if not tool_executor or not result.function_calls:
                    _td(
                        "tool_loop_break",
                        reason="no_executor_or_no_fc",
                        has_executor=tool_executor is not None,
                        fc_count=len(result.function_calls),
                    )
                    break
                hosted_calls = [fc for fc in result.function_calls if fc.name in hosted_tool_names]
                if not hosted_calls:
                    _td(
                        "tool_loop_break",
                        reason="no_hosted_calls",
                        all_fc_names=[fc.name for fc in result.function_calls],
                        hosted_names=list(hosted_tool_names),
                    )
                    break
                # Execute hosted tools and continue conversation
                accumulated_calls.extend(hosted_calls)
                for call in hosted_calls:
                    try:
                        args = _json.loads(call.arguments) if call.arguments else {}
                    except _json.JSONDecodeError:
                        args = {}
                    tool_output = tool_executor.execute(call.name, args)
                    provider_request = provider_request.model_copy(
                        update={
                            "messages": provider_request.messages
                            + [
                                ProviderMessage(
                                    role="assistant",
                                    content="",
                                    tool_calls=[
                                        {
                                            "id": call.call_id,
                                            "type": "function",
                                            "function": {
                                                "name": call.name,
                                                "arguments": args,
                                            },
                                        }
                                    ],
                                ),
                                ProviderMessage(
                                    role="tool",
                                    content=tool_output,
                                    tool_call_id=call.call_id,
                                ),
                            ]
                        }
                    )
                # Loop continues - model sees tool results and generates final answer
            # Merge accumulated hosted calls with any remaining passthrough calls
            passthrough_calls = [
                fc for fc in result.function_calls if fc.name not in hosted_tool_names
            ]
            all_calls = accumulated_calls + passthrough_calls
            if all_calls:
                result = result.model_copy(update={"function_calls": all_calls})
            record_routing_plan(response_id, routing_plan)
            log_routing_decision(
                request_id=request_id,
                response_id=response_id,
                previous_response_id=request.previous_response_id,
                decision=routing_plan.actual,
            )
            record_model_call(
                provider.last_generation_metadata,
                kind="response",
                outcome="completed",
                request_id=response_id,
                tier=routing_plan.actual.tier.value,
                escalation_count=route_lease.escalation_count if route_lease else 0,
                tool_count=len(request.tools),
            )
            _td(
                "response_complete",
                response_id=response_id,
                content_len=len(result.content),
                function_calls=[
                    {"name": fc.name, "call_id": fc.call_id} for fc in result.function_calls
                ],
            )
            self._send_json(
                HTTPStatus.OK,
                build_response(
                    result,
                    response_id=response_id,
                    message_id=message_id,
                    created_at=created_at,
                    model=virtual_model.id,
                ).model_dump(mode="json", exclude_none=True),
            )
            if request.store:
                # Build stored messages including tool calls for continuation.
                stored_messages = list(provider_request.messages)
                if result.function_calls:
                    # Store assistant message with tool_calls for each call
                    for fc in result.function_calls:
                        try:
                            fc_args = json.loads(fc.arguments) if fc.arguments else {}
                        except json.JSONDecodeError:
                            fc_args = {}
                        stored_messages.append(
                            ProviderMessage(
                                role="assistant",
                                content=result.content or "",
                                tool_calls=[
                                    {
                                        "id": fc.call_id,
                                        "type": "function",
                                        "function": {
                                            "name": fc.name,
                                            "arguments": fc_args,
                                        },
                                    }
                                ],
                            )
                        )
                elif result.content:
                    stored_messages.append(
                        ProviderMessage(role="assistant", content=result.content)
                    )
                RESPONSE_STATE.put(response_id, stored_messages, route_lease)
            return
        max_tool_rounds = 5
        accumulated_stream_calls: list[ProviderFunctionCall] = []
        completed_response = None
        last_sequence = -1
        prior_text = ""
        last_stream_events = None
        try:
            for _tool_round in range(max_tool_rounds):
                # Get provider stream with fallback retry
                while True:
                    provider_events = iter(provider.stream(provider_request))
                    try:
                        first_provider_event = next(provider_events)
                        break
                    except (ProviderError, WorkerError, OSError) as error:
                        fallback = resolve_gateway_fallback(
                            settings,
                            routing_settings,
                            provider_request,
                            routing_plan.actual.tier,
                        )
                        if fallback is None:
                            record_routing_plan(response_id, routing_plan)
                            raise
                        settings, routing_plan = fallback
                        if route_lease is not None:
                            route_lease, escalation = escalate_route_lease(
                                route_lease,
                                routing_plan.actual,
                                escalation_reason_for(error),
                                request_id=request_id,
                                response_id=response_id,
                                max_escalations=routing_settings.max_escalations_per_lease,
                            )
                            record_escalation(escalation)
                            record_route_lease(route_lease)
                            log_escalation(escalation)
                            routing_plan = routing_plan.model_copy(
                                update={
                                    "actual": routing_plan.actual.model_copy(
                                        update={"lease_id": route_lease.lease_id}
                                    )
                                }
                            )
                        settings = self._settings_with_tier_secret(settings)
                        provider = create_provider(settings)
                        lease.model = settings.llm_model
                        lease.route = routing_plan.actual.tier.value
                        lease.idle_cleanup = (
                            provider.unload_model if isinstance(provider, OllamaProvider) else None
                        )
                    except StopIteration as error:
                        raise ProviderError(
                            "provider stream ended before the first event",
                            category="empty_stream",
                        ) from error

                if _tool_round == 0:
                    record_routing_plan(response_id, routing_plan)
                    log_routing_decision(
                        request_id=request_id,
                        response_id=response_id,
                        previous_response_id=request.previous_response_id,
                        decision=routing_plan.actual,
                    )
                    self.send_response(HTTPStatus.OK)
                    self.send_header("Content-Type", "text/event-stream; charset=utf-8")
                    self.send_header("Cache-Control", "no-store")
                    self.send_header("Connection", "close")
                    self.end_headers()

                raw_events = chain([first_provider_event], provider_events)
                last_stream_events = provider_events

                # Intercept ProviderToolCallsEvent — stop forwarding and
                # hand control back to the tool loop.
                tool_calls_event = None

                def _intercept_events(source):
                    nonlocal tool_calls_event
                    for ev in source:
                        if isinstance(ev, ProviderToolCallsEvent):
                            tool_calls_event = ev
                            return
                        yield ev

                try:
                    for event in map_provider_events(
                        _intercept_events(raw_events),
                        provider=settings.llm_provider,
                        model=virtual_model.id,
                        response_id=response_id,
                        message_id=message_id,
                        created_at=created_at,
                        start_sequence=last_sequence + 1 if _tool_round > 0 else 0,
                        emit_preamble=(_tool_round == 0),
                        prior_text=prior_text,
                    ):
                        last_sequence = event.sequence_number
                        if event.type == "response.completed":
                            completed_response = event.response
                            # Don't emit completed yet — we may need to add
                            # function calls below.
                            continue
                        self.wfile.write(encode_sse(event))
                        self.wfile.flush()
                except (BrokenPipeError, ConnectionResetError):
                    return
                except (ProviderError, WorkerError, ValidationError, ValueError, OSError) as error:
                    failed = ResponseObject(
                        id=response_id,
                        created_at=created_at,
                        status="failed",
                        model=request.model,
                        output=[],
                        output_text="",
                        error=ResponseErrorDetail(
                            message=str(error),
                            type="server_error",
                        ),
                    )
                    self.wfile.write(
                        encode_sse(
                            ResponseStreamEvent(
                                type="response.failed",
                                sequence_number=last_sequence + 1,
                                response=failed,
                            )
                        )
                    )
                    self.wfile.flush()
                    return

                if tool_calls_event is None:
                    break  # No tool calls — streaming is complete

                # Separate hosted vs passthrough calls
                hosted_calls = [
                    fc for fc in tool_calls_event.function_calls if fc.name in hosted_tool_names
                ]
                passthrough_calls = [
                    fc for fc in tool_calls_event.function_calls if fc.name not in hosted_tool_names
                ]

                # Emit passthrough function calls as SSE events for Codex
                if passthrough_calls:
                    _td(
                        "stream_passthrough_calls",
                        calls=[
                            {"name": fc.name, "call_id": fc.call_id} for fc in passthrough_calls
                        ],
                    )
                    accumulated_stream_calls.extend(passthrough_calls)
                    # Emit each function call with full SSE lifecycle
                    for fc in passthrough_calls:
                        fc_item = ResponseFunctionCall(
                            id=fc.call_id,
                            status="in_progress",
                            call_id=fc.call_id,
                            name=fc.name,
                            arguments="",
                        )
                        # output_item.added with in_progress status
                        last_sequence += 1
                        self.wfile.write(
                            encode_sse(
                                ResponseStreamEvent(
                                    type="response.output_item.added",
                                    sequence_number=last_sequence,
                                    output_index=0,
                                    item=fc_item,
                                )
                            )
                        )
                        self.wfile.flush()
                        # function_call_arguments.delta with full args
                        last_sequence += 1
                        self.wfile.write(
                            encode_sse(
                                ResponseStreamEvent(
                                    type="response.function_call_arguments.delta",
                                    sequence_number=last_sequence,
                                    output_index=0,
                                    content_index=0,
                                    item_id=fc.call_id,
                                    delta=fc.arguments,
                                )
                            )
                        )
                        self.wfile.flush()
                        # function_call_arguments.done
                        last_sequence += 1
                        self.wfile.write(
                            encode_sse(
                                ResponseStreamEvent(
                                    type="response.function_call_arguments.done",
                                    sequence_number=last_sequence,
                                    output_index=0,
                                    content_index=0,
                                    item_id=fc.call_id,
                                    text=fc.arguments,
                                )
                            )
                        )
                        self.wfile.flush()
                        # output_item.done with completed status
                        fc_item_done = ResponseFunctionCall(
                            id=fc.call_id,
                            status="completed",
                            call_id=fc.call_id,
                            name=fc.name,
                            arguments=fc.arguments,
                        )
                        last_sequence += 1
                        self.wfile.write(
                            encode_sse(
                                ResponseStreamEvent(
                                    type="response.output_item.done",
                                    sequence_number=last_sequence,
                                    output_index=0,
                                    item=fc_item_done,
                                )
                            )
                        )
                        self.wfile.flush()
                    # Now emit response.completed with all output items
                    break

                if not hosted_calls:
                    break  # No hosted calls to execute
                accumulated_stream_calls.extend(hosted_calls)
                for call in hosted_calls:
                    try:
                        call_args = json.loads(call.arguments) if call.arguments else {}
                    except json.JSONDecodeError:
                        call_args = {}
                    tool_output = tool_executor.execute(call.name, call_args)
                    provider_request = provider_request.model_copy(
                        update={
                            "messages": provider_request.messages
                            + [
                                ProviderMessage(
                                    role="assistant",
                                    content="",
                                    tool_calls=[
                                        {
                                            "id": call.call_id,
                                            "type": "function",
                                            "function": {
                                                "name": call.name,
                                                "arguments": call_args,
                                            },
                                        }
                                    ],
                                ),
                                ProviderMessage(
                                    role="tool",
                                    content=tool_output,
                                    tool_call_id=call.call_id,
                                ),
                            ]
                        }
                    )
                # Accumulate text so far for the final response
                if completed_response is not None:
                    prior_text = completed_response.output_text
                completed_response = None
                # Loop continues — model sees tool results and generates answer
        finally:
            if last_stream_events is not None:
                close = getattr(last_stream_events, "close", None)
                if close is not None:
                    close()
        # Emit the final response.completed SSE event with all output items
        # (text message + any function calls).
        if completed_response is not None:
            # Build output items: text message + function calls
            output_items: list[ResponseOutputMessage | ResponseFunctionCall] = []
            if completed_response.output_text or not accumulated_stream_calls:
                output_items.append(
                    ResponseOutputMessage(
                        id=message_id,
                        status="completed",
                        content=[
                            ResponseOutputText(text=completed_response.output_text or prior_text)
                        ],
                    )
                )
            for fc in accumulated_stream_calls:
                output_items.append(
                    ResponseFunctionCall(
                        id=fc.call_id,
                        status="completed",
                        call_id=fc.call_id,
                        name=fc.name,
                        arguments=fc.arguments,
                    )
                )
            final_response = ResponseObject(
                id=response_id,
                created_at=created_at,
                status="completed",
                model=virtual_model.id,
                output=output_items,
                output_text=completed_response.output_text or prior_text,
                usage=completed_response.usage,
            )
            last_sequence += 1
            try:
                self.wfile.write(
                    encode_sse(
                        ResponseStreamEvent(
                            type="response.completed",
                            sequence_number=last_sequence,
                            response=final_response,
                        )
                    )
                )
                self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError):
                pass
            record_model_call(
                provider.last_generation_metadata,
                kind="response",
                outcome="completed",
                request_id=response_id,
                tier=routing_plan.actual.tier.value,
                escalation_count=route_lease.escalation_count if route_lease else 0,
                tool_count=len(request.tools),
            )
        if request.store and completed_response is not None:
            # Store messages including function calls for continuation
            stored_messages = list(provider_request.messages)
            if accumulated_stream_calls:
                for fc in accumulated_stream_calls:
                    try:
                        fc_args = json.loads(fc.arguments) if fc.arguments else {}
                    except json.JSONDecodeError:
                        fc_args = {}
                    stored_messages.append(
                        ProviderMessage(
                            role="assistant",
                            content=completed_response.output_text or "",
                            tool_calls=[
                                {
                                    "id": fc.call_id,
                                    "type": "function",
                                    "function": {"name": fc.name, "arguments": fc_args},
                                }
                            ],
                        )
                    )
            else:
                stored_messages.append(
                    ProviderMessage(
                        role="assistant",
                        content=completed_response.output_text,
                    )
                )
            RESPONSE_STATE.put(response_id, stored_messages, route_lease)

    def do_GET(self) -> None:
        request_url = urlsplit(self.path)
        if request_url.path == "/":
            body = INDEX_HTML.encode("utf-8")
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)
            return
        if not self._local_request():
            self._send_json(HTTPStatus.FORBIDDEN, {"error": "Local requests only"})
            return
        try:
            if request_url.path == "/api/settings":
                self._send_json(HTTPStatus.OK, load_public_settings(self.env_path))
                return
            if request_url.path == "/api/v2/settings":
                self._send_json(HTTPStatus.OK, public_gateway_settings(self.env_path))
                return
            if request_url.path == "/api/models":
                models = create_provider(self._settings()).list_models()
                self._send_json(HTTPStatus.OK, {"models": models})
                return
            if request_url.path == "/api/runtime":
                settings = self._settings()
                provider = create_provider(settings)
                models = provider.running_models() if isinstance(provider, OllamaProvider) else []
                self._send_json(
                    HTTPStatus.OK,
                    {"provider": settings.llm_provider.value, "models": models},
                )
                return
            if request_url.path == "/api/system":
                self._send_json(HTTPStatus.OK, read_system_metrics())
                return
            if request_url.path == "/api/inference":
                self._send_json(HTTPStatus.OK, INFERENCE_QUEUE.status())
                return
            if request_url.path == "/api/unload-policy":
                from dotenv import dotenv_values as _dv

                env_vals = _dv(self.env_path)
                policy = str(env_vals.get("LLM_UNLOAD_POLICY") or INFERENCE_QUEUE._unload_policy)
                options = ["immediate", "5", "10", "30", "never"]
                self._send_json(HTTPStatus.OK, {"policy": policy, "options": options})
                return
            if request_url.path == "/api/statistics":
                self._send_json(HTTPStatus.OK, summarize_model_calls())
                return
            if request_url.path == "/api/v2/statistics":
                query = parse_qs(request_url.query)
                raw_baseline = query.get("baseline_cloud_tokens", [None])[-1]
                baseline = int(raw_baseline) if raw_baseline is not None else None
                if baseline is not None and baseline < 0:
                    raise ValueError("baseline_cloud_tokens must be non-negative")
                self._send_json(HTTPStatus.OK, summarize_v2_statistics(baseline))
                return
            if request_url.path == "/api/v2/router/metrics":
                self._send_json(HTTPStatus.OK, summarize_routing())
                return
            if request_url.path == "/api/v2/router/status":
                routing = public_gateway_settings(self.env_path)
                routing["metrics"] = summarize_routing()
                self._send_json(HTTPStatus.OK, routing)
                return
            if request_url.path == "/api/v2/router/decision":
                response_id = parse_qs(request_url.query).get("response_id", [""])[-1]
                if not response_id:
                    raise ValueError("response_id is required")
                plan = get_routing_plan(response_id)
                if plan is None:
                    self._send_json(HTTPStatus.NOT_FOUND, {"error": "decision not found"})
                else:
                    self._send_json(
                        HTTPStatus.OK,
                        plan.model_dump(mode="json", exclude_none=True),
                    )
                return
            if request_url.path == "/health":
                self._send_json(HTTPStatus.OK, {"status": "ok"})
                return
            if request_url.path == "/ready":
                routing = load_gateway_routing_settings(self.env_path)
                self._send_json(
                    HTTPStatus.OK,
                    {
                        "status": "ready",
                        "router_mode": routing.mode.value,
                        "configured_tiers": [tier.value for tier in routing.tiers],
                    },
                )
                return
            if request_url.path == "/api/health":
                health = create_provider(self._settings()).check_connection()
                self._send_json(HTTPStatus.OK, health.model_dump())
                return
            if request_url.path == "/v1/models":
                self._openai_models()
                return
            self._send_json(HTTPStatus.NOT_FOUND, {"error": "Not found"})
        except RequestBodyTooLarge as error:
            self._send_request_too_large(error)
        except (ProviderError, WorkerError, ValidationError, ValueError, OSError) as error:
            self._send_json(HTTPStatus.BAD_REQUEST, {"error": str(error)})

    def do_PUT(self) -> None:
        if not self._local_request():
            self._send_json(HTTPStatus.FORBIDDEN, {"error": "Local requests only"})
            return
        if self.path not in {"/api/settings", "/api/v2/settings", "/api/unload-policy"}:
            self._send_json(HTTPStatus.NOT_FOUND, {"error": "Not found"})
            return
        try:
            if self.path == "/api/unload-policy":
                body = self._read_json(max_bytes=REQUEST_LIMITS.max_ui_request_bytes)
                policy = str(body.get("policy", "immediate")).strip()
                valid = {"immediate", "never"} | {"5", "10", "30"}
                if policy not in valid:
                    raise ValueError(f"Invalid policy: {policy}. Must be one of: {sorted(valid)}")
                INFERENCE_QUEUE.set_unload_policy(policy)
                from dotenv import set_key as _sk

                _sk(str(self.env_path), "LLM_UNLOAD_POLICY", policy)
                self._send_json(HTTPStatus.OK, {"policy": policy})
                return
            if self.path == "/api/v2/settings":
                value = GatewaySettingsInput.model_validate(
                    self._read_json(max_bytes=REQUEST_LIMITS.max_ui_request_bytes)
                )
                self._send_json(
                    HTTPStatus.OK,
                    save_gateway_settings(value, self.env_path),
                )
                return
            value = ProviderSettingsInput.model_validate(
                self._read_json(max_bytes=REQUEST_LIMITS.max_ui_request_bytes)
            )
            result = save_provider_settings(value, self.env_path)
            health = create_provider(load_web_worker_settings(self.env_path)).check_connection()
            result["details"] = health.details
            self._send_json(HTTPStatus.OK, result)
        except (ProviderError, WorkerError, ValidationError, ValueError, OSError) as error:
            self._send_json(HTTPStatus.BAD_REQUEST, {"error": str(error)})

    def do_POST(self) -> None:
        response_started = False
        if not self._local_request():
            self._send_json(HTTPStatus.FORBIDDEN, {"error": "Local requests only"})
            return
        if self.path == "/api/v2/discover-models":
            try:
                body = self._read_json(max_bytes=REQUEST_LIMITS.max_ui_request_bytes)
                value = TierModelDiscoveryInput.model_validate(body)
                api_key = value.api_key.get_secret_value() if value.api_key else None
                if api_key is None and value.tier is not None:
                    # Fall back to the key stored for this tier only, never to
                    # another tier's key or unrelated environment credentials.
                    api_key = _resolve_tier_stored_key(value.tier, self.env_path)
                models = discover_tier_models(
                    value.provider, value.base_url, api_key, self.env_path
                )
                self._send_json(HTTPStatus.OK, {"models": models})
            except RequestBodyTooLarge as error:
                self._send_request_too_large(error)
            except ValidationError:
                self._send_json(
                    HTTPStatus.BAD_REQUEST, {"error": "Неверный Base URL или параметры запроса"}
                )
            except (ProviderError, ProviderConfigurationError, WorkerError, ValueError, OSError) as error:
                HTTP_LOGGER.info("model discovery failed: %s", error)
                self._send_json(HTTPStatus.BAD_REQUEST, {"error": _discovery_error_message(error)})
            return
        if self.path == "/v1/chat/completions":
            try:
                with INFERENCE_QUEUE.acquire() as lease:
                    self._chat_completion(lease)
            except RequestBodyTooLarge as error:
                self._send_request_too_large(error)
            except (ProviderError, WorkerError, ValidationError, ValueError, OSError) as error:
                self._send_json(
                    HTTPStatus.BAD_REQUEST,
                    {"error": {"message": str(error), "type": "invalid_request_error"}},
                )
            return
        if self.path == "/v1/responses":
            request_id = f"req_{uuid.uuid4().hex}"
            started = time.monotonic()
            raw_length = self.headers.get("Content-Length")
            content_length = int(raw_length) if raw_length and raw_length.isdigit() else None
            status = HTTPStatus.INTERNAL_SERVER_ERROR
            try:
                with INFERENCE_QUEUE.acquire() as lease:
                    self._response(lease, request_id)
                status = HTTPStatus.OK
            except RequestBodyTooLarge as error:
                status = HTTPStatus.REQUEST_ENTITY_TOO_LARGE
                self._send_request_too_large(error)
                HTTP_LOGGER.info(
                    json.dumps(
                        {
                            "event": "request rejected: body too large",
                            "request_id": request_id,
                            "received_bytes": error.received_bytes,
                            "max_bytes": error.max_bytes,
                        },
                        separators=(",", ":"),
                    )
                )
            except (ProviderError, WorkerError, ValidationError, ValueError, OSError) as error:
                status = HTTPStatus.BAD_REQUEST
                self._send_json(
                    HTTPStatus.BAD_REQUEST,
                    {"error": {"message": str(error), "type": "invalid_request_error"}},
                )
            finally:
                HTTP_LOGGER.info(
                    json.dumps(
                        {
                            "request_id": request_id,
                            "method": "POST",
                            "path": "/v1/responses",
                            "content_length": content_length,
                            "max_request_bytes": REQUEST_LIMITS.max_responses_request_bytes,
                            "status": int(status),
                            "elapsed_ms": round((time.monotonic() - started) * 1000, 2),
                        },
                        separators=(",", ":"),
                    )
                )
            return
        if self.path != "/api/ollama/pull":
            self._send_json(HTTPStatus.NOT_FOUND, {"error": "Not found"})
            return
        try:
            with INFERENCE_QUEUE.acquire():
                model = validate_model_name(
                    str(
                        self._read_json(max_bytes=REQUEST_LIMITS.max_ui_request_bytes).get(
                            "model", ""
                        )
                    )
                )
                settings = self._settings()
                if settings.llm_provider is not ProviderName.OLLAMA:
                    raise ValueError("Select the Ollama provider before installing a local model")
                provider = create_provider(settings)
                if not isinstance(provider, OllamaProvider):
                    raise ValueError("Configured provider cannot install Ollama models")
                chunks = iter(provider.pull_model(model))
                first_chunk = next(chunks)
                self.send_response(HTTPStatus.OK)
                self.send_header("Content-Type", "application/x-ndjson; charset=utf-8")
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                response_started = True
                for chunk in chain((first_chunk,), chunks):
                    self.wfile.write(json.dumps(chunk, ensure_ascii=False).encode("utf-8") + b"\n")
                    self.wfile.flush()
        except RequestBodyTooLarge as error:
            if not response_started:
                self._send_request_too_large(error)
        except StopIteration:
            if not response_started:
                self._send_json(HTTPStatus.BAD_REQUEST, {"error": "Ollama returned no progress"})
        except (ProviderError, WorkerError, ValidationError, ValueError, OSError) as error:
            if not response_started:
                self._send_json(HTTPStatus.BAD_REQUEST, {"error": str(error)})


def run_web_server(host: str = "127.0.0.1", port: int = 8765, env_path: Path = Path(".env")) -> int:
    global REQUEST_LIMITS, RESPONSE_STATE
    initialize_container_settings(env_path)
    REQUEST_LIMITS = load_request_limits(env_path)
    state_values = dotenv_values(env_path)
    RESPONSE_STATE = ResponseStateStore(
        max_entries=int(str(state_values.get("GATEWAY_RESPONSE_STATE_MAX_ENTRIES") or 256)),
        ttl_seconds=float(str(state_values.get("GATEWAY_RESPONSE_STATE_TTL_SECONDS") or 7200)),
    )
    unload_policy = str(state_values.get("LLM_UNLOAD_POLICY") or "immediate")
    INFERENCE_QUEUE.set_unload_policy(unload_policy)
    handler = type("ConfiguredWorkerWebHandler", (WorkerWebHandler,), {"env_path": env_path})
    server = ThreadingHTTPServer((host, port), handler)
    print(f"Local Code Worker UI: http://{host}:{port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0
