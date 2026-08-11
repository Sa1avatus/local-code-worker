# ruff: noqa: E501

import json
import time
import uuid
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from itertools import chain
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlsplit

from dotenv import dotenv_values
from pydantic import SecretStr, ValidationError

from .config import WorkerSettings
from .exceptions import ProviderError, WorkerError
from .models import JsonMode, ProviderName
from .providers import create_provider
from .providers.adapter import CanonicalProviderAdapter
from .providers.base import ProviderMessage
from .providers.ollama import OllamaProvider
from .responses.adapter import adapt_response_request
from .responses.builder import build_response
from .responses.schemas import (
    ResponseCreateRequest,
    ResponseErrorDetail,
    ResponseObject,
)
from .responses.state import ResponseStateStore
from .responses.streaming import ResponseStreamEvent, encode_sse, map_provider_events
from .routing.gateway import resolve_gateway_fallback, resolve_gateway_route
from .routing.routellm_adapter import ROUTELLM_BACKENDS
from .system_metrics import read_system_metrics
from .usage_statistics import (
    record_model_call,
    record_routing_plan,
    summarize_model_calls,
    summarize_v2_statistics,
)
from .virtual_models import VIRTUAL_MODEL_REGISTRY
from .web_config import (
    initialize_container_settings,
    load_gateway_routing_settings,
    load_public_settings,
    load_web_worker_settings,
    public_gateway_settings,
    save_gateway_settings,
    save_provider_settings,
)
from .web_models import GatewaySettingsInput, ProviderSettingsInput, validate_model_name

MAX_REQUEST_BYTES = 64 * 1024
RESPONSE_STATE = ResponseStateStore()
LOCAL_HOSTS = {
    "localhost",
    "127.0.0.1",
    "::1",
    "host.docker.internal",
    "local-code-worker-web",
}

INDEX_HTML = r"""<!doctype html>
<html lang="ru"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Local Code Worker</title><style>
:root{color-scheme:dark;--bg:#0b1020;--panel:#131b31;--line:#2b3859;--text:#e8edff;--muted:#93a4ca;--accent:#7c9cff;--ok:#55d6a3;--bad:#ff7d91}*{box-sizing:border-box}
body{margin:0;background:radial-gradient(circle at 10% 0,#1c2850 0,transparent 38%),var(--bg);font:15px/1.5 system-ui;color:var(--text)}
main{max-width:920px;margin:42px auto;padding:0 20px}.hero{display:flex;justify-content:space-between;gap:24px;align-items:end;margin-bottom:22px}h1{font-size:32px;margin:0}p{color:var(--muted)}
.card{background:color-mix(in srgb,var(--panel) 94%,transparent);border:1px solid var(--line);border-radius:18px;padding:22px;box-shadow:0 22px 70px #0005}.grid{display:grid;grid-template-columns:1fr 1fr;gap:16px}.wide{grid-column:1/-1}
label{display:block;color:var(--muted);font-size:13px;margin-bottom:6px}input,select,button,textarea{width:100%;border:1px solid var(--line);border-radius:10px;background:#0d1428;color:var(--text);padding:11px 12px;font:inherit}button{cursor:pointer;background:var(--accent);color:#081126;border:0;font-weight:750}button.secondary{background:#243252;color:var(--text)}button:disabled{opacity:.45;cursor:not-allowed}.actions{display:flex;gap:12px;margin-top:18px}.status{min-height:24px;margin:12px 0 0;color:var(--muted)}.ok{color:var(--ok)}.bad{color:var(--bad)}textarea{height:150px;resize:vertical;font-family:ui-monospace,monospace}.keyrow{display:flex;gap:9px}.keyrow input{flex:1}.keyrow button{width:auto}.pill{padding:5px 9px;border:1px solid var(--line);border-radius:99px;color:var(--muted);font-size:12px}@media(max-width:650px){.grid{grid-template-columns:1fr}.wide{grid-column:auto}.hero{display:block}.actions{flex-direction:column}}
.tiers{display:grid;grid-template-columns:repeat(3,1fr);gap:12px}.tier{padding:16px;border:1px solid var(--line);border-radius:14px;background:#0d1428}.tier h3{margin:0 0 2px}.tier p{font-size:12px;margin:0 0 12px}.tier label{margin-top:9px}.check{display:flex;align-items:center;gap:8px}.check input{width:auto}.route-head{display:grid;grid-template-columns:2fr 1fr 1fr;gap:12px;margin-bottom:16px}@media(max-width:850px){.tiers{grid-template-columns:1fr}.route-head{grid-template-columns:1fr}}
</style></head><body><main><div class="hero"><div><h1>Local Code Worker</h1><p>Провайдеры, ключи и локальные модели — в одном локальном интерфейсе.</p></div><span class="pill" id="health">проверка…</span></div>
<section class="card" id="routingSettings"><h2 style="margin-top:0">Маршрутизация моделей</h2><p>Запрос начинается с локальных уровней. STRONG используйте как облачную страховку, если локальные модели не справились.</p><div class="route-head"><div><label for="routeMode">Режим</label><select id="routeMode"><option value="router">Router — применять выбор</option><option value="observe_only">Observe only — только наблюдать</option><option value="legacy">Legacy — одна модель ниже</option></select></div><div class="check"><input id="routeLlm" type="checkbox"><label for="routeLlm">RouteLLM</label></div><div><label for="routeThreshold">Порог RouteLLM</label><input id="routeThreshold" type="number" min="0" max="1" step="0.05"></div></div><div class="tiers" id="tierCards"></div><datalist id="routingModels"></datalist><div class="actions"><button id="saveRouting">Сохранить маршрутизацию</button><button class="secondary" id="discoverRouting">Найти локальные модели</button></div><div class="status" id="routingStatus"></div></section>
<section class="card"><div class="grid"><div><label for="provider">Провайдер</label><select id="provider"><option value="ollama">Ollama (локально)</option><option value="openai-compatible">OpenAI-compatible API</option></select></div><div><label for="baseUrl">Base URL</label><input id="baseUrl"></div><div><label for="model">Модель</label><select id="model"></select></div><div><label for="contextLength">Контекст (токены)</label><input id="contextLength" type="number" min="512" max="131072" step="512"></div><div id="pullBlock"><label for="pullModel">Имя модели для скачивания (необязательно)</label><input id="pullModel" placeholder="qwen2.5-coder:7b-instruct-q5_K_M"></div><div id="keyBlock"><label for="apiKey">API-ключ (пусто = сохранить текущий)</label><div class="keyrow"><input id="apiKey" type="password" autocomplete="new-password" placeholder="••••••••"><button class="secondary" id="clearKey" type="button">Удалить</button></div></div></div>
<div class="actions"><button id="save">Сохранить и проверить</button><button class="secondary" id="refresh">Обновить модели</button><button class="secondary" id="pull">Скачать модель</button></div><div class="status" id="status"></div><textarea id="progress" readonly placeholder="Прогресс загрузки Ollama…"></textarea></section><section class="card" style="margin-top:18px"><div class="hero" style="margin-bottom:12px"><div><h2 style="margin:0">Мониторинг системы и моделей</h2><p style="margin:4px 0 0">Обновляется каждые 15 секунд.</p></div><span class="pill" id="runtimeUpdated">проверка…</span></div><div class="grid" id="metrics"></div><div class="status" id="runtime">Проверка состояния Ollama…</div></section><section class="card" style="margin-top:18px"><h2 style="margin:0">Статистика обращений</h2><p>Токены, средняя скорость и проверка предложенного кода по моделям.</p><div class="grid" id="usageStats"></div></section></main>
<script>
const $=id=>document.getElementById(id), provider=$('provider'), baseUrl=$('baseUrl'), model=$('model'), contextLength=$('contextLength'), pullModel=$('pullModel'), apiKey=$('apiKey'), status=$('status'), progress=$('progress'), runtime=$('runtime'), runtimeUpdated=$('runtimeUpdated'), metrics=$('metrics'), usageStats=$('usageStats');let clearKey=false;
const tierNames=['local','mid','strong'],tierLabels={local:'LOCAL',mid:'MID',strong:'STRONG'},tierHelp={local:'Первая локальная модель: быстрые и простые задачи.',mid:'Локальная модель для рассуждений и сложного исполнения.',strong:'Последний уровень; здесь можно указать облачную модель.'},clearedTierKeys=new Set();
function routingMessage(text,ok=true){const node=$('routingStatus');node.textContent=text;node.className='status '+(ok?'ok':'bad')}
function tierCard(name){const card=document.createElement('div');card.className='tier';card.dataset.tier=name;card.innerHTML=`<h3>${tierLabels[name]}</h3><p>${tierHelp[name]}</p><div class="check"><input id="${name}Enabled" type="checkbox"><label for="${name}Enabled">Уровень включён</label></div><label for="${name}Provider">Провайдер</label><select id="${name}Provider"><option value="ollama">Ollama</option><option value="openai-compatible">OpenAI-compatible</option></select><label for="${name}BaseUrl">Base URL</label><input id="${name}BaseUrl"><label for="${name}Model">Модель</label><input id="${name}Model" list="routingModels"><label for="${name}Context">Контекст</label><input id="${name}Context" type="number" min="512" max="131072" step="512"><label for="${name}Key">API-ключ (пусто = оставить)</label><div class="keyrow"><input id="${name}Key" type="password" autocomplete="new-password"><button class="secondary" type="button" data-clear-key="${name}">×</button></div><small id="${name}KeyState"></small>`;return card}
for(const name of tierNames)$('tierCards').appendChild(tierCard(name));
function fillTier(name,data){$(name+'Enabled').checked=data.enabled;$(name+'Provider').value=data.provider;$(name+'BaseUrl').value=data.base_url||'';$(name+'Model').value=data.model;$(name+'Context').value=data.context_length||32768;$(name+'Key').value='';$(name+'KeyState').textContent=data.api_key_configured?'Ключ сохранён':'Ключ не задан'}
function tierPayload(name){const key=$(name+'Key').value,clear=clearedTierKeys.has(name),action=clear?'clear':key?'replace':'keep';return {enabled:$(name+'Enabled').checked,provider:$(name+'Provider').value,base_url:$(name+'BaseUrl').value,model:$(name+'Model').value,context_length:Number($(name+'Context').value),api_key_action:action,api_key:action==='replace'?key:null}}
async function loadRouting(){try{const data=await jsonFetch('/api/v2/settings');$('routeMode').value=data.mode;$('routeLlm').checked=data.routellm_enabled;$('routeThreshold').value=data.routellm_threshold;for(const name of tierNames)fillTier(name,data.tiers[name]);routingMessage('Маршрутизация загружена')}catch(e){routingMessage(e.message,false)}}
async function saveRouting(){try{const tiers=Object.fromEntries(tierNames.map(name=>[name,tierPayload(name)])),data=await jsonFetch('/api/v2/settings',{method:'PUT',body:JSON.stringify({mode:$('routeMode').value,tiers,routellm_enabled:$('routeLlm').checked,routellm_threshold:Number($('routeThreshold').value)})});clearedTierKeys.clear();for(const name of tierNames)fillTier(name,data.tiers[name]);routingMessage('Маршрутизация сохранена',true)}catch(e){routingMessage(e.message,false)}}
async function discoverRouting(){try{const data=await jsonFetch('/api/models'),list=$('routingModels');list.replaceChildren();for(const name of data.models){const option=document.createElement('option');option.value=name;list.appendChild(option)}routingMessage(`Найдено локальных моделей: ${data.models.length}`)}catch(e){routingMessage(e.message,false)}}
function message(text,ok=true){status.textContent=text;status.className='status '+(ok?'ok':'bad')}
function sync(){const local=provider.value==='ollama';$('keyBlock').style.display=local?'none':'block';$('pullBlock').style.display=local?'block':'none';$('pull').disabled=!local;if(local&&(!baseUrl.value||baseUrl.value.includes('example')))baseUrl.value='http://localhost:11434'}
async function jsonFetch(url,options={}){const response=await fetch(url,{headers:{'Content-Type':'application/json'},...options});const data=await response.json();if(!response.ok)throw new Error(data.error||response.statusText);return data}
function bytes(value){return typeof value==='number'?(value/1024/1024/1024).toFixed(1)+' ГБ':'—'}
function meter(title,value,caption,percent=0){const card=document.createElement('div');card.style.cssText='padding:14px;border:1px solid var(--line);border-radius:12px;background:#0d1428';const heading=document.createElement('div');heading.style.color='var(--muted)';heading.textContent=title;const number=document.createElement('div');number.style.cssText='font-size:25px;font-weight:750;margin-top:3px;color:var(--accent)';number.textContent=value;const bar=document.createElement('div');bar.style.cssText='height:7px;background:#253252;border-radius:99px;margin:10px 0 7px;overflow:hidden';const fill=document.createElement('div');fill.style.cssText=`height:100%;width:${Math.max(0,Math.min(100,percent))}%;background:linear-gradient(90deg,var(--accent),var(--ok));border-radius:inherit;transition:width .35s ease`;bar.appendChild(fill);const text=document.createElement('div');text.style.color='var(--muted)';text.textContent=caption;card.append(heading,number,bar,text);return card}
function renderMetrics(data){metrics.replaceChildren();const gpu=data.gpus&&data.gpus[0];if(gpu){metrics.append(meter(gpu.name,`${gpu.usage_percent}% GPU`,`${gpu.temperature_celsius} °C · ${bytes(gpu.memory_used_bytes)} / ${bytes(gpu.memory_total_bytes)} VRAM`,gpu.usage_percent),meter('Питание и частота',`${gpu.power_watts} Вт`,`лимит ${gpu.power_limit_watts} Вт · ${gpu.clock_mhz} МГц`,gpu.power_limit_watts?gpu.power_watts/gpu.power_limit_watts*100:0))}else{metrics.append(meter('GPU','Недоступна','Docker не передал NVIDIA NVML в контейнер'))}metrics.append(meter('CPU контейнера',`${data.cpu.usage_percent}% CPU`,'текущая загрузка',data.cpu.usage_percent),meter('RAM контейнера',`${data.memory.usage_percent}% RAM`,`${bytes(data.memory.used_bytes)} / ${bytes(data.memory.total_bytes)}`,data.memory.usage_percent))}
function renderUsage(data){usageStats.replaceChildren();if(!data.models.length){usageStats.textContent='Пока нет завершённых обращений.';return}for(const item of data.models){usageStats.append(meter(item.model,`${item.completion_tokens} выходных токенов`,`${item.requests} запросов · ${item.prompt_tokens} входных · ${item.tokens_per_second} ток/с · код: ${item.code_valid} успешно, ${item.code_invalid} с ошибкой`))}}
function runtimeCard(item){const total=item.size||0,vram=item.size_vram||0,gpu=total?Math.round(vram/total*100):0,card=document.createElement('div');card.style.cssText='padding:12px;border:1px solid var(--line);border-radius:10px;margin-top:8px';const title=document.createElement('strong');title.textContent=item.name;const details=document.createElement('div');details.style.color='var(--muted)';details.textContent=`GPU: ${gpu}% · CPU: ${100-gpu}% · VRAM модели: ${bytes(vram)} из ${bytes(total)} · Контекст: ${item.context_length||'—'}`;card.append(title,details);return card}
async function runtimeStatus(){try{const [data,system,usage]=await Promise.all([jsonFetch('/api/runtime'),jsonFetch('/api/system'),jsonFetch('/api/statistics')]);renderMetrics(system);renderUsage(usage);runtime.replaceChildren();if(data.provider!=='ollama'){runtime.textContent='Статус размещения доступен только для Ollama.'}else if(!data.models.length){runtime.textContent='Сейчас ни одна модель не загружена в Ollama.'}else{for(const item of data.models)runtime.appendChild(runtimeCard(item));}runtimeUpdated.textContent='обновлено '+new Date().toLocaleTimeString()}catch(e){runtime.textContent='Не удалось получить статус: '+e.message;runtimeUpdated.textContent='ошибка'}}
async function load(){try{const data=await jsonFetch('/api/settings');provider.value=data.provider;baseUrl.value=data.base_url;contextLength.value=data.context_length;sync();await models(data.model||'');await runtimeStatus();$('health').textContent=data.api_key_configured?'ключ сохранён':'локальная конфигурация'}catch(e){message(e.message,false)}}
async function save(){try{const action=clearKey?'clear':apiKey.value?'replace':'keep';const data=await jsonFetch('/api/settings',{method:'PUT',body:JSON.stringify({provider:provider.value,base_url:baseUrl.value,model:model.value,context_length:Number(contextLength.value),api_key_action:action,api_key:action==='replace'?apiKey.value:null})});apiKey.value='';clearKey=false;message('Настройки сохранены. Выгрузите модель через ollama stop, чтобы применить новый контекст.',true);$('health').textContent=data.api_key_configured?'ключ сохранён':'готово'}catch(e){message(e.message,false)}}
async function models(preferredModel=model.value){preferredModel=typeof preferredModel==='string'?preferredModel:'';try{const data=await jsonFetch('/api/models');const names=[...new Set((Array.isArray(data.models)?data.models:[]).filter(name=>typeof name==='string'&&name.trim()).map(name=>name.trim()))];if(preferredModel&&!names.includes(preferredModel))names.unshift(preferredModel);model.replaceChildren();for(const name of names){const option=document.createElement('option');option.value=name;option.textContent=name;model.appendChild(option)}if(preferredModel)model.value=preferredModel;message(`Найдено моделей: ${data.models.length}`)}catch(e){message(e.message,false)}}
async function pull(){progress.value='';message('Загрузка модели…');const requestedModel=pullModel.value.trim()||model.value;try{const response=await fetch('/api/ollama/pull',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({model:requestedModel})});if(!response.ok){const d=await response.json();throw new Error(d.error)}const reader=response.body.getReader(),decoder=new TextDecoder();let pending='';while(true){const {value,done}=await reader.read();pending+=decoder.decode(value||new Uint8Array(),{stream:!done});const lines=pending.split('\n');pending=lines.pop();for(const line of lines)if(line){const item=JSON.parse(line);progress.value+=JSON.stringify(item)+'\n';progress.scrollTop=progress.scrollHeight}if(done)break}pullModel.value='';await models(requestedModel);message('Модель установлена',true)}catch(e){message(e.message,false)}}
provider.addEventListener('change',sync);$('save').onclick=save;$('refresh').onclick=()=>models();$('pull').onclick=pull;$('clearKey').onclick=()=>{clearKey=true;apiKey.value='';message('Ключ будет удалён после сохранения')};$('saveRouting').onclick=saveRouting;$('discoverRouting').onclick=discoverRouting;$('tierCards').onclick=event=>{const name=event.target.dataset.clearKey;if(name){clearedTierKeys.add(name);$(name+'Key').value='';$(name+'KeyState').textContent='Ключ будет удалён после сохранения'}};load();loadRouting();window.setInterval(runtimeStatus,15000);
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

    def _read_json(self) -> dict[str, Any]:
        try:
            length = int(self.headers.get("Content-Length", "0"))
        except ValueError as error:
            raise ValueError("Invalid Content-Length") from error
        if length <= 0 or length > MAX_REQUEST_BYTES:
            raise ValueError("Request body size is invalid")
        payload = json.loads(self.rfile.read(length))
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
        self._send_json(
            HTTPStatus.OK,
            {
                "object": "list",
                "data": [
                    {
                        "id": model.id,
                        "object": "model",
                        "created": 0,
                        "owned_by": "local-code-worker",
                    }
                    for model in VIRTUAL_MODEL_REGISTRY.list_models()
                ],
            },
        )

    def _chat_request(self) -> tuple[WorkerSettings, list[dict[str, str]], bool, str]:
        payload = self._read_json()
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
            updates["llm_max_output_tokens"] = min(settings.llm_max_output_tokens, max_tokens)
        response_format = payload.get("response_format")
        if response_format is not None:
            if not isinstance(response_format, dict):
                raise ValueError("response_format must be an object")
            response_type = response_format.get("type")
            if response_type == "json_object":
                updates["llm_json_mode"] = JsonMode.JSON_OBJECT
            elif response_type not in {None, "text"}:
                raise ValueError("only text and json_object response formats are supported")
        stream = payload.get("stream", False)
        if not isinstance(stream, bool):
            raise ValueError("stream must be a boolean")
        return settings.model_copy(update=updates), messages, stream, virtual_model.id

    def _chat_completion(self) -> None:
        settings, messages, stream, public_model = self._chat_request()
        provider = create_provider(settings)
        content = provider.chat(
            messages,
            None,
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

    def _response(self) -> None:
        request = ResponseCreateRequest.model_validate(self._read_json())
        virtual_model = VIRTUAL_MODEL_REGISTRY.resolve(request.model)
        if request.tools and request.stream:
            raise ValueError("streaming function tools are not supported yet")
        settings = self._settings().model_copy(update={"llm_stream": request.stream})
        provider_request = adapt_response_request(
            request,
            max_output_characters=settings.llm_max_output_characters,
            json_mode=settings.llm_json_mode,
        )
        if request.previous_response_id is not None:
            previous_messages = RESPONSE_STATE.get(request.previous_response_id)
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
        )
        settings = self._settings_with_tier_secret(settings)
        provider = create_provider(settings)
        message_id = f"msg_{uuid.uuid4().hex}"
        created_at = int(time.time())
        if not request.stream:
            while True:
                try:
                    result = CanonicalProviderAdapter(provider).complete(provider_request)
                    break
                except (ProviderError, WorkerError, OSError):
                    fallback = resolve_gateway_fallback(
                        settings,
                        routing_settings,
                        routing_plan.actual.tier,
                    )
                    if fallback is None:
                        record_routing_plan(response_id, routing_plan)
                        raise
                    settings, routing_plan = fallback
                    settings = self._settings_with_tier_secret(settings)
                    provider = create_provider(settings)
            record_routing_plan(response_id, routing_plan)
            record_model_call(
                provider.last_generation_metadata,
                kind="response",
                outcome="completed",
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
                RESPONSE_STATE.put(
                    response_id,
                    provider_request.messages
                    + [ProviderMessage(role="assistant", content=result.content)],
                )
            return

        while True:
            provider_events = iter(provider.stream(provider_request))
            try:
                first_provider_event = next(provider_events)
                break
            except (ProviderError, WorkerError, OSError):
                fallback = resolve_gateway_fallback(
                    settings,
                    routing_settings,
                    routing_plan.actual.tier,
                )
                if fallback is None:
                    record_routing_plan(response_id, routing_plan)
                    raise
                settings, routing_plan = fallback
                settings = self._settings_with_tier_secret(settings)
                provider = create_provider(settings)
            except StopIteration as error:
                raise ProviderError(
                    "provider stream ended before the first event",
                    category="empty_stream",
                ) from error
        record_routing_plan(response_id, routing_plan)
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "text/event-stream; charset=utf-8")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Connection", "close")
        self.end_headers()
        provider_events = chain([first_provider_event], provider_events)
        last_sequence = -1
        completed_response = None
        try:
            try:
                for event in map_provider_events(
                    provider_events,
                    provider=settings.llm_provider,
                    model=virtual_model.id,
                    response_id=response_id,
                    message_id=message_id,
                    created_at=created_at,
                ):
                    last_sequence = event.sequence_number
                    if event.type == "response.completed":
                        completed_response = event.response
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
        finally:
            close = getattr(provider_events, "close", None)
            if close is not None:
                close()
        if completed_response is not None:
            record_model_call(
                provider.last_generation_metadata,
                kind="response",
                outcome="completed",
            )
        if request.store and completed_response is not None:
            RESPONSE_STATE.put(
                response_id,
                provider_request.messages
                + [
                    ProviderMessage(
                        role="assistant",
                        content=completed_response.output_text,
                    )
                ],
            )

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
            if request_url.path == "/api/health":
                health = create_provider(self._settings()).check_connection()
                self._send_json(HTTPStatus.OK, health.model_dump())
                return
            if request_url.path == "/v1/models":
                self._openai_models()
                return
            self._send_json(HTTPStatus.NOT_FOUND, {"error": "Not found"})
        except (ProviderError, WorkerError, ValidationError, ValueError, OSError) as error:
            self._send_json(HTTPStatus.BAD_REQUEST, {"error": str(error)})

    def do_PUT(self) -> None:
        if not self._local_request():
            self._send_json(HTTPStatus.FORBIDDEN, {"error": "Local requests only"})
            return
        if self.path not in {"/api/settings", "/api/v2/settings"}:
            self._send_json(HTTPStatus.NOT_FOUND, {"error": "Not found"})
            return
        try:
            if self.path == "/api/v2/settings":
                value = GatewaySettingsInput.model_validate(self._read_json())
                self._send_json(
                    HTTPStatus.OK,
                    save_gateway_settings(value, self.env_path),
                )
                return
            value = ProviderSettingsInput.model_validate(self._read_json())
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
        if self.path == "/v1/chat/completions":
            try:
                self._chat_completion()
            except (ProviderError, WorkerError, ValidationError, ValueError, OSError) as error:
                self._send_json(
                    HTTPStatus.BAD_REQUEST,
                    {"error": {"message": str(error), "type": "invalid_request_error"}},
                )
            return
        if self.path == "/v1/responses":
            try:
                self._response()
            except (ProviderError, WorkerError, ValidationError, ValueError, OSError) as error:
                self._send_json(
                    HTTPStatus.BAD_REQUEST,
                    {"error": {"message": str(error), "type": "invalid_request_error"}},
                )
            return
        if self.path != "/api/ollama/pull":
            self._send_json(HTTPStatus.NOT_FOUND, {"error": "Not found"})
            return
        try:
            model = validate_model_name(str(self._read_json().get("model", "")))
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
        except StopIteration:
            if not response_started:
                self._send_json(HTTPStatus.BAD_REQUEST, {"error": "Ollama returned no progress"})
        except (ProviderError, WorkerError, ValidationError, ValueError, OSError) as error:
            if not response_started:
                self._send_json(HTTPStatus.BAD_REQUEST, {"error": str(error)})


def run_web_server(host: str = "127.0.0.1", port: int = 8765, env_path: Path = Path(".env")) -> int:
    initialize_container_settings(env_path)
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
