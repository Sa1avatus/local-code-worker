# ruff: noqa: E501

import json
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from itertools import chain
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

from pydantic import ValidationError

from .config import WorkerSettings
from .exceptions import ProviderError, WorkerError
from .models import ProviderName
from .providers import create_provider
from .providers.ollama import OllamaProvider
from .web_config import (
    initialize_container_settings,
    load_public_settings,
    load_web_worker_settings,
    save_provider_settings,
)
from .web_models import ProviderSettingsInput, validate_model_name

MAX_REQUEST_BYTES = 64 * 1024
LOCAL_HOSTS = {"localhost", "127.0.0.1", "::1"}

INDEX_HTML = r'''<!doctype html>
<html lang="ru"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Local Code Worker</title><style>
:root{color-scheme:dark;--bg:#0b1020;--panel:#131b31;--line:#2b3859;--text:#e8edff;--muted:#93a4ca;--accent:#7c9cff;--ok:#55d6a3;--bad:#ff7d91}*{box-sizing:border-box}
body{margin:0;background:radial-gradient(circle at 10% 0,#1c2850 0,transparent 38%),var(--bg);font:15px/1.5 system-ui;color:var(--text)}
main{max-width:920px;margin:42px auto;padding:0 20px}.hero{display:flex;justify-content:space-between;gap:24px;align-items:end;margin-bottom:22px}h1{font-size:32px;margin:0}p{color:var(--muted)}
.card{background:color-mix(in srgb,var(--panel) 94%,transparent);border:1px solid var(--line);border-radius:18px;padding:22px;box-shadow:0 22px 70px #0005}.grid{display:grid;grid-template-columns:1fr 1fr;gap:16px}.wide{grid-column:1/-1}
label{display:block;color:var(--muted);font-size:13px;margin-bottom:6px}input,select,button,textarea{width:100%;border:1px solid var(--line);border-radius:10px;background:#0d1428;color:var(--text);padding:11px 12px;font:inherit}button{cursor:pointer;background:var(--accent);color:#081126;border:0;font-weight:750}button.secondary{background:#243252;color:var(--text)}button:disabled{opacity:.45;cursor:not-allowed}.actions{display:flex;gap:12px;margin-top:18px}.status{min-height:24px;margin:12px 0 0;color:var(--muted)}.ok{color:var(--ok)}.bad{color:var(--bad)}textarea{height:150px;resize:vertical;font-family:ui-monospace,monospace}.keyrow{display:flex;gap:9px}.keyrow input{flex:1}.keyrow button{width:auto}.pill{padding:5px 9px;border:1px solid var(--line);border-radius:99px;color:var(--muted);font-size:12px}@media(max-width:650px){.grid{grid-template-columns:1fr}.wide{grid-column:auto}.hero{display:block}.actions{flex-direction:column}}
</style></head><body><main><div class="hero"><div><h1>Local Code Worker</h1><p>Провайдеры, ключи и локальные модели — в одном локальном интерфейсе.</p></div><span class="pill" id="health">проверка…</span></div>
<section class="card"><div class="grid"><div><label for="provider">Провайдер</label><select id="provider"><option value="ollama">Ollama (локально)</option><option value="openai-compatible">OpenAI-compatible API</option></select></div><div><label for="baseUrl">Base URL</label><input id="baseUrl"></div><div><label for="model">Модель</label><input id="model" list="models"><datalist id="models"></datalist></div><div id="keyBlock"><label for="apiKey">API-ключ (пусто = сохранить текущий)</label><div class="keyrow"><input id="apiKey" type="password" autocomplete="new-password" placeholder="••••••••"><button class="secondary" id="clearKey" type="button">Удалить</button></div></div></div>
<div class="actions"><button id="save">Сохранить и проверить</button><button class="secondary" id="refresh">Обновить модели</button><button class="secondary" id="pull">Скачать модель</button></div><div class="status" id="status"></div><textarea id="progress" readonly placeholder="Прогресс загрузки Ollama…"></textarea></section></main>
<script>
const $=id=>document.getElementById(id), provider=$('provider'), baseUrl=$('baseUrl'), model=$('model'), apiKey=$('apiKey'), status=$('status'), progress=$('progress');let clearKey=false;
function message(text,ok=true){status.textContent=text;status.className='status '+(ok?'ok':'bad')}
function sync(){const local=provider.value==='ollama';$('keyBlock').style.display=local?'none':'block';$('pull').disabled=!local;if(local&&(!baseUrl.value||baseUrl.value.includes('example')))baseUrl.value='http://localhost:11434'}
async function jsonFetch(url,options={}){const response=await fetch(url,{headers:{'Content-Type':'application/json'},...options});const data=await response.json();if(!response.ok)throw new Error(data.error||response.statusText);return data}
async function load(){try{const data=await jsonFetch('/api/settings');provider.value=data.provider;baseUrl.value=data.base_url;model.value=data.model||'';$('health').textContent=data.api_key_configured?'ключ сохранён':'локальная конфигурация';sync();await models()}catch(e){message(e.message,false)}}
async function save(){try{const action=clearKey?'clear':apiKey.value?'replace':'keep';const data=await jsonFetch('/api/settings',{method:'PUT',body:JSON.stringify({provider:provider.value,base_url:baseUrl.value,model:model.value,api_key_action:action,api_key:action==='replace'?apiKey.value:null})});apiKey.value='';clearKey=false;message('Настройки сохранены. '+data.details,true);$('health').textContent=data.api_key_configured?'ключ сохранён':'готово'}catch(e){message(e.message,false)}}
async function models(){try{const data=await jsonFetch('/api/models');$('models').innerHTML='';for(const name of data.models){const o=document.createElement('option');o.value=name;$('models').appendChild(o)}message(`Найдено моделей: ${data.models.length}`)}catch(e){message(e.message,false)}}
async function pull(){progress.value='';message('Загрузка модели…');try{const response=await fetch('/api/ollama/pull',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({model:model.value})});if(!response.ok){const d=await response.json();throw new Error(d.error)}const reader=response.body.getReader(),decoder=new TextDecoder();let pending='';while(true){const {value,done}=await reader.read();pending+=decoder.decode(value||new Uint8Array(),{stream:!done});const lines=pending.split('\n');pending=lines.pop();for(const line of lines)if(line){const item=JSON.parse(line);progress.value+=JSON.stringify(item)+'\n';progress.scrollTop=progress.scrollHeight}if(done)break}message('Модель установлена',true);await models()}catch(e){message(e.message,false)}}
provider.addEventListener('change',sync);$('save').onclick=save;$('refresh').onclick=models;$('pull').onclick=pull;$('clearKey').onclick=()=>{clearKey=true;apiKey.value='';message('Ключ будет удалён после сохранения')};load();
</script></body></html>'''


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

    def do_GET(self) -> None:
        if self.path == "/":
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
            if self.path == "/api/settings":
                self._send_json(HTTPStatus.OK, load_public_settings(self.env_path))
                return
            if self.path == "/api/models":
                models = create_provider(self._settings()).list_models()
                self._send_json(HTTPStatus.OK, {"models": models})
                return
            if self.path == "/api/health":
                health = create_provider(self._settings()).check_connection()
                self._send_json(HTTPStatus.OK, health.model_dump())
                return
            self._send_json(HTTPStatus.NOT_FOUND, {"error": "Not found"})
        except (ProviderError, WorkerError, ValidationError, ValueError, OSError) as error:
            self._send_json(HTTPStatus.BAD_REQUEST, {"error": str(error)})

    def do_PUT(self) -> None:
        if not self._local_request():
            self._send_json(HTTPStatus.FORBIDDEN, {"error": "Local requests only"})
            return
        if self.path != "/api/settings":
            self._send_json(HTTPStatus.NOT_FOUND, {"error": "Not found"})
            return
        try:
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
