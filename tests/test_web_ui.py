from local_code_worker.web_app import INDEX_HTML


def test_model_picker_uses_native_select_and_separate_pull_input() -> None:
    assert '<select id="model"></select>' in INDEX_HTML
    assert '<input id="pullModel"' in INDEX_HTML
    assert '<datalist id="models">' not in INDEX_HTML
    assert 'list="models"' not in INDEX_HTML
    assert 'id="contextLength"' in INDEX_HTML


def test_discovered_models_are_safe_options_and_preserve_selection() -> None:
    assert "document.createElement('option')" in INDEX_HTML
    assert "option.textContent=name" in INDEX_HTML
    assert "model.replaceChildren()" in INDEX_HTML
    assert "if(preferredModel&&!names.includes(preferredModel))" in INDEX_HTML
    assert "await models(data.model||'')" in INDEX_HTML
    assert "preferredModel=typeof preferredModel==='string'?preferredModel:''" in INDEX_HTML
    assert "$('refresh').onclick=()=>models()" in INDEX_HTML


def test_pull_prefers_manual_name_and_selects_downloaded_model() -> None:
    assert "pullModel.value.trim()||model.value" in INDEX_HTML
    assert "await models(requestedModel)" in INDEX_HTML
    assert "pullModel.value=''" in INDEX_HTML


def test_runtime_status_uses_ollama_ps_and_polls_every_15_seconds() -> None:
    assert 'id="runtime"' in INDEX_HTML
    assert 'id="runtimeUpdated"' in INDEX_HTML
    assert "'/api/runtime'" in INDEX_HTML
    assert "window.setInterval(runtimeStatus,15000)" in INDEX_HTML


def test_dashboard_requests_system_metrics_and_renders_cards() -> None:
    assert 'id="metrics"' in INDEX_HTML
    assert "'/api/system'" in INDEX_HTML
    assert "'/api/inference'" in INDEX_HTML
    assert "inference.active?'Выполняется':'Ожидание'" in INDEX_HTML
    assert "`в очереди: ${inference.waiting}" in INDEX_HTML
    assert "renderMetrics(system)" in INDEX_HTML
    assert "linear-gradient(90deg,var(--accent),var(--ok))" in INDEX_HTML
    assert "CPU системы" in INDEX_HTML
    assert "RAM системы" in INDEX_HTML
    assert "CPU контейнера" not in INDEX_HTML
    assert "RAM контейнера" not in INDEX_HTML
    assert 'id="usageStats"' in INDEX_HTML
    assert "'/api/statistics'" in INDEX_HTML


def test_usage_statistics_render_as_compact_safe_table() -> None:
    assert 'class="usage-table"' in INDEX_HTML
    assert '<tbody id="usageStats"></tbody>' in INDEX_HTML
    assert 'id="usageUpdated"' in INDEX_HTML
    assert "const row=document.createElement('tr')" in INDEX_HTML
    assert "modelName.textContent=item.model" in INDEX_HTML
    assert "API успешно" in INDEX_HTML
    assert "API с ошибкой" in INDEX_HTML
    assert "Proposal валиден" in INDEX_HTML
    assert "Proposal невалиден" in INDEX_HTML
    assert "item.api_completed,item.api_failed,item.code_valid,item.code_invalid" in INDEX_HTML
    assert "cell.colSpan=9" in INDEX_HTML
    assert "usageStats.appendChild(row)" in INDEX_HTML


def test_context_length_is_loaded_saved_and_requires_model_reload() -> None:
    assert "contextLength.value=data.context_length" in INDEX_HTML
    assert "context_length:Number(contextLength.value)" in INDEX_HTML
    assert "ollama stop" in INDEX_HTML


def test_routing_ui_configures_all_tiers_through_v2_settings() -> None:
    assert 'id="routeMode"' in INDEX_HTML
    assert "tierNames=['local','mid','strong']" in INDEX_HTML
    assert "jsonFetch('/api/v2/settings')" in INDEX_HTML
    assert "method:'PUT'" in INDEX_HTML
    assert "api_key_action:action" in INDEX_HTML
    assert "loadRouting()" in INDEX_HTML


def test_legacy_provider_panel_is_hidden_for_full_router_modes() -> None:
    assert "legacySettings=provider.closest('section.card')" in INDEX_HTML
    assert "legacyModes=new Set(['legacy','observe_only','shadow','canary'])" in INDEX_HTML
    assert "legacySettings.hidden=!legacyModes.has($('routeMode').value)" in INDEX_HTML
    assert "$('routeMode').addEventListener('change',syncLegacySettings)" in INDEX_HTML
    assert "$('routeMode').value=data.mode;syncLegacySettings()" in INDEX_HTML


def test_routing_ui_discovers_models_per_tier_without_inserting_html() -> None:
    assert '<select id="${name}Model"></select>' in INDEX_HTML
    assert 'id="${name}FindModels"' in INDEX_HTML
    assert 'data-find-models="${name}"' in INDEX_HTML
    assert "Найти модели" in INDEX_HTML
    assert "discoverRouting" not in INDEX_HTML
    assert "Найти локальные модели" not in INDEX_HTML
    assert 'id="routingModels"' not in INDEX_HTML
    assert "async function findModels(name)" in INDEX_HTML
    assert "'/api/v2/discover-models'" in INDEX_HTML
    assert "select.replaceChildren()" in INDEX_HTML
    assert "const option=document.createElement('option')" in INDEX_HTML
    assert "option.value=model" in INDEX_HTML
    assert "option.textContent=model" in INDEX_HTML
    assert "setTierModel(name,current)" in INDEX_HTML


def test_routing_ui_find_button_sits_below_api_key() -> None:
    assert 'KeyState"></small><button class="secondary" id="${name}FindModels"' in INDEX_HTML
    assert (
        'id="${name}FindModels" type="button" data-find-models="${name}">Найти модели</button>'
        in INDEX_HTML
    )


def test_routing_ui_find_models_uses_only_its_own_card_settings() -> None:
    assert (
        "tier:name,provider:$(name+'Provider').value,base_url:baseUrl,"
        "api_key:$(name+'Key').value||null" in INDEX_HTML
    )


def test_routing_ui_has_independent_loading_and_results_per_tier() -> None:
    assert "tierFindLoading={local:false,mid:false,strong:false}" in INDEX_HTML
    assert "tierModelLists={local:[],mid:[],strong:[]}" in INDEX_HTML
    assert "if(tierFindLoading[name])return" in INDEX_HTML
    assert "tierModelLists[name]=models" in INDEX_HTML
    assert "tierFindLoading[name]=true" in INDEX_HTML
    assert "tierFindLoading[name]=false" in INDEX_HTML


def test_routing_ui_requires_base_url_before_search() -> None:
    assert "if(!baseUrl){statusNode.textContent='Укажите Base URL'" in INDEX_HTML


def test_routing_ui_preserves_current_model_after_search() -> None:
    assert "const select=$(name+'Model'),current=select.value" in INDEX_HTML
    assert "setTierModel(name,current)" in INDEX_HTML


def test_routing_ui_keeps_save_routing_button() -> None:
    assert 'id="saveRouting"' in INDEX_HTML
    assert "$('saveRouting').onclick=saveRouting" in INDEX_HTML


def test_routing_ui_has_per_tier_num_parallel_field() -> None:
    assert 'id="${name}Parallel" type="number" min="1" max="64"' in INDEX_HTML
    assert "Параллелизм (слотов)" in INDEX_HTML
    assert "$(name+'Parallel').value=data.num_parallel" in INDEX_HTML
    assert "num_parallel:Number($(name+'Parallel').value)||1" in INDEX_HTML


def test_routing_ui_has_per_tier_think_field() -> None:
    assert 'id="${name}Think"' in INDEX_HTML
    assert "Think (рассуждения)" in INDEX_HTML
    # 4 combined states: default / thinking+show / thinking+hide / off
    assert '<option value="show">включено с выводом</option>' in INDEX_HTML
    assert '<option value="hide">включено без вывода</option>' in INDEX_HTML
    assert '<option value="off">выключено</option>' in INDEX_HTML
    # fillTier maps (think, show_reasoning) back onto the dropdown value
    assert (
        "$(name+'Think').value=data.think===true?(data.show_reasoning===false?'hide':'show'):(data.think===false?'off':'')"
        in INDEX_HTML
    )
    # tierPayload sends both fields derived from the dropdown value
    assert "show_reasoning=(tv==='show')?true:(tv==='hide'?false:null)" in INDEX_HTML
    assert "think:think,show_reasoning:show_reasoning" in INDEX_HTML
    # think-level slider (low -> max) sits under the Think field
    assert 'id="${name}ThinkLevel" type="range" min="0" max="4"' in INDEX_HTML
    assert "Уровень размышлений" in INDEX_HTML
    assert "thinkLevels=['low','medium','high','max']" in INDEX_HTML
    assert "think_level:think_level" in INDEX_HTML


def test_routing_dashboard_renders_status_metrics() -> None:
    assert 'id="routerMetrics"' in INDEX_HTML
    assert "'/api/v2/router/status'" in INDEX_HTML
    assert "renderRouting(routing,inference)" in INDEX_HTML
    assert "success_rate_by_tier" in INDEX_HTML
    assert "cloud_tokens_saved" in INDEX_HTML
