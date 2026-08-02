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
    assert "renderMetrics(system)" in INDEX_HTML
    assert "linear-gradient(90deg,var(--accent),var(--ok))" in INDEX_HTML


def test_context_length_is_loaded_saved_and_requires_model_reload() -> None:
    assert "contextLength.value=data.context_length" in INDEX_HTML
    assert "context_length:Number(contextLength.value)" in INDEX_HTML
    assert "ollama stop" in INDEX_HTML
