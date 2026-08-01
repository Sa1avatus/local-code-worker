from local_code_worker.web_app import INDEX_HTML


def test_model_picker_uses_native_select_and_separate_pull_input() -> None:
    assert '<select id="model"></select>' in INDEX_HTML
    assert '<input id="pullModel"' in INDEX_HTML
    assert '<datalist id="models">' not in INDEX_HTML
    assert 'list="models"' not in INDEX_HTML


def test_discovered_models_are_safe_options_and_preserve_selection() -> None:
    assert "document.createElement('option')" in INDEX_HTML
    assert "option.textContent=name" in INDEX_HTML
    assert "model.replaceChildren()" in INDEX_HTML
    assert "if(preferredModel&&!names.includes(preferredModel))" in INDEX_HTML
    assert "await models(data.model||'')" in INDEX_HTML


def test_pull_prefers_manual_name_and_selects_downloaded_model() -> None:
    assert "pullModel.value.trim()||model.value" in INDEX_HTML
    assert "await models(requestedModel)" in INDEX_HTML
    assert "pullModel.value=''" in INDEX_HTML
