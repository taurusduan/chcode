import json
from pathlib import Path
from unittest.mock import patch, MagicMock, AsyncMock

import pytest

from chcode.config import (
    load_model_json,
    save_model_json,
    load_workplace,
    save_workplace,
    load_tavily_api_key,
    save_tavily_api_key,
    ensure_config_dir,
)


class TestLoadModelJson:
    def test_no_file(self, tmp_path: Path, monkeypatch):
        import chcode.config as mod
        mod._model_json_cache = None
        monkeypatch.setattr(mod, "MODEL_JSON", tmp_path / "model.json")
        assert load_model_json() == {}

    def test_valid_file(self, tmp_path: Path, monkeypatch):
        import chcode.config as mod
        mod._model_json_cache = None
        f = tmp_path / "model.json"
        f.write_text(json.dumps({"default": {"model": "gpt-4o"}}))
        monkeypatch.setattr(mod, "MODEL_JSON", f)
        data = load_model_json()
        assert data["default"]["model"] == "gpt-4o"

    def test_mtime_cache(self, tmp_path: Path, monkeypatch):
        import chcode.config as mod
        mod._model_json_cache = None
        f = tmp_path / "model.json"
        f.write_text(json.dumps({"default": {"model": "a"}}))
        monkeypatch.setattr(mod, "MODEL_JSON", f)
        d1 = load_model_json()
        d2 = load_model_json()
        assert d1 is d2

    def test_invalid_json(self, tmp_path: Path, monkeypatch):
        import chcode.config as mod
        mod._model_json_cache = None
        f = tmp_path / "model.json"
        f.write_text("not json{{{")
        monkeypatch.setattr(mod, "MODEL_JSON", f)
        assert load_model_json() == {}


class TestSaveModelJson:
    def test_saves_and_invalidates_cache(self, tmp_path: Path, monkeypatch):
        import chcode.config as mod
        mod._model_json_cache = None
        f = tmp_path / "model.json"
        monkeypatch.setattr(mod, "MODEL_JSON", f)
        save_model_json({"default": {"model": "test"}})
        assert f.exists()
        data = json.loads(f.read_text())
        assert data["default"]["model"] == "test"
        assert mod._model_json_cache is None


class TestLoadWorkplace:
    def test_no_file_v2(self, tmp_path: Path, monkeypatch):
        import chcode.config as mod
        monkeypatch.setattr(mod, "SETTING_JSON", tmp_path / "nope.json")
        assert load_workplace() is None

    def test_valid(self, tmp_path: Path, monkeypatch):
        import chcode.config as mod
        f = tmp_path / "settings.json"
        f.write_text(json.dumps({"workplace_path": str(tmp_path)}))
        monkeypatch.setattr(mod, "SETTING_JSON", f)
        assert load_workplace() == tmp_path


class TestSaveWorkplace:
    def test_saves(self, tmp_path: Path, monkeypatch):
        import chcode.config as mod
        f = tmp_path / "settings.json"
        monkeypatch.setattr(mod, "SETTING_JSON", f)
        monkeypatch.setattr(mod, "CONFIG_DIR", tmp_path)
        save_workplace(tmp_path / "myproject")
        data = json.loads(f.read_text())
        assert "myproject" in data["workplace_path"]


class TestLoadSaveTavilyApiKey:
    def test_no_file_v3(self, tmp_path: Path, monkeypatch):
        import chcode.config as mod
        monkeypatch.delenv("TAVILY_API_KEY", raising=False)
        monkeypatch.setattr(mod, "SETTING_JSON", tmp_path / "nope.json")
        assert load_tavily_api_key() == ""

    def test_save_and_load(self, tmp_path: Path, monkeypatch):
        import chcode.config as mod
        f = tmp_path / "settings.json"
        f.write_text("{}")
        monkeypatch.setattr(mod, "SETTING_JSON", f)
        monkeypatch.setattr(mod, "CONFIG_DIR", tmp_path)
        save_tavily_api_key("tvly-test123")
        key = load_tavily_api_key()
        assert key == "tvly-test123"


class TestEnsureConfigDir:
    def test_creates_dir(self, tmp_path: Path, monkeypatch):
        import chcode.config as mod
        d = tmp_path / "newconfig"
        monkeypatch.setattr(mod, "CONFIG_DIR", d)
        result = ensure_config_dir()
        assert d.exists()
        assert result == d
