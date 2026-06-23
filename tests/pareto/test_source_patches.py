from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def test_cityflow_reset_supports_deterministic_seed():
    source = (ROOT / "utils" / "cityflow_env.py").read_text()
    assert "CITYFLOW_SEED" in source
    assert 'cityflow_config["seed"] = int(cityflow_seed)' in source
    assert '"seed": int(np.random.randint(0, 100))' not in source


def test_chatgpt_key_comes_from_environment():
    source = (ROOT / "models" / "chatgpt.py").read_text()
    assert "os.environ.get(\"OPENAI_API_KEY\"" in source
    assert "YOUR_KEY_HERE" not in source
