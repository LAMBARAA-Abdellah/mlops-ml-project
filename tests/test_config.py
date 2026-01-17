import yaml

def test_config_load():
    cfg = yaml.safe_load(open("config/train.yaml"))
    assert "data" in cfg
    assert "model" in cfg
    assert "split" in cfg
