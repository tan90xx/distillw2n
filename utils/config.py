import os
from omegaconf import OmegaConf

DEFAULT_DICT = {
    # Configuration ID
    'id': "null",
    # Training configuration
    'seed': 1234,
    'lr': 1e-6,
    'b1': 0.5,
    'b2': 0.9,
    'segment_length': 32270,
    # Model configuration
    'n_channels': 16,    
    'n_embed_dim': 256,
    'n_reencoder_layer': 1,
    'n_encoder_layer': 12,
    'sample_rate': 16000,
    'n_mels': 128,
    'n_fft': 1024,
    'win_length': 1024,
    'hop_length': 320,
    'trainable': True,
    'padding': 'same',
    # ROOT
    'pseudo_rate': 0.4,
    'datasets_root': '/data/ssd1/tianyi.tan/soundstream',
    'F0_model_path': './libs/JDC/bst.t7',
}   

class ConfigItem(dict):
    __slots__ = ()

    def __init__(self, config_dict=None):
        if config_dict is None:
            config_dict = dict()
        if isinstance(config_dict, ConfigItem):
            config_dict = config_dict.to_dict()
        assert isinstance(config_dict, dict)

        # Set attributes (not dict in ConfigItem)
        for key, value in config_dict.items():
            if isinstance(value, (list, tuple)):
                value = [ConfigItem(x) if isinstance(x, dict) else x for x in value]
            elif isinstance(value, dict):
                value = ConfigItem(value)
            elif isinstance(value, ConfigItem):
                value = ConfigItem(value.to_dict())
            elif isinstance(value, str) and value.lower() == 'none':
                value = None
            self[key] = value
    
    def __getattr__(self, item):
        try:
            return self[item]
        except KeyError:
            raise AttributeError(item)
    
    def __setattr__(self, name, value):
        self[name] = value
    
    def to_dict(self, recursive=True):
        conf_dict = {}
        for k, v in self.items():
            if isinstance(v, ConfigItem) and recursive:
                v = v.to_dict(recursive)
            conf_dict[k] = v
        return conf_dict

    def update(self, obj):
        assert isinstance(obj, (ConfigItem, dict))

        for k, v in obj.items():
            if k not in self or not isinstance(v, (ConfigItem, dict)):
                self[k] = v
            else:
                self[k].update(v)


class Config(ConfigItem):
    def __init__(self, yaml_object, dot_list=None):
        super().__init__(DEFAULT_DICT)
        
        # Check yaml_object
        if isinstance(yaml_object, str):
            assert os.path.isfile(yaml_object), yaml_object
            cfg = OmegaConf.load(yaml_object)
            if dot_list is not None:
                cfg_extra = OmegaConf.from_dotlist(dot_list)
                cfg = OmegaConf.merge(cfg, cfg_extra)
            yaml_object = OmegaConf.to_container(cfg, resolve=True)

        if isinstance(yaml_object, dict):
            yaml_object = ConfigItem(yaml_object)
        
        assert isinstance(yaml_object, ConfigItem)

        self.update(yaml_object)