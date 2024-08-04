from juju_tools.models.modeling_utils import BaseModel
from juju_tools.utils import *
from juju_tools.models.llama3 import LLaMaAutoRegressive

model_names = {
    "llama3-8b":
        {
            "layers":
                {
                    "gq_attn": (
                        "proj_q", "proj_k", "proj_v", "proj_o"),
                    "ffn": (
                        "proj_up", "proj_down", "gate"),
                    "attn_prenorm": (
                        "gate", ""),
                    "ffn_prenorm": (
                        "gate", ""),
                },
            "embed": (""),
            "cls_head": (""),
            "pre_norm": "gate",
        }
}

quant_method = ("nf4")


#TODO define an inference mode
#TODO method for loading in Llama weights and converting to quant. version
#TODO auto way of mapping elements to devices ex. inp. layers to cpu and pass thru layers on cuda

class ModelLoader:

    def __init__(self,
                 name: Optional[str] = None,
                 model: Optional[BaseModel] = None,
                 config: Optional[Config] = None,
                 quant_type: Optional[str] = None,
                 load_to_mem: bool = False,
                 default_device: str = "meta",
                 weight_path: str = None,
                 mode: str = "inference") -> None:

        assert (name != None) ^ (model != None), "Need to provide either a model type or model (not both!)"

        self.config: Config = None
        self.base_model: BaseModel = None

        self.layer_names: List[str] = []

        if name != None:

            assert name.lower() in model_names, "Model name does not exist!"

            if name.lower() == "llama3-8b":
                self.config = LLaMaConfig(
                    dim=4096,
                    attn_heads=32,
                    kv_heads=8,
                    rope_base=5e5,
                    vocab_size=128256,
                    n_layers=32, ) if config == None else config
                self.base_model = LLaMaAutoRegressive(config=self.config,
                                                      load_all=load_to_mem, map=model_names[name])
                self.layer_names = model_names[name].keys()

            else:
                self.base_model = model
            pass

        pass

        assert self.base_model != None, "Something went wrong, base model is None!"

        if not self.base_model.loaded():
            self.base_model.load(default_device)
            logger.info(f" Model has been loaded... ")
        pass

        if quant_type != None:
            assert quant_type in quant_method, "Quantization method not supported!"

            self.base_model.to_quant_layers(module_types=self.config.quant_modules,
                                            layer_types=self.config.quant_layers,
                                            weight_path=weight_path,
                                            quant_type=quant_type)
        pass

    def memory_footprint(self) -> List:
        for layer in self.layer_names:
            pass

    def get_config(self) -> Config:
        return self.config

    def get_model(self) -> BaseModel:
        return self.base_model
