from juju_tools.utils import *
from juju_tools.utils import Config

from juju_tools.utils.layers import RMSNorm, LinearNFK
import bitsandbytes as bnb


class BaseModel(Module):

    def __init__(self,
                 config: Config,
                 load_all: bool = True,
                 map: Optional[Dict] = None,
                 **kwargs):
        super().__init__(**kwargs)

        self.config = config

        self.blocks_to_load = 0
        self.load_all = load_all

        self.layers = nn.ModuleList([])

        self.stages_to_load = []

        self.model_map = map

    def is_quant(self) -> None:
        pass

    @torch.no_grad()
    def to_quant_layers(self,
                        module_types: List[str],
                        layer_types: List[str],
                        weight_path: str = None,
                        quant_type: str = "nf4", ) -> None:
        logger.info("Quantizing Model Layers... ")

        # torch.set_default_dtype(torch.bfloat16)

        assert self.loaded(), "Model layers must be loaded to a device first!"

        torch.cuda.empty_cache()

        logger.info(str(module_types))
        logger.info(str(layer_types))
        for name, module in self.named_children():
            torch.cuda.empty_cache()

            if not (name in module_types): continue

            loaded_module = getattr(self, str(name))

            if type(loaded_module) == nn.ModuleList:

                for idx in range(len(loaded_module)):

                    logger.info(f"Quantizing block: {idx} ")
                    for layer_type in self.model_map[str(name)]:
                        if not (layer_type in layer_types): continue

                        loaded_layer = getattr(self.layers[idx], layer_type)

                        for sub_layer in self.model_map[str(name)][layer_type]:

                            torch.cuda.empty_cache()

                            if sub_layer == "" or sub_layer is None: continue

                            loaded_sub_layer = getattr(loaded_layer, sub_layer)
                            weight = None

                            if weight_path is not None:
                                weight = torch.load(f"{weight_path}/{layer_type}.{sub_layer}.{idx}.pth", weights_only=True).to(
                                    self.config.device).to(torch.bfloat16)

                            if isinstance(loaded_sub_layer, nn.Linear):
                                shape = loaded_sub_layer.weight.shape

                                # TODO bias maybe...
                                if quant_type == "nf4":
                                    if weight is None:
                                        setattr(loaded_layer, sub_layer,
                                                LinearNFK(in_features=shape[1], out_features=shape[0],
                                                          device=self.config.device))
                                    else:
                                        setattr(loaded_layer, sub_layer,
                                                LinearNFK(weight=weight, device=self.config.device))
                                        # param = bnb.nn.Params4bit(weight.to(self.config.device), requires_grad=False, quant_type="nf4")
                                        # mod = bnb.nn.Linear4bit(shape[1], shape[0], bias=False)
                                        # mod.weight = param
                                        #
                                        # mod.to(self.config.device)
                                        # setattr(loaded_layer, sub_layer, mod)


                            elif isinstance(loaded_sub_layer, nn.Parameter):
                                if weight is None:
                                    weight = nn.Parameter(torch.randn(self.config.dim, dtype=torch.bfloat16,
                                                                      device=self.config.device), requires_grad=False)
                                else:
                                    weight = nn.Parameter(weight)

                                setattr(loaded_layer, sub_layer, weight)

            elif type(loaded_module) == nn.Embedding:

                shape = loaded_module.weight.shape
                new_module = nn.Embedding(shape[0], shape[1], dtype=torch.bfloat16, padding_idx=self.config.pad_tok_idx,
                                          device=self.config.device)  # padding_idx=self.config.pad_tok_idx,

                if weight_path is not None:
                    weight = torch.load(f"{weight_path}/{name}.pth").to(self.config.device).to(torch.bfloat16)
                else:
                    weight = new_module.weight

                new_module.weight.data = weight

                setattr(self, name, new_module)
                logger.info(f"Loaded {name}")

            elif type(loaded_module) == nn.Linear:
                shape = loaded_module.weight.shape
                new_module = nn.Linear(shape[1], shape[0], bias=self.config.bias,
                                       device=self.config.device)

                if weight_path is not None:
                    weight = torch.load(f"{weight_path}/{name}.pth", map_location = torch.device("cpu")).to(torch.bfloat16) # to(self.config.device).
                    new_module.weight.data = weight
                else:
                    new_module.weight.data.normal_(mean=0.1, std=.02).to(self.config.device).to(torch.bfloat16)

                setattr(self, name, new_module)
                logger.info(f"Loaded {name}")

            elif type(loaded_module) == RMSNorm:

                if weight_path is not None:
                    weight = torch.load(f"{weight_path}/{name}.pth").to(self.config.device).to(torch.bfloat16)
                    weight = nn.Parameter(weight)
                else:
                    shape = loaded_module.gate.shape
                    weight = nn.Parameter(torch.randn(shape, device=self.config.device, dtype=torch.bfloat16))

                setattr(loaded_module, self.model_map[str(name)], weight)
                logger.info(f"Loaded {name}")

        return
        # exit()

        for idx in range(len(self.layers)):
            logger.info(f" Quantizing layer: {idx} ")

            for layer_type in self.model_map["layers"]:
                if not (layer_type in layer_types): continue

                loaded_layer = getattr(self.layers[idx], layer_type)

                for sub_layer in self.model_map["layers"][layer_type]:

                    if sub_layer == "" or sub_layer == None: continue

                    torch.cuda.empty_cache()

                    weight = None

                    if preloaded_weight != None:
                        weight = preloaded_weight[f"{idx}.{sub_layer}"].to(self.config.device)

                    # logger.info(f"{layer_type} {sub_layer}")

                    loaded_sub_layer = getattr(loaded_layer, sub_layer)

                    if isinstance(loaded_sub_layer, nn.Linear):
                        shape = loaded_sub_layer.weight.shape

                        if weight is None:
                            weight = torch.empty(shape).normal_(mean=0., std=.02).to(torch.bfloat16).to(
                                self.config.device)
                            # TODO bias maybe...

                        if quant_type == "nf4":
                            setattr(loaded_layer, sub_layer, LinearNFK(weight=weight, device=self.config.device))

                    elif isinstance(loaded_sub_layer, nn.Parameter):
                        if weight is None:
                            weight = nn.Parameter(
                                torch.randn(self.config.dim, dtype=torch.float32, device=self.config.device),
                                requires_grad=False)
                        setattr(loaded_layer, sub_layer, weight)

        logger.info(self)

    def _init_weights(self) -> None:
        assert self.loaded(), "Cannot pass forward - model has not been loaded!"

        for idx, module in enumerate(self.modules()):
            if isinstance(module, nn.Linear):
                module.weight.data.normal_(mean=0., std=.02)
                if module.bias != None:
                    module.bias.data.zero_()

            elif isinstance(module, nn.Embedding):
                module.weight.data.normal_(mean=0., std=.02)
                if module.padding_idx != None:
                    module.weight.data[module.padding_idx].zero_()

            # print(f"Loaded weights for module {idx}")

    def _add_block(self) -> None:
        self.blocks_to_load += 1

    def loaded(self) -> bool:
        return self.blocks_to_load <= 0

    def load(self, device: Optional[Tensor] = None) -> None:
        self.blocks_to_load -= 1

    @torch.no_grad()
    def generate() -> Tuple[Tensor, ...]:
        pass

# param = bnb.nn.Params4bit(weight.to(self.config.device), requires_grad=False, quant_type="nf4")
# mod = bnb.nn.Linear4bit(shape[1], shape[0], bias=False)
# mod.weight = param

# mod.to(self.config.device)
# setattr(loaded_layer, sub_layer, mod)
