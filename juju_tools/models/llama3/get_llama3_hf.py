from juju_tools.utils import *

from safetensors import safe_open

remap = {
    "embed_tokens": "embed",
    "mlp": {
        "gate_proj": "ffn.proj_up",
        "down_proj": "ffn.proj_down",
        "up_proj": "ffn.gate"
    },
    "self_attn": {
        "q_proj": "gq_attn.proj_q",
        "k_proj": "gq_attn.proj_k",
        "v_proj": "gq_attn.proj_v",
        "o_proj": "gq_attn.proj_o"
    },
    "post_attention_layernorm": "ffn_prenorm.gate",
    "input_layernorm": "attn_prenorm.gate",
    "lm_head": "cls_head",
    "norm": "pre_norm"
}


def save_model_weights(hf_path: str = "/home/v18/.cache/", save_area: str = ""):
    for idx in range(4):
        with safe_open(
                f"{hf_path}huggingface/hub/models--meta-llama--Meta-Llama-3-8B-Instruct/snapshots/e1945c40cd546c78e41f1151f4db032b271faeaa/model-0000{idx + 1}-of-00004.safetensors",
                framework="pt", device=0) as f:

            for idx, k in enumerate(f.keys()):
                # logger.info(f"{k}")
                k_sp = k.split(".")
                name_to_save = ""
                # logger.info(f"{k_sp}")

                if len(k_sp) == 3:
                    name_to_save = remap[k_sp[1]]
                elif len(k_sp) == 2:
                    name_to_save = remap[k_sp[0]]
                elif len(k_sp) == 6 or len(k_sp) == 5:
                    mod_name = k.split(".")[3]
                    if mod_name in ("mlp", "self_attn"):
                        name_to_save = remap[mod_name][k_sp[4]] + '.' + k_sp[2]
                    elif mod_name in ("input_layernorm", "post_attention_layernorm"):
                        name_to_save = remap[mod_name] + '.' + k_sp[2]

                logger.info(f"{k_sp} ===> {name_to_save}")

                o = f.get_tensor(k)
                torch.save(o, f"{save_area}llama3_8b_weights/{name_to_save}.pth")
