from juju_tools.utils.consts import *

from juju_tools.models.loading_utils import ModelLoader
from juju_tools.models.llama3.get_llama3_hf import save_model_weights
from juju_tools.models.llama3 import Tokenizer, ChatFormat

# from meta's llama repo
def sample_top_p(probs, p):
    probs_sort, probs_idx = torch.sort(probs, dim=-1, descending=True)
    probs_sum = torch.cumsum(probs_sort, dim=-1)
    mask = probs_sum - probs_sort > p
    probs_sort[mask] = 0.0
    probs_sort.div_(probs_sort.sum(dim=-1, keepdim=True))
    next_token = torch.multinomial(probs_sort, num_samples=1)
    next_token = torch.gather(probs_idx, -1, next_token)
    return next_token


@torch.inference_mode()
def run_local_model() -> None:
    with torch.no_grad():
        tok = Tokenizer("juju_tools/models/llama3/tokenizer.model")

        cf = ChatFormat(tok)

        llama_loader = ModelLoader(name="llama3-8b", weight_path="llama3_8b_weights", quant_type="nf4")

        config = llama_loader.config

        gen_len = 60

        message = [
            {
                "role": "system",
                "content": "You are a chatbot named Durk. Your responses should start with 'Durk:'. Please be as in-depth as possible."
            },
            {
                "role": "user",
                "content": "Write me a comprehensive essay on the second World War."
            }
        ]

        y = [cf.encode_dialog_prompt(message)]

        # y = [tok.encode(s=y, bos=True, eos=False)]
        tokens = torch.full((1, len(y[0]) + gen_len), fill_value=config.pad_tok_idx,
                            dtype=torch.long, device=config.device)

        for idx, t in enumerate(y):
            tokens[idx, :len(t)] = torch.tensor(t, dtype=torch.long, device=config.device)

        text_mask = tokens != config.pad_tok_idx

        stops = torch.tensor(list(tok.stop_tokens))

        prev_tok = 0

        for curr_tok in range(len(y[0]), gen_len + len(y[0])):

            logger.info(f"Current Token Being Generated {curr_tok} | {tokens[..., prev_tok:curr_tok]}")

            logits = llama_loader.get_model()(input=tokens[..., prev_tok:curr_tok], shift=curr_tok)

            # next_tok = torch.argmax(logits[:, -1], dim=-1).reshape(-1)
            probs = torch.softmax(logits[:, -1] / .6, dim=-1)
            next_tok = sample_top_p(probs, .9)  # torch.argmax(logits[:, -1], dim=-1).reshape(-1) #

            next_tok = torch.where(text_mask[:, curr_tok], tokens[:, curr_tok], next_tok)

            tokens[:, curr_tok] = next_tok

            logger.info(f"TOKEN: {next_tok}")

            gen_toks = tokens[0, :curr_tok + 1].detach().cpu().numpy().tolist()

            logger.info(f"{tok.decode(gen_toks)}")
            prev_tok = curr_tok

            if torch.isin(next_tok.cpu(), stops):
                break


def get_hf_weights() -> None:
    logger.info(f"{os.getcwd()}")
    hf_path = input("Enter the local hugging face path: ")
    if hf_path == "":
        save_model_weights(save_area=str(os.getcwd()) + "/")
    else:
        save_model_weights(hf_path, str(os.getcwd()) + "/")