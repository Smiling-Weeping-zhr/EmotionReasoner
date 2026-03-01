
import os
import typer
import os
# import ipdb
import torch
import requests
from omegaconf import OmegaConf
from typing import List
from enum import Enum

requests.packages.urllib3.disable_warnings()
from utils.registry import registry
from utils.tools import set_iterative_seed

app = typer.Typer(
    pretty_exceptions_enable=False,
)

class RunType(str, Enum):
    eval = "eval"
    train = "train"
    reason = "reason"

###############################################################################
#   main pipeline
@app.command("eval_qwen_omni")
def cmd_eval_qwen_omni(
    dataset: str = typer.Option(...),
    eval_type: str = typer.Option("CLS Metric"),
    run_type: RunType = typer.Option(RunType.eval),
) -> None:
    from scripts.eval_qwen_omni import eval_Qwen_omni
    from transformers import Qwen2_5OmniForConditionalGeneration, Qwen2_5OmniProcessor

    directory = '/root/autodl-tmp/EMER/experiment/eval_qwen_omni'
    cfg = OmegaConf.load(os.path.join(directory, 'hparams.yaml'))
    cfg.model.name = 'qwen_omni'
    cfg.dataset.name = dataset
    cfg.run.type = run_type
    model_path = cfg.model.ckpt
    model = Qwen2_5OmniForConditionalGeneration.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        attn_implementation="flash_attention_2",
    )
    model.disable_talker()
    model.eval()
    processor = Qwen2_5OmniProcessor.from_pretrained(model_path)
    if eval_type == "CLS Metric":
        eval_Qwen_omni(cfg=cfg, model=model, processor=processor)
    else:
        raise ValueError(f"We support this eval type {eval_type}")
    pass

@app.command("eval_qwen_omni_rl")
def cmd_eval_qwen_omni_rl(
    dataset: str = typer.Option(...),
    peft_ckpt: str = typer.Option(None, help="Path to the PEFT checkpoint for the Qwen Omni model."),
    eval_type: str = typer.Option("CLS Metric")
) -> None:
    from scripts.eval_qwen_omni import eval_Qwen_omni
    from transformers import Qwen2_5OmniForConditionalGeneration, Qwen2_5OmniProcessor
    from peft import PeftModel

    directory = '/root/autodl-tmp/EMER/experiment/eval_qwen_omni_rl'
    cfg = OmegaConf.load(os.path.join(directory, 'hparams.yaml'))
    cfg.dataset.name = dataset
    if peft_ckpt is not None:
        cfg.model.peft_ckpt = peft_ckpt
    cfg.model.name = 'qwen_omni_rl'
    model_path = cfg.model.ckpt
    peft_path = cfg.model.peft_ckpt
    print(f"Loading model from {model_path} and peft from {peft_path}")
    processor = Qwen2_5OmniProcessor.from_pretrained(model_path)
    
    base_model = Qwen2_5OmniForConditionalGeneration.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        attn_implementation="flash_attention_2",
    )
    base_model.disable_talker()
    model = PeftModel.from_pretrained(base_model, peft_path, is_local=True, strict=False)
    model.eval()
    if eval_type == 'CLS Metric':
        eval_Qwen_omni(cfg=cfg, model=model, processor=processor)
    else:
        raise ValueError(f"We didn't support the {eval_type} Metric...")
    pass
    
@app.command("calculate_metrics")
def cmd_calculate_metrics(model: str=typer.Option()) -> None:
    """ Calculate metrics for the specified dataset and model.
    """
    if model not in ['affectgpt', 'qwen-audio', 'emotion_llama', 'affectgpt_icl', 'valley', 'pandagpt', 'ola', 'qwen_omni', 'qwen_omni_icl', 'qwen_omni_rl', 'qwen2_5_vl', 'comparative_model']:
        raise ValueError("Unsupported model name. Please choose from 'affectgpt', 'qwen-audio', 'qwen_omni', or 'emotion-llama'.")
    directory = f'/root/autodl-tmp/EMER/experiment/eval_{model}' # the disposition of metrics
    cfg = OmegaConf.load(os.path.join(directory, 'hparams.yaml'))
    from scripts.calculate_metrics import main_zeroshot_scores
    datasets = cfg.dataset.eval_datasets
    cfg.model_name = model
    cfg.model.llama_model = "Qwen25" # unified llm to calculate the metrics
    for dataset in datasets:
        cfg.dataset.name = dataset
        input_dir = f"./experiment/eval_{model}/output/{dataset}"
        main_zeroshot_scores(input_dir=input_dir, dataset=dataset, cfg=cfg)

@app.command("calculate_overlap_score")
def cmd_calculate_overlap_score(
    model: str=typer.Option(),
    eval_type: str=typer.Option("OverLap"), 
    dataset: str = typer.Option("OV-MERD")) -> None:
    from scripts.overlap_score import calculate_overlab_clue_emotion, calculate_LACE_UTAI_MEER_metric
    directory = f'/root/autodl-tmp/EMER/experiment/eval_explanation'
    cfg = OmegaConf.load(os.path.join(directory, 'hparams.yaml'))
    cfg.dataset.name = dataset
    cfg.model_name = model
    if eval_type == "OverLap":
        calculate_overlab_clue_emotion(dir=directory, cfg=cfg)
    elif eval_type == "Ours":
        calculate_LACE_UTAI_MEER_metric(dir=directory, cfg=cfg)
    else:
        raise ValueError(f"We didn't support the eval_type {eval_type}")
    pass

def main() -> None:
    set_iterative_seed((1027*419), "scripts.shell.main")
    try:
        app()
    except Exception as exc:
        raise exc from exc
    finally:
        print("[[[ Program Completed...]]]")
    return