import os
import torch
from peft import LoraConfig, get_peft_model
import ast
import re
import json
import dataclasses
from transformers import (
    AutoProcessor,
    BitsAndBytesConfig, 
    HfArgumentParser, 
)
from model.load_model import get_qwen_vl_generation_backbone, load_qwen_vl_generation_model
from trainer import QwenSFTTrainer
from dataset import make_supervised_data_module
from params import DataArguments, ModelArguments, TrainingArguments
from train.medical_bleu_rouge_eval_callback import MedicalBleuRougeEvalCallback
from train.train_utils import get_peft_state_maybe_zero_3, get_peft_state_non_lora_maybe_zero_3, safe_save_model_for_hf_trainer
import pathlib

local_rank = None

def rank0_print(*args):
    if local_rank == 0 or local_rank == '0' or local_rank is None:
        print(*args)

def _as_jsonable(obj):
    if dataclasses.is_dataclass(obj):
        return dataclasses.asdict(obj)
    if hasattr(obj, "to_dict"):
        try:
            return obj.to_dict()
        except Exception:
            pass
    if hasattr(obj, "__dict__"):
        return dict(obj.__dict__)
    return str(obj)

def _rank0_dump_training_args(model_args, data_args, training_args) -> None:
    payload = {
        "model_args": _as_jsonable(model_args),
        "data_args": _as_jsonable(data_args),
        "training_args": _as_jsonable(training_args),
    }
    rank0_print("==== Parsed training arguments (rank0) ====")
    rank0_print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))
    rank0_print("==== End arguments ====")

def _rank0_print_trainable_params(model) -> None:
    total_params = 0
    trainable_params = 0
    for p in model.parameters():
        n = p.numel()
        total_params += n
        if p.requires_grad:
            trainable_params += n
    pct = (trainable_params / total_params * 100.0) if total_params else 0.0
    rank0_print(
        f"==== Trainable parameters (rank0) ====\n"
        f"total_params: {total_params:,}\n"
        f"trainable_params: {trainable_params:,}\n"
        f"trainable_percent: {pct:.4f}%\n"
        f"==== End trainable parameters ===="
    )


def _checkpoint_step(path: pathlib.Path) -> int:
    try:
        return int(path.name.split("-")[-1])
    except ValueError:
        return -1


def _matches_save_steps(checkpoint_dir: pathlib.Path, desired_save_steps: int) -> bool:
    """
    When resuming, Transformers can keep using some checkpoint state fields (via trainer_state.json).
    If an old run used save_steps=1, resuming will effectively keep saving every step and slow training.
    We treat such checkpoints as incompatible with the current run's save_steps.
    """
    trainer_state = checkpoint_dir / "trainer_state.json"
    if not trainer_state.exists():
        return False
    try:
        payload = json.load(open(trainer_state, "r", encoding="utf-8"))
        ckpt_save_steps = payload.get("save_steps", None)
        if ckpt_save_steps is None:
            return True
        ckpt_save_steps_int = int(float(ckpt_save_steps))
        return ckpt_save_steps_int == int(desired_save_steps)
    except Exception:
        # If parsing fails, be conservative: don't resume from it.
        return False


def _get_last_complete_checkpoint(output_dir: str, desired_save_steps: int):
    checkpoints = sorted(
        pathlib.Path(output_dir).glob("checkpoint-*"),
        key=_checkpoint_step,
        reverse=True,
    )
    for checkpoint in checkpoints:
        trainer_state = checkpoint / "trainer_state.json"
        if not trainer_state.exists():
            rank0_print(f"Skipping incomplete checkpoint without trainer_state.json: {checkpoint}")
            continue
        if not _matches_save_steps(checkpoint, desired_save_steps):
            rank0_print(
                f"Skipping checkpoint with mismatched save_steps (want {desired_save_steps}): {checkpoint}"
            )
            continue
        return str(checkpoint)
    return None


def _grounding_label(text: str):
    text = (text or "").lower()
    if re.search(r"不一致|不匹配|不是|mismatch|not consistent|incorrect", text):
        return "mismatch"
    if re.search(r"一致|匹配|是|match|consistent|correct", text):
        return "match"
    return None


def sample_eval_metrics(eval_pred):
    predictions = eval_pred.predictions
    references = eval_pred.references
    label_total = 0
    label_correct = 0
    for pred, ref in zip(predictions, references):
        ref_label = _grounding_label(ref)
        if ref_label is None:
            continue
        label_total += 1
        label_correct += int(_grounding_label(pred) == ref_label)
    metrics = {"sample_eval_count": len(predictions)}
    if label_total:
        metrics["grounding_label_accuracy"] = label_correct / label_total
    return metrics

def find_target_linear_names(model, num_lora_modules=-1, lora_namespan_exclude=[], verbose=True):
    linear_cls = torch.nn.modules.Linear
    embedding_cls = torch.nn.modules.Embedding
    lora_module_names = []

    for name, module in model.named_modules():
        if any(ex_keyword in name for ex_keyword in lora_namespan_exclude):
            continue
        if isinstance(module, (linear_cls, embedding_cls)):
            lora_module_names.append(name)
    
    if num_lora_modules > 0:
        lora_module_names = lora_module_names[-num_lora_modules:]
    if verbose:
        rank0_print(f"Found {len(lora_module_names)} lora modules: {lora_module_names}")
    return lora_module_names

def set_requires_grad(parameters, requires_grad):
    for p in parameters:
        p.requires_grad = requires_grad

def configure_vision_tower(model, training_args, compute_dtype, device):
    backbone = get_qwen_vl_generation_backbone(model)
    vision_tower = backbone.visual
    vision_tower.to(dtype=compute_dtype, device=device)

    vision_model_params = backbone.visual.parameters()
    set_requires_grad(vision_model_params, not training_args.freeze_vision_tower)
    
    # Handle merger specifically
    merger_params = backbone.visual.merger.parameters()
    set_requires_grad(merger_params, not training_args.freeze_merger)

    if hasattr(backbone.visual, "deepstack_merger_list"):
        deepstack_merger_list_params = backbone.visual.deepstack_merger_list.parameters()
        set_requires_grad(deepstack_merger_list_params, not training_args.freeze_merger)

def configure_llm(model, training_args):
    backbone = get_qwen_vl_generation_backbone(model)
    lm_head = model.lm_head.parameters()
    set_requires_grad(lm_head, not training_args.freeze_llm)

    llm_params = backbone.language_model.parameters()
    set_requires_grad(llm_params, not training_args.freeze_llm)

def unfreeze_topk_layers(model, k_llm: int = 0, k_vis: int = 0):
    backbone = get_qwen_vl_generation_backbone(model)

    if k_llm and hasattr(backbone, "language_model") and hasattr(backbone.language_model, "layers"):
        for layer in backbone.language_model.layers[-k_llm:]:
            for p in layer.parameters():
                p.requires_grad = True

    if k_vis and hasattr(backbone, "visual") and hasattr(backbone.visual, "blocks"):
        for blk in backbone.visual.blocks[-k_vis:]:
            for p in blk.parameters():
                p.requires_grad = True


def train():
    global local_rank

    parser = HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments))
    
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    
    if data_args.nframes is not None and data_args.fps is not None:
        raise ValueError("You cannot set both `nframes` and `fps` at the same time. Please set only one of them.")

    if training_args.lora_enable and not training_args.freeze_llm:
        raise ValueError("If `lora_enable` is True, `freeze_llm` must also be True.")

    if not training_args.lora_enable:
        assert not training_args.vision_lora, \
            "Error: training_args.lora_enable is not enabled, but training_args.vision_lora is enabled."
        
    if training_args.vision_lora and not training_args.freeze_vision_tower:
        raise ValueError("If `vision_lora` is True, `freeze_vision_tower` must also be True.")

    else:
        if training_args.lora_namespan_exclude is not None:
            training_args.lora_namespan_exclude = ast.literal_eval(training_args.lora_namespan_exclude)
        else:
            training_args.lora_namespan_exclude = []

        if not training_args.vision_lora:
            training_args.lora_namespan_exclude += ["visual"]

    local_rank = training_args.local_rank
    _rank0_dump_training_args(model_args, data_args, training_args)
    compute_dtype = (torch.float16 if training_args.fp16 else (torch.bfloat16 if training_args.bf16 else torch.float32))

    bnb_model_from_pretrained_args = {}
    if training_args.bits in [4,8]:
        bnb_model_from_pretrained_args.update(dict(
            device_map={"":training_args.device},
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=training_args.bits==4,
                load_in_8bit=training_args.bits==8,
                llm_int8_skip_modules=["visual", "lm_head"],
                llm_int8_threshold=6.0,
                llm_int8_has_fp16_weight=False,
                bnb_4bit_compute_dtype=compute_dtype,
                bnb_4bit_use_double_quant=training_args.double_quant,
                bnb_4bit_quant_type=training_args.quant_type,
            )
        ))

    model = load_qwen_vl_generation_model(
        model_args.model_id,
        dtype=compute_dtype,
        attn_implementation="sdpa" if training_args.disable_flash_attn2 else "flash_attention_2",
        **bnb_model_from_pretrained_args,
    )
    if training_args.use_liger_kernel and model.config.model_type in {"qwen3_5", "qwen3_5_moe"}:
        rank0_print(f"Disabling Liger kernel for unsupported model_type: {model.config.model_type}")
        training_args.use_liger_kernel = False
        if hasattr(training_args, "liger_kernel_config"):
            training_args.liger_kernel_config = None

    model.config.use_cache = False
    model_to_configure = model
    configure_llm(model_to_configure, training_args)
    configure_vision_tower(model_to_configure, training_args, compute_dtype, training_args.device)

    unfreeze_topk_layers(
        model_to_configure,
        k_llm=getattr(training_args, "unfreeze_topk_llm", 0),
        k_vis=getattr(training_args, "unfreeze_topk_vision", 0),
    )

    if training_args.gradient_checkpointing:
        if training_args.vision_lora:
            training_args.gradient_checkpointing_kwargs = {"use_reentrant": False}
        else:
            training_args.gradient_checkpointing_kwargs = {"use_reentrant": True}
        
        model.enable_input_require_grads()

    if training_args.bits in [4,8]:
        model.config.dtype = (torch.float32 if training_args.fp16 else (torch.bfloat16 if training_args.bf16 else torch.float32))
        from peft import prepare_model_for_kbit_training
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=training_args.gradient_checkpointing, gradient_checkpointing_kwargs=training_args.gradient_checkpointing_kwargs)
    
    if training_args.lora_enable:
        lora_namespan_exclude = training_args.lora_namespan_exclude
        peft_config = LoraConfig(
            r=training_args.lora_rank,
            lora_alpha=training_args.lora_alpha,
            target_modules=find_target_linear_names(model, lora_namespan_exclude=lora_namespan_exclude, num_lora_modules=training_args.num_lora_modules),
            lora_dropout=training_args.lora_dropout,
            bias=training_args.lora_bias
        )
        if training_args.bits == 16:
            if training_args.bf16:
                model.to(torch.bfloat16)
            if training_args.fp16:
                model.to(torch.float16)
        rank0_print("Adding LoRA to the model...")
        model = get_peft_model(model, peft_config)

        # Peft maodel makes vision tower and merger freezed again.
        # Configuring fuction could be called here, but sometimes it does not work properly.
        # So I just made it this way.
        # Need to be fixed in the future.

        if not training_args.freeze_vision_tower:
            for name, param in model.named_parameters():
                if "visual" in name:
                    param.requires_grad = True

        if not training_args.freeze_merger:
            for name, param in model.named_parameters():
                if "merger" in name:
                    param.requires_grad = True

    _rank0_print_trainable_params(model)

    processor = AutoProcessor.from_pretrained(model_args.model_id)

    # model.config.tokenizer_model_max_length = processor.tokenizer.model_max_length

    if training_args.bits in [4, 8]:
        from peft.tuners.lora import LoraLayer
        for name, module in model.named_modules():
            if isinstance(module, LoraLayer):
                if training_args.bf16:
                    module = module.to(torch.bfloat16)
            if 'norm' in name:
                module = module.to(torch.float32)
            
            if 'lm_head' in name or 'embed_token' in name:
                if hasattr(module, 'weight'):
                    if training_args.bf16 and module.weight.dtype == torch.float32:
                        module = module.to(torch.bfloat16)

    data_module = make_supervised_data_module(model_id=model_args.model_id,
                                              processor=processor,
                                              data_args=data_args)

    callbacks = []
    if getattr(training_args, "medical_eval_bleu_steps", 0) and training_args.medical_eval_bleu_steps > 0:
        repo_root = pathlib.Path(__file__).resolve().parents[2]
        eval_script = training_args.medical_eval_script or str(
            repo_root / "scripts" / "eval_medical_bleu_rouge.sh"
        )
        val_root = training_args.medical_eval_validation_root or os.path.join(
            training_args.output_dir, "validation"
        )
        data_json = (
            training_args.medical_eval_data_path
            or data_args.eval_path
            or data_args.data_path
        )
        img_folder = (
            training_args.medical_eval_image_folder
            or data_args.eval_image_folder
            or data_args.image_folder
        )
        base_m = training_args.medical_eval_base_model or model_args.model_id
        if not data_json:
            raise ValueError(
                "启用 medical_eval_bleu_steps 时需要评测数据 JSON：请设置 --data_path / --eval_path 或 --medical_eval_data_path"
            )
        if not img_folder:
            raise ValueError(
                "启用 medical_eval_bleu_steps 时需要图像目录：请设置 --image_folder / --eval_image_folder 或 --medical_eval_image_folder"
            )
        callbacks.append(
            MedicalBleuRougeEvalCallback(
                eval_script=eval_script,
                validation_root=val_root,
                base_model=base_m,
                data_path=data_json,
                image_folder=img_folder,
                eval_every_steps=training_args.medical_eval_bleu_steps,
                keep_best_n=training_args.medical_eval_keep_best_n,
                eval_batch_size=training_args.medical_eval_batch_size,
                max_new_tokens=training_args.medical_eval_max_new_tokens,
                cuda_visible_devices=training_args.medical_eval_cuda_visible_devices,
                sort_key=training_args.medical_eval_sort_key,
            )
        )
        rank0_print(
            f"Medical BLEU/ROUGE eval: every {training_args.medical_eval_bleu_steps} steps on save, "
            f"artifacts under {val_root} (keep best {training_args.medical_eval_keep_best_n} runs/)"
        )

    trainer = QwenSFTTrainer(
        model=model,
        processing_class=processor,
        args=training_args,
        callbacks=callbacks,
        compute_metrics=sample_eval_metrics if data_args.eval_path is not None else None,
        **data_module
    )

    last_checkpoint = _get_last_complete_checkpoint(
        training_args.output_dir,
        desired_save_steps=int(getattr(training_args, "save_steps", 0) or 0),
    )
    if last_checkpoint is not None:
        rank0_print(f"Resuming from checkpoint: {last_checkpoint}")
        trainer.train(resume_from_checkpoint=last_checkpoint)
    else:
        if list(pathlib.Path(training_args.output_dir).glob("checkpoint-*")):
            rank0_print("No complete checkpoint found. Starting a fresh training run.")
        trainer.train()

    trainer.save_state()

    model.config.use_cache = True
    
    if training_args.lora_enable:
        state_dict = get_peft_state_maybe_zero_3(
            model.named_parameters(), training_args.lora_bias
        )

        non_lora_state_dict = get_peft_state_non_lora_maybe_zero_3(
            model.named_parameters(), require_grad_only=True
        )

        if local_rank == 0 or local_rank == -1:
            model.config.save_pretrained(training_args.output_dir)
            model.save_pretrained(training_args.output_dir, state_dict=state_dict)
            processor.save_pretrained(training_args.output_dir)
            torch.save(non_lora_state_dict, os.path.join(training_args.output_dir, "non_lora_state_dict.bin"))
    else:
        safe_save_model_for_hf_trainer(trainer, output_dir=training_args.output_dir)



if __name__ == "__main__":
    train()
