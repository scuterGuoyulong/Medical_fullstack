from dataclasses import dataclass, field
from typing import Optional

try:
    from accelerate.utils import ParallelismConfig as _PC
except Exception:
    class _PC:
        pass

import transformers.training_args as _ta
if not hasattr(_ta, "ParallelismConfig"):
    _ta.ParallelismConfig = _PC

from transformers import TrainingArguments as HFTrainingArguments
from trl import DPOConfig as DPOConfigTRL
from trl import GRPOConfig as GRPOConfigTRL


@dataclass
class ModelArguments:
    model_id: Optional[str] = field(default="Qwen/Qwen2-VL-7B-Instruct")


@dataclass
class CLSArguments(HFTrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    adam_beta1: float = field(default=0.9)
    adam_beta2: float = field(default=0.999)
    adam_epsilon: float = field(default=1e-8)

    freeze_vision_tower: bool = field(default=False)
    freeze_llm: bool = field(default=False)
    freeze_merger: bool = field(default=False)
    disable_flash_attn2: bool = field(default=False)
    unfreeze_topk_llm: int = 0
    unfreeze_topk_vision: int = 0
    mlp_head_dim: Optional[int] = field(default=0)
    mlp_head_dropout: Optional[float] = field(default=0.0)
    
    loss_type : str = field(
        default="cross_entropy",
        metadata={"help": "Loss type to use. Should be one of `cross_entropy`, `focal_loss`, `class_balanced_cross_entropy`, or `class_balanced_focal_loss`."}
    )
    focal_alpha: Optional[str] = field(
        default=None,
        metadata={"help": "Focal Loss alpha value. If None use CrossEntropyLoss. ex '1.0,7.5'"}
    )
    focal_gamma: float = field(
        default=0.0,
        metadata={"help": "Focal Loss gamma value"}
    )
    num_labels: int = field(
        default=2,
        metadata={"help": "Number of labels for classification."}
    )
    class_balanced_beta: float = field(
        default=0.999,
        metadata={"help": "Beta value for Class Balanced Loss. If 0.0, use standard CrossEntropyLoss."}
    )
    early_stopping_patience: int = field(
        default=0,
        metadata={"help": "Number of epochs with no improvement after which training will be stopped."}
    )
    early_stopping_threshold: float = field(
        default=0.0,
        metadata={"help": "Minimum change in the monitored quantity to qualify as an improvement."}
    )

    max_seq_length: int = field(
        default=32768, # This is the default value of the qwen2-vl model
        metadata={
            "help":
                "Maximum sequence length. Sequences will be right padded (and possibly truncated)."
        },
    )
    double_quant: bool = field(
        default=True,
        metadata={"help": "Compress the quantization statistics through double quantization."}
    )
    quant_type: str = field(
        default="nf4",
        metadata={"help": "Quantization data type to use. Should be one of `fp4` or `nf4`."}
    )
    bits: int = field(
        default=16,
        metadata={"help": "How many bits to use."}
    )
    lora_enable: bool = False
    vision_lora: bool = False
    use_dora: bool = False
    lora_rank: int = 64
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    lora_weight_path: str = ""
    lora_bias: str = "none"
    vision_lr: Optional[float] = None
    merger_lr: Optional[float] = None
    head_lr: Optional[float] = None
    lora_namespan_exclude: str = field(default=None, metadata={"help": "List of namespan to exclude for LoRA"})
    num_lora_modules: int = -1
    use_liger_kernel: bool = True


@dataclass
class TrainingArguments(HFTrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    adam_beta1: float = field(default=0.9)
    adam_beta2: float = field(default=0.999)
    adam_epsilon: float = field(default=1e-8)

    freeze_vision_tower: bool = field(default=False)
    freeze_llm: bool = field(default=False)
    freeze_merger: bool = field(default=False)
    disable_flash_attn2: bool = field(default=False)
    unfreeze_topk_llm: int = 0
    unfreeze_topk_vision: int = 0

    max_seq_length: int = field(
        default=32768, # This is the default value of the qwen2-vl model
        metadata={
            "help":
                "Maximum sequence length. Sequences will be right padded (and possibly truncated)."
        },
    )

    double_quant: bool = field(
        default=True,
        metadata={"help": "Compress the quantization statistics through double quantization."}
    )
    quant_type: str = field(
        default="nf4",
        metadata={"help": "Quantization data type to use. Should be one of `fp4` or `nf4`."}
    )
    bits: int = field(
        default=16,
        metadata={"help": "How many bits to use."}
    )
    lora_enable: bool = False
    vision_lora: bool = False
    use_dora: bool = False
    lora_rank: int = 64
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    lora_weight_path: str = ""
    lora_bias: str = "none"
    vision_lr: Optional[float] = None
    merger_lr: Optional[float] = None
    lora_namespan_exclude: str = field(default=None, metadata={"help": "List of namespan to exclude for LoRA"})
    num_lora_modules: int = -1
    use_liger_kernel: bool = True

    # Generation-based evaluation settings
    generation_max_new_tokens: int = field(
        default=512,
        metadata={"help": "Maximum number of new tokens to generate during evaluation."}
    )
    sample_eval_save_predictions: bool = field(
        default=False,
        metadata={"help": "When True, generation-based eval writes predictions/references under output_dir/sample_eval_predictions."}
    )

    # 训练过程中周期性运行 scripts/eval_medical_bleu_rouge.sh（在每次保存 checkpoint 且 global_step 整除该值时）
    medical_eval_bleu_steps: int = field(
        default=0,
        metadata={
            "help": ">0 时，在 on_save 且 global_step 为该步数整数倍时运行医学 BLEU/ROUGE 评测；建议与 save_steps 一致或为约数。"
        },
    )
    medical_eval_validation_root: Optional[str] = field(
        default=None,
        metadata={"help": "评测输出根目录，默认 <output_dir>/validation。"},
    )
    medical_eval_script: Optional[str] = field(
        default=None,
        metadata={"help": "评测 shell 脚本路径，默认 <repo>/scripts/eval_medical_bleu_rouge.sh。"},
    )
    medical_eval_keep_best_n: int = field(
        default=3,
        metadata={"help": "在 validation/runs/ 下仅保留指标最优的 N 个 step 目录；CSV 仍保留全部历史。"},
    )
    medical_eval_sort_key: str = field(
        default="rougeL_finetuned",
        metadata={
            "help": "按 CSV 列名排序保留最优 runs（如 rougeL_finetuned、bleu4_finetuned）。"
        },
    )
    medical_eval_data_path: Optional[str] = field(
        default=None,
        metadata={"help": "评测 JSON；默认依次使用 DataArguments.eval_path、data_path。"},
    )
    medical_eval_image_folder: Optional[str] = field(
        default=None,
        metadata={"help": "评测图像根目录；默认依次使用 eval_image_folder、image_folder。"},
    )
    medical_eval_base_model: Optional[str] = field(
        default=None,
        metadata={"help": "评测中的基座模型路径；默认与 --model_id 相同。"},
    )
    medical_eval_batch_size: int = field(default=8, metadata={"help": "评测子进程 batch_size。"})
    medical_eval_max_new_tokens: int = field(default=1024, metadata={"help": "评测生成 max_new_tokens。"})
    medical_eval_cuda_visible_devices: Optional[str] = field(
        default=None,
        metadata={"help": "若设置，评测子进程会设置 CUDA_VISIBLE_DEVICES（例如另一张卡，减轻与训练争用）。"},
    )

@dataclass
class DPOArguments(DPOConfigTRL):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    adam_beta1: float = field(default=0.9)
    adam_beta2: float = field(default=0.999)
    adam_epsilon: float = field(default=1e-8)

    freeze_vision_tower: bool = field(default=False)
    freeze_llm: bool = field(default=False)
    freeze_merger: bool = field(default=False)
    disable_flash_attn2: bool = field(default=False)
    unfreeze_topk_llm: int = 0
    unfreeze_topk_vision: int = 0

    max_seq_length: int = field(
        default=32768, # This is the default value of the qwen2-vl model
        metadata={
            "help":
                "Maximum sequence length. Sequences will be right padded (and possibly truncated)."
        },
    )
    double_quant: bool = field(
        default=True,
        metadata={"help": "Compress the quantization statistics through double quantization."}
    )
    quant_type: str = field(
        default="nf4",
        metadata={"help": "Quantization data type to use. Should be one of `fp4` or `nf4`."}
    )
    bits: int = field(
        default=16,
        metadata={"help": "How many bits to use."}
    )
    lora_enable: bool = False
    vision_lora: bool = False
    use_dora: bool = False
    lora_rank: int = 64
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    lora_weight_path: str = ""
    lora_bias: str = "none"
    vision_lr: Optional[float] = None
    merger_lr: Optional[float] = None
    lora_namespan_exclude: str = field(default=None, metadata={"help": "List of namespan to exclude for LoRA"})
    num_lora_modules: int = -1
    use_liger_loss: bool = True
    beta: float = field(
        default=0.1,
        metadata={"help": "The beta value for DPO."}
    )
    precompute_ref_log_probs: bool = field(
        default=False,
        metadata={"help": "Whether to precompute the reference log probabilities."}
    )
    dpo_loss:str = field(
        default="sigmoid",
        metadata={"help": "The type of DPO loss to use."}
    )

@dataclass
class GRPOArguments(GRPOConfigTRL):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    adam_beta1: float = field(default=0.9)
    adam_beta2: float = field(default=0.999)
    adam_epsilon: float = field(default=1e-8)

    freeze_vision_tower: bool = field(default=False)
    freeze_llm: bool = field(default=False)
    freeze_merger: bool = field(default=False)
    disable_flash_attn2: bool = field(default=False)
    unfreeze_topk_llm: int = 0
    unfreeze_topk_vision: int = 0

    double_quant: bool = field(
        default=True,
        metadata={"help": "Compress the quantization statistics through double quantization."}
    )
    quant_type: str = field(
        default="nf4",
        metadata={"help": "Quantization data type to use. Should be one of `fp4` or `nf4`."}
    )
    bits: int = field(
        default=16,
        metadata={"help": "How many bits to use."}
    )
    lora_enable: bool = False
    vision_lora: bool = False
    use_dora: bool = False
    lora_rank: int = 64
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    lora_weight_path: str = ""
    lora_bias: str = "none"
    vision_lr: Optional[float] = None
    merger_lr: Optional[float] = None
    lora_namespan_exclude: str = field(default=None, metadata={"help": "List of namespan to exclude for LoRA"})
    num_lora_modules: int = -1
    beta: float = field(
        default=0.04,
        metadata={
            "help": "KL coefficient. If `0.0`, the reference model is not loaded, reducing memory usage and improving "
            "training speed, but may be numerically unstable for long training runs."
        },
    )
    temperature: float = 0.9
    top_p: float = 1.0
    top_k: int = 50
    min_p: Optional[float] = None
    repetition_penalty: float = 1.0
    max_completion_length: int = 256
    max_prompt_length: int = 512
    use_liger_loss: bool = True


@dataclass
class DataArguments:
    data_path: str = field(
        default=None, metadata={"help": "Path to the training data."}
    )
    eval_path: str= field(
        default=None, metadata={"help": "Path to the evaluation data."}
    )
    eval_image_folder: Optional[str] = field(
        default=None, metadata={"help": "Path to the evaluation image data."}
    )
    eval_max_samples: Optional[int] = field(
        default=None,
        metadata={"help": "Limit eval dataset size for lightweight generation checks."},
    )
    lazy_preprocess: bool = False
    image_folder: Optional[str] = field(default=None)
    image_min_pixels: Optional[int] = field(default=3136)
    image_max_pixels: Optional[int] = field(default=12845056)
    video_min_pixels: Optional[int] = field(default=100352)
    video_max_pixels: Optional[int] = field(default=602112)
    image_resized_width: int = field(default=None)
    image_resized_height: int = field(default=None)
    video_resized_width: int = field(default=None)
    video_resized_height: int = field(default=None)
    fps: Optional[int] = field(default=None, metadata={"help": "Frames per second for video data."})
    nframes: Optional[int] = field(default=None, metadata={"help": "Number of frames for video data."})
    enable_reasoning: bool = field(
        default=False,
        metadata={"help": "Enable reasoning-field parsing and model-specific <think> prompt formatting when supported."},
    )
