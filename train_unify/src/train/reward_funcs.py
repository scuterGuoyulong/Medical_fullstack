import re

try:
    from math_verify import LatexExtractionConfig, parse, verify
    from latex2sympy2_extended import NormalizationConfig
except ImportError:
    LatexExtractionConfig = None
    NormalizationConfig = None
    parse = None
    verify = None

def accuracy_reward(completions, assistant, **kwargs):
    """Reward function that checks if the completion is correct using either symbolic verification or exact string matching."""
    rewards = []

    for completion, sol in zip(completions, assistant):
        if parse is None or verify is None or LatexExtractionConfig is None or NormalizationConfig is None:
            rewards.append(float(completion.strip().lower() == sol.strip().lower()))
            continue

        try:
            gold_parsed = parse(sol, extraction_mode="first_match")
        except Exception as e:
            gold_parsed = []

        if len(gold_parsed) != 0:
            # Try parsing predicted answer too
            try:
                answer_parsed = parse(
                    completion,
                    extraction_config=[
                        LatexExtractionConfig(
                            normalization_config=NormalizationConfig(
                                nits=False,
                                malformed_operators=False,
                                basic_latex=True,
                                boxed="all",
                                units=True,
                            ),
                            boxed_match_priority=0,
                            try_extract_without_anchor=False,
                        )
                    ],
                    extraction_mode="first_match",
                )
                reward = float(verify(gold_parsed, answer_parsed))
            except Exception as e:
                print(f"verify failed: {e}, answer: {completion}, gold: {sol}")
                reward = None
        else:
            # fallback to text match
            reward = float(completion.strip().lower() == sol.strip().lower())

        rewards.append(reward)

    return rewards

def format_reward(completions, **kwargs):
    """Reward function that checks if the completion has a specific format."""
    pattern = r"^<think>\n.*?\n</think>\n<answer>\n.*?\n</answer>$"
    matches = [re.match(pattern, content, re.DOTALL | re.MULTILINE) for content in completions]
    rewards = [1.0 if match else 0.0 for match in matches]
    return rewards


_REGION_ALIASES = {
    "左上肺野": ["左上肺野", "左上肺", "left upper lung"],
    "左中肺野": ["左中肺野", "左中肺", "left middle lung", "left mid lung"],
    "左下肺野": ["左下肺野", "左下肺", "left lower lung"],
    "右上肺野": ["右上肺野", "右上肺", "right upper lung"],
    "右中肺野": ["右中肺野", "右中肺", "right middle lung", "right mid lung"],
    "右下肺野": ["右下肺野", "右下肺", "right lower lung"],
    "左肺野": ["左肺野", "左肺", "left lung"],
    "右肺野": ["右肺野", "右肺", "right lung"],
    "双肺野": ["双肺野", "双肺", "bilateral lung", "bilateral lungs"],
    "心影纵隔区": ["心影纵隔区", "心影", "纵隔", "cardiomediastinal", "mediastinum"],
    "左肋膈角": ["左肋膈角", "left costophrenic"],
    "右肋膈角": ["右肋膈角", "right costophrenic"],
    "左肺门": ["左肺门", "left hilum", "left hilar"],
    "右肺门": ["右肺门", "right hilum", "right hilar"],
}


def _flatten_completion(completion):
    if isinstance(completion, list) and completion:
        last = completion[-1]
        if isinstance(last, dict):
            return str(last.get("content", ""))
    return str(completion)


def _expected_label(text):
    if "不一致" in text or "mismatch" in text.lower():
        return "mismatch"
    if "一致" in text or "match" in text.lower():
        return "match"
    return None


def _predicted_label(text):
    text_l = text.lower()
    if re.search(r"不一致|不匹配|不是|mismatch|not consistent|incorrect", text_l):
        return "mismatch"
    if re.search(r"一致|匹配|是|match|consistent|correct", text_l):
        return "match"
    return None


def _expected_region(text):
    for canonical, aliases in _REGION_ALIASES.items():
        if any(alias.lower() in text.lower() for alias in aliases):
            return canonical
    return None


def _mentions_region(text, canonical):
    aliases = _REGION_ALIASES.get(canonical, [canonical])
    text_l = text.lower()
    return any(alias.lower() in text_l for alias in aliases)


def grounding_label_reward(completions, assistant, **kwargs):
    """Reward consistency with the expected grounding decision.

    The reward is active only for X-ray grounding samples whose reference answer
    contains 一致/不一致 (or match/mismatch). Non-grounding samples receive None.
    """
    rewards = []
    for completion, reference in zip(completions, assistant):
        expected = _expected_label(str(reference))
        if expected is None:
            rewards.append(None)
            continue
        pred = _predicted_label(_flatten_completion(completion))
        rewards.append(1.0 if pred == expected else 0.0)
    return rewards


def grounding_region_reward(completions, assistant, **kwargs):
    """Reward mentioning the expected anatomic region for grounding samples."""
    rewards = []
    for completion, reference in zip(completions, assistant):
        region = _expected_region(str(reference))
        if region is None:
            rewards.append(None)
            continue
        rewards.append(1.0 if _mentions_region(_flatten_completion(completion), region) else 0.0)
    return rewards


def grounding_cot_format_reward(completions, assistant=None, **kwargs):
    """Reward concise long-CoT grounding structure.

    This expects generated text to include a thinking block and a final answer
    carrying the grounding decision. It stays inactive for non-grounding samples
    when references are available and do not look like grounding answers.
    """
    rewards = []
    references = assistant or [None] * len(completions)
    for completion, reference in zip(completions, references):
        if reference is not None and _expected_label(str(reference)) is None:
            rewards.append(None)
            continue
        text = _flatten_completion(completion)
        has_think = bool(re.search(r"<think>[\s\S]*?</think>", text))
        has_decision = _predicted_label(text) is not None
        rewards.append(1.0 if has_think and has_decision else 0.0)
    return rewards
