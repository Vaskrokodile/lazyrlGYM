"""
lazyRLGYM - Reward module
=========================
Difficulty estimation, length x difficulty reward shaping, and a
pluggable LLM-as-judge.
"""
from config import RewardConfig, reward_cfg

from reward.difficulty import (
    classify_difficulty,
    consistency_score,
    estimate_difficulty,
    estimate_difficulty_from_scores,
    extract_final_answer,
)
from reward.judge import (
    APIJudge,
    BaseJudge,
    LocalJudge,
    RuleBasedJudge,
    get_judge,
)
from reward.shaping import (
    analyze_length_distribution,
    compute_all_rewards,
    compute_shaped_reward,
    count_thinking_tokens,
    get_length_target,
)

__all__ = [
    # config
    "RewardConfig",
    "reward_cfg",
    # difficulty
    "classify_difficulty",
    "consistency_score",
    "estimate_difficulty",
    "estimate_difficulty_from_scores",
    "extract_final_answer",
    # shaping
    "analyze_length_distribution",
    "compute_all_rewards",
    "compute_shaped_reward",
    "count_thinking_tokens",
    "get_length_target",
    # judge
    "APIJudge",
    "BaseJudge",
    "LocalJudge",
    "RuleBasedJudge",
    "get_judge",
]
