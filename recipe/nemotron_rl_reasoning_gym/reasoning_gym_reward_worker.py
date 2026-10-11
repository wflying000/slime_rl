"""One-shot CPU scoring worker, terminated by the caller on timeout."""

import contextlib
import json
import sys
from pathlib import Path

# Executing this file by path only puts its recipe directory on sys.path.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from recipe.rl_reward.timeout_utils import RewardTimeout, is_timeout_error


def main() -> None:
    request = json.load(sys.stdin)
    # Keep stdout exclusively for the result protocol. Importing the reward
    # module does not import torch; Sample is only used for type checking.
    with contextlib.redirect_stdout(sys.stderr):
        from recipe.nemotron_rl_reasoning_gym.reasoning_gym_reward import _score_answer

        try:
            reward = _score_answer(request["answer"], request["label"], request["metadata"])
        except (Exception, RewardTimeout) as exc:
            if not is_timeout_error(exc):
                raise
            result = {"timeout": True, "error": f"{type(exc).__name__}: {exc}"}
        else:
            result = {"reward": float(reward)}
    print(json.dumps(result, allow_nan=False))


if __name__ == "__main__":
    main()
