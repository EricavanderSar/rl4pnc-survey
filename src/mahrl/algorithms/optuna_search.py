from typing import Dict
from ray.tune.search.optuna import OptunaSearch


class MyOptunaSearch(OptunaSearch):
    def on_trial_result(self, trial_id: str, result: Dict):
        if self._metric not in result:
            return
        else:
            super().on_trial_result(trial_id, result)
