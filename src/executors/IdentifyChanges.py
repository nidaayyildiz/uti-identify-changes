import os
import math
import sys

import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), '../../../../'))

from sdks.novavision.src.base.component import Component
from sdks.novavision.src.helper.executor import Executor
from components.IdentifyChanges.src.utils.response import build_response
from components.IdentifyChanges.src.models.PackageModel import PackageModel



def cosine_similarity(a, b):
    dot = np.dot(a, b)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(dot / (norm_a * norm_b))


class IdentifyChanges(Component):

    def __init__(self, request, bootstrap):
        super().__init__(request, bootstrap)
        self.request.model = PackageModel(**(self.request.data))

        self.input_data = self.request.get_param("inputData")


        self.warmup = int(self.request.get_param("Warmup") or 3)


        self.identify_changes_strategy = self.request.get_param("IdentifyChangesStrategy")


        self.strategy_name = self._get_strategy_name()


        self.smoothing_factor = self._get_smoothing_factor()
        self.window_size = self._get_window_size()


        self.state = bootstrap.get("state", None)
        if self.state is None:
            self.state = self._create_initial_state()


        self.output_data = []



    def _get_strategy_name(self):

        strategy = self.identify_changes_strategy
        if hasattr(strategy, 'name'):
            return strategy.name
        if isinstance(strategy, dict):
            return strategy.get("name", "EMA")
        return str(strategy) if strategy else "EMA"

    def _get_smoothing_factor(self):


        default = 0.1
        if self.strategy_name != "EMA":
            return default
        strategy = self.identify_changes_strategy

        if hasattr(strategy, 'smoothingFactor'):
            sf = strategy.smoothingFactor
            if hasattr(sf, 'value'):
                return float(sf.value)
            return float(sf)

        if isinstance(strategy, dict):
            sf = strategy.get("smoothingFactor", {})
            if isinstance(sf, dict):
                return float(sf.get("value", default))
            return float(sf) if sf else default
        return default

    def _get_window_size(self):

        default = 10
        strategy = self.identify_changes_strategy

        if self.strategy_name == "SMA":
            if hasattr(strategy, 'windowSize'):
                ws = strategy.windowSize
                if hasattr(ws, 'value'):
                    return int(ws.value)
                return int(ws)
            if isinstance(strategy, dict):
                ws = strategy.get("windowSize", {})
                if isinstance(ws, dict):
                    return int(ws.get("value", default))
                return int(ws) if ws else default

        if self.strategy_name == "SlidingWindow":
            try:
                ws = self.request.get_param("WindowSize")
                if ws is not None:
                    return int(ws)
            except Exception:
                pass
        return default



    @staticmethod
    def _create_initial_state():
        return {

            "average": None,
            "std": None,
            "var": None,
            "M2": None,
            "sliding_window": [],
            "samples": 0,

            "cosine_similarity_avg": None,
            "cosine_similarity_std": None,
            "cosine_similarity_var": None,
            "cosine_similarity_m2": None,
            "cosine_similarity_sliding_window": [],
        }

    @staticmethod
    def bootstrap(config: dict) -> dict:
        return {
            "state": {
                "average": None,
                "std": None,
                "var": None,
                "M2": None,
                "sliding_window": [],
                "samples": 0,
                "cosine_similarity_avg": None,
                "cosine_similarity_std": None,
                "cosine_similarity_var": None,
                "cosine_similarity_m2": None,
                "cosine_similarity_sliding_window": [],
            }
        }



    def process_embedding(self, embedding_list):

        s = self.state


        is_outlier = False
        percentile = 0.5
        z_score = 0
        warming_up = False


        embedding = np.array(embedding_list, dtype=np.float64)
        norm = np.linalg.norm(embedding)
        if norm != 0:
            embedding = embedding / norm


        if s["average"] is not None:
            avg_arr = np.array(s["average"])

            cs = cosine_similarity(embedding, avg_arr)


            if s["cosine_similarity_avg"] is None:
                s["cosine_similarity_avg"] = cs
                s["cosine_similarity_std"] = 0
                s["cosine_similarity_var"] = 0
                s["cosine_similarity_m2"] = 0
            else:

                if self.strategy_name == "EMA":
                    sf = self.smoothing_factor

                    s["cosine_similarity_avg"] = (1 - sf) * s["cosine_similarity_avg"] + sf * cs

                    diff = cs - s["cosine_similarity_avg"]

                    s["cosine_similarity_var"] = (1 - sf) * s["cosine_similarity_var"] + sf * (diff ** 2)


                    s["cosine_similarity_std"] = float(np.sqrt(s["cosine_similarity_var"]))


                elif self.strategy_name == "SMA":

                    count = s["samples"] + 1

                    delta = cs - s["cosine_similarity_avg"]

                    s["cosine_similarity_avg"] = cs / count + s["cosine_similarity_avg"] * s["samples"] / count


                    delta2 = cs - s["cosine_similarity_avg"]


                    s["cosine_similarity_m2"] = s["cosine_similarity_m2"] + delta * delta2


                    var = s["cosine_similarity_m2"] / (count - 1)


                    s["cosine_similarity_std"] = float(np.sqrt(var))


                elif self.strategy_name == "SlidingWindow":

                    s["cosine_similarity_sliding_window"].append(cs)


                    if len(s["cosine_similarity_sliding_window"]) > self.window_size:
                        s["cosine_similarity_sliding_window"].pop(0)


                    s["cosine_similarity_avg"] = float(np.mean(s["cosine_similarity_sliding_window"]))

                    s["cosine_similarity_std"] = float(np.std(s["cosine_similarity_sliding_window"]))


            if s["cosine_similarity_std"] != 0:
                z_score = (cs - s["cosine_similarity_avg"]) / s["cosine_similarity_std"]
                percentile = 1 - 0.5 * (1 + math.erf(z_score / math.sqrt(2)))
            else:
                z_score = 0
                percentile = 0.5


        if s["samples"] < self.warmup:
            is_outlier = False
            warming_up = True
        else:
   
            tp = self.threshold_percentile
            is_outlier = (percentile <= tp / 2) or (percentile >= (1 - tp / 2))


        if s["average"] is None:

            s["average"] = embedding.tolist()

            s["std"] = np.zeros_like(embedding).tolist()

            s["var"] = np.zeros_like(embedding).tolist()

            s["M2"] = np.zeros_like(embedding).tolist()
        else:
            avg_arr = np.array(s["average"])
            var_arr = np.array(s["var"])
            m2_arr = np.array(s["M2"])


            if self.strategy_name == "EMA":
                sf = self.smoothing_factor
                avg_arr = (1 - sf) * avg_arr + sf * embedding

                diff = embedding - avg_arr

                var_arr = (1 - sf) * var_arr + sf * (diff ** 2)

                std_arr = np.sqrt(var_arr)

                s["average"] = avg_arr.tolist()
                s["var"] = var_arr.tolist()
                s["std"] = std_arr.tolist()


            elif self.strategy_name == "SMA":
                count = s["samples"] + 1

                delta = embedding - avg_arr

                avg_arr = avg_arr + delta / count

  
                delta2 = embedding - avg_arr

                m2_arr = m2_arr + delta * delta2

  
                var = m2_arr / (count - 1)

                std_arr = np.sqrt(var)

                s["average"] = avg_arr.tolist()
                s["M2"] = m2_arr.tolist()
                s["std"] = std_arr.tolist()


            elif self.strategy_name == "SlidingWindow":
                # Roboflow: self.sliding_window.append(embedding)
                s["sliding_window"].append(embedding.tolist())

                if len(s["sliding_window"]) > self.window_size:
                    s["sliding_window"].pop(0)

                window_arr = np.array(s["sliding_window"])
                s["average"] = np.mean(window_arr, axis=0).tolist()

                s["std"] = np.std(window_arr, axis=0).tolist()


        s["samples"] = s["samples"] + 1


        return {
            "is_outlier": is_outlier,
            "percentile": float(percentile),
            "z_score": float(z_score),
            "average": s["average"],
            "std": s["std"],
            "warming_up": warming_up,
        }


    @property
    def threshold_percentile(self):
        try:
            val = self.request.get_param("ThresholdPercentile")
            if val is not None:
                return float(val)
        except Exception:
            pass
        return 0.2


    def run(self):
        input_data = self.input_data

        if isinstance(input_data, list):
            for item in input_data:
                if isinstance(item, dict):
                    embedding = item.get("embedding")
                else:
                    embedding = item
                result = self.process_embedding(embedding)
                output_item = {}
                if isinstance(item, dict) and "uID" in item:
                    output_item["uID"] = item["uID"]
                output_item.update(result)
                self.output_data.append(output_item)
        elif isinstance(input_data, dict):
            embedding = input_data.get("embedding", input_data)
            result = self.process_embedding(embedding)
            output_item = dict(input_data)
            output_item.update(result)
            self.output_data.append(output_item)

        packageModel = build_response(context=self)
        return packageModel


if "__main__" == __name__:
    Executor(sys.argv[1]).run()
