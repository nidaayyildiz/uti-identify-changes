
from pydantic import Field, validator
from typing import List, Optional, Union, Literal
from sdks.novavision.src.base.model import Package, Image, Inputs, Configs, Outputs, Response, Request, Output, Input, Config


class InputData(Input):
    name: Literal["inputData"] = "inputData"
    value: Union[dict, list]
    type: str = "object"

    @validator("type", pre=True, always=True)
    def set_type_based_on_value(cls, value, values):
        v = values.get("value")
        if isinstance(v, list):
            return "list"
        return "object"

    class Config:
        title = "Input Data"

class OutputData(Output):
    name: Literal["outputData"] = "outputData"
    value: Union[dict, list]
    type: str = "object"

    @validator("type", pre=True, always=True)
    def set_type_based_on_value(cls, value, values):
        v = values.get("value")
        if isinstance(v, list):
            return "list"
        return "object"

    class Config:
        title = "Output Data"

class SmoothingFactor(Config):
    """Smoothing factor (alpha) used by the EMA strategy when updating the
    running mean and variance of cosine similarity scores and embedding
    vectors.

    Each new sample contributes `alpha` weight while the existing historical
    aggregate keeps `(1 - alpha)` weight. Lower values (e.g. 0.05) produce a
    heavily smoothed signal that reacts slowly to change and is robust to
    noise; higher values (e.g. 0.3) react quickly to new data but are more
    sensitive to short-term fluctuations. Valid range is [0.0, 1.0]."""

    name: Literal["SmoothingFactor"] = "SmoothingFactor"
    value: float = Field(ge=0.0, le=1.0, default=0.05)
    type: Literal["number"] = "number"
    field: Literal["textInput"] = "textInput"
    placeHolder: Literal["[0.0, 1.0]"] = "[0.0, 1.0]"

    class Config:
        title = "Smoothing Factor"
        json_schema_extra = {"shortDescription": "Factor for smoothing the data."}

class Warmup(Config):
    """Minimum number of samples that must be observed before the component
    starts emitting outlier / change decisions.

    During the warmup phase running statistics (mean, variance, std) are
    accumulated but no `is_outlier` decision is reported, because z-score
    and percentile computed from very few samples are statistically
    unreliable and would produce false positives at start-up. Once the
    observed sample count reaches `Warmup`, the detector activates. Must
    be >= 2."""

    name: Literal["Warmup"] = "Warmup"
    value: int = Field(ge=2, default=10)
    type: Literal["number"] = "number"
    field: Literal["textInput"] = "textInput"
    placeHolder: Literal["[2, ...]"] = "[2, ...]"

    class Config:
        title = "Warmup "
        json_schema_extra = {"shortDescription": "Minimum samples collected before detection starts."}

class WindowSize(Config):
    """Size of the FIFO window used by the SlidingWindow strategy (and the
    SMA configuration that exposes it).

    Only the most recent `WindowSize` samples are retained in memory; older
    samples are discarded on overflow. On every update the mean and
    standard deviation are recomputed from the current window contents.
    Larger windows give a more stable baseline but react more slowly to
    genuine distribution shifts; smaller windows adapt faster at the cost
    of noisier statistics. Must be >= 2."""

    name: Literal["WindowSize"] = "WindowSize"
    value: int = Field(ge=2, default=10)
    type: Literal["number"] = "number"
    field: Literal["textInput"] = "textInput"
    placeHolder: Literal["[2, ...]"] = "[2, ...]"

    class Config:
        title = "Window Size"
        json_schema_extra = {"shortDescription": "Number of recent embeddings kept for comparison."}


class EMA(Config):
    """Exponential Moving Average strategy.

    Maintains the running mean and variance of cosine similarity and of the
    embedding vector (element-wise) with exponentially decaying weights:
    recent samples have stronger influence than older ones, and no history
    is stored (O(1) memory). The decay rate is controlled by the
    `SmoothingFactor` (alpha) sub-parameter — lower alpha = more
    smoothing / slower adaptation, higher alpha = faster adaptation but
    noisier estimates."""

    name: Literal["EMA"] = "EMA"
    smoothingFactor: SmoothingFactor
    value: Literal["EMA"] = "EMA"
    type: Literal["bool"] = "bool"
    field: Literal["option"] = "option"

    class Config:
        title = "Exponential Moving Average (EMA)"


class SMA(Config):
    """Simple Moving Average strategy, implemented with Welford's online
    algorithm.

    Updates mean and variance incrementally in a numerically stable way
    while giving equal weight to every sample ever observed, without
    keeping the history (O(1) memory). Well suited when the underlying
    distribution is expected to be stationary; adapts very slowly to
    permanent regime changes because old samples are never forgotten."""

    name: Literal["SMA"] = "SMA"
    windowSize: WindowSize
    value: Literal["SMA"] = "SMA"
    type: Literal["bool"] = "bool"
    field: Literal["option"] = "option"

    class Config:
        title = "Simple Moving Average (SMA)"

class SlidingWindow(Config):
    """Sliding Window strategy.

    Keeps the last `WindowSize` samples in a FIFO buffer and on every
    update recomputes mean and standard deviation from the buffer
    contents (both for the cosine similarity scalar and for the embedding
    vector, element-wise). Reacts quickly to distribution shifts because
    samples older than the window are dropped entirely, at the cost of
    O(WindowSize) memory and per-step recomputation."""

    name: Literal["SlidingWindow"] = "SlidingWindow"
    value: Literal["SlidingWindow"] = "SlidingWindow"
    type: Literal["bool"] = "bool"
    field: Literal["option"] = "option"

    class Config:
        title = "Sliding Window"




class IdentifyChangesStrategy(Config):
    """Algorithm used to maintain the running statistics (mean / variance /
    std) of the cosine similarity stream and the embedding stream.

    The selected strategy decides *how* the historical baseline is built;
    the downstream change-detection step (z-score → percentile → outlier
    decision) is identical for all strategies. Available options:

    - **EMA**: exponentially weighted average, O(1) memory, controlled by
      `smoothingFactor`.
    - **SMA**: Welford online mean/variance, O(1) memory, equal weight to
      all past samples.
    - **SlidingWindow**: last-N samples kept in a FIFO buffer, O(N)
      memory, controlled by `windowSize`."""

    name: Literal["IdentifyChangesStrategy"] = "IdentifyChangesStrategy"
    value: Union[EMA, SMA, SlidingWindow]
    type: Literal["object"] = "object"
    field: Literal["dropdownlist"] = "dropdownlist"

    class Config:
        title = "Strategy"



class IdentifyChangesInputs(Inputs):
    inputData: InputData


class IdentifyChangesConfigs(Configs):
    warmup: Warmup
    identifyChangesStrategy: IdentifyChangesStrategy


class IdentifyChangesOutputs(Outputs):
    outputData: OutputData


class IdentifyChangesRequest(Request):
    inputs: Optional[IdentifyChangesInputs]
    configs: IdentifyChangesConfigs

    class Config:
        json_schema_extra = {
            "target": "configs"
        }


class IdentifyChangesResponse(Response):
    outputs: IdentifyChangesOutputs


class IdentifyChangesExecutor(Config):
    name: Literal["IdentifyChanges"] = "IdentifyChanges"
    value: Union[IdentifyChangesRequest, IdentifyChangesResponse]
    type: Literal["object"] = "object"
    field: Literal["option"] = "option"

    class Config:
        title = "Package"
        json_schema_extra = {
            "target": {
                "value": 0
            }
        }


class ConfigExecutor(Config):
    name: Literal["ConfigExecutor"] = "ConfigExecutor"
    value: Union[IdentifyChangesExecutor]
    type: Literal["executor"] = "executor"
    field: Literal["dependentDropdownlist"] = "dependentDropdownlist"

    class Config:
        title = "Task"
        json_schema_extra = {
            "target": "value"
        }


class PackageConfigs(Configs):
    executor: ConfigExecutor


class PackageModel(Package):
    configs: PackageConfigs
    type: Literal["component"] = "component"
    name: Literal["IdentifyChanges"] = "IdentifyChanges"
