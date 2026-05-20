
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

    name: Literal["SmoothingFactor"] = "SmoothingFactor"
    value: float = Field(ge=0.0, le=1.0, default=0.05)
    type: Literal["number"] = "number"
    field: Literal["textInput"] = "textInput"
    placeHolder: Literal["[0.0, 1.0]"] = "[0.0, 1.0]"

    class Config:
        title = "Smoothing Factor"
        json_schema_extra = {"shortDescription": "Factor for smoothing the data."}

class Warmup(Config):

    name: Literal["Warmup"] = "Warmup"
    value: int = Field(ge=2, default=10)
    type: Literal["number"] = "number"
    field: Literal["textInput"] = "textInput"
    placeHolder: Literal["[2, ...]"] = "[2, ...]"

    class Config:
        title = "Warmup "
        json_schema_extra = {"shortDescription": "Minimum samples collected before detection starts."}

class WindowSize(Config):

    name: Literal["WindowSize"] = "WindowSize"
    value: int = Field(ge=2, default=10)
    type: Literal["number"] = "number"
    field: Literal["textInput"] = "textInput"
    placeHolder: Literal["[2, ...]"] = "[2, ...]"

    class Config:
        title = "Window Size"
        json_schema_extra = {"shortDescription": "Number of recent embeddings kept for comparison."}


class EMA(Config):
    name: Literal["EMA"] = "EMA"
    smoothingFactor: SmoothingFactor
    value: Literal["EMA"] = "EMA"
    type: Literal["bool"] = "bool"
    field: Literal["option"] = "option"

    class Config:
        title = "EMA"


class SMA(Config):
    name: Literal["SMA"] = "SMA"
    windowSize: WindowSize
    value: Literal["SMA"] = "SMA"
    type: Literal["bool"] = "bool"
    field: Literal["option"] = "option"

    class Config:
        title = "SMA"

class SlidingWindow(Config):
    name: Literal["SlidingWindow"] = "SlidingWindow"
    value: Literal["SlidingWindow"] = "SlidingWindow"
    type: Literal["bool"] = "bool"
    field: Literal["option"] = "option"

    class Config:
        title = "Sliding Window"

    class Config:
        title = "SMA"


class IdentifyChangesStrategy(Config):

    name: Literal["IdentifyChangesStrategy"] = "IdentifyChangesStrategy"
    value: Union[SMA, EMA, SlidingWindow]
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
