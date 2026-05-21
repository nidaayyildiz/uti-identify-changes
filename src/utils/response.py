from sdks.novavision.src.helper.package import PackageHelper
from components.IdentifyChanges.src.models.PackageModel import IdentifyChangesOutputs, IdentifyChangesResponse, OutputData, PackageModel, PackageConfigs, ConfigExecutor, IdentifyChangesExecutor


def build_response(context):
    outputData = OutputData(value=context.output_data)
    outputs = IdentifyChangesOutputs(outputData=outputData)
    identifyChangesResponse = IdentifyChangesResponse(outputs=outputs)
    identifyChangesExecutor = IdentifyChangesExecutor(value=identifyChangesResponse)
    configExecutor = ConfigExecutor(value=identifyChangesExecutor)
    packageConfigs = PackageConfigs(executor=configExecutor)

    package = PackageHelper(packageModel=PackageModel, packageConfigs=packageConfigs)
    return package.build_model(context)
