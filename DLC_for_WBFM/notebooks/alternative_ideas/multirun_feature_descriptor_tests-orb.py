import prefect
from prefect import task, Flow, Parameter, unmapped
from prefect.run_configs import LocalRun
from prefect.executors import LocalDaskExecutor
from wbfm.notebooks.alternative_ideas.opencv_feature_descriptor_accuracy_alternative import test_feature_encoder


@task
def run_feature_test(encoder_type, encoder_kwargs):
    # DEBUG = Parameter("DEBUG", default=False)
    test_feature_encoder(encoder_type, encoder_kwargs=encoder_kwargs, DEBUG=False)


with Flow("feature_tests") as flow:
    ##
    encoder_type2 = Parameter("encoder_type2", default='orb')
    kwargs_list2 = Parameter("kwargs_list2", default=[
        dict(patchSize=15),
        dict(patchSize=15, nfeatures=5000),
        dict(patchSize=15, firstLevel=1),
        dict(patchSize=15, nlevels=16),
        dict(patchSize=31),  # Default
        dict(patchSize=31, nfeatures=5000),
        dict(patchSize=31, firstLevel=1),
        dict(patchSize=31, nlevels=16),
        dict(patchSize=51),
        dict(patchSize=51, nfeatures=5000),
        dict(patchSize=51, firstLevel=1),
        dict(patchSize=51, nlevels=16),
        dict(patchSize=71),
        dict(patchSize=71, nfeatures=5000),
        dict(patchSize=71, firstLevel=1),
        dict(patchSize=71, nlevels=16)
    ])
    run_feature_test.map(unmapped(encoder_type2), kwargs_list2)

flow.run_config = LocalRun()
flow.executor = LocalDaskExecutor(scheduler="processes", num_workers=32)
flow.run()
