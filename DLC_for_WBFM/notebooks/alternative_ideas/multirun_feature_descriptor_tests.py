import prefect
from prefect import task, Flow, Parameter
from prefect.run_configs import LocalRun
from prefect.executors import LocalDaskExecutor
from DLC_for_WBFM.notebooks.alternative_ideas.opencv_feature_descriptor_accuracy_alternative import test_feature_encoder


@task
def run_feature_test(encoder_type):
    DEBUG = Parameter("DEBUG", default=True)
    test_feature_encoder(encoder_type, DEBUG)


with Flow("feature_tests") as flow:
    # encoder_type = Parameter("modes", default=['baseline', 'vgg_different_defaults'])
    # encoder_type = Parameter("modes", default=['latch', 'latch_different_defaults',
    #                                            'daisy', 'daisy_different_defaults',
    #                                            'akaze',
    #                                            'freak', 'freak_different_defaults'])
    encoder_type = Parameter("modes", default=['orb'])
    run_feature_test.map(encoder_type)

flow.run_config = LocalRun()
flow.executor = LocalDaskExecutor(scheduler="processes", num_workers=8)
flow.run()
