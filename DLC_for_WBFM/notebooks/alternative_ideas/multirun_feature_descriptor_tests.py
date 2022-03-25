import prefect
from prefect import task, Flow, Parameter, unmapped
from prefect.run_configs import LocalRun
from prefect.executors import LocalDaskExecutor
from DLC_for_WBFM.notebooks.alternative_ideas.opencv_feature_descriptor_accuracy_alternative import test_feature_encoder


@task
def run_feature_test(encoder_type, encoder_kwargs):
    # DEBUG = Parameter("DEBUG", default=False)
    test_feature_encoder(encoder_type, encoder_kwargs=encoder_kwargs, DEBUG=False)


with Flow("feature_tests") as flow:
    # encoder_type = Parameter("modes", default=['latch', 'latch_different_defaults',
    #                                            'daisy', 'daisy_different_defaults',
    #                                            'akaze',
    #                                            'freak', 'freak_different_defaults'])
    # encoder_type = Parameter("modes", default=['orb', 'orb_different_defaults'])
    encoder_type1 = Parameter("encoder_type1", default='vgg_different_defaults')
    kwargs_list1 = Parameter("kwargs_list1", default=[
        dict(scale_factor=0.5),
        dict(scale_factor=1.0),
        dict(scale_factor=3.0),
        dict(scale_factor=6.0),  # Close to default
        dict(scale_factor=10.0),
        dict(scale_factor=0.5, isigma=0.5),
        dict(scale_factor=1.0, isigma=0.5),
        dict(scale_factor=3.0, isigma=0.5),
        dict(scale_factor=6.0, isigma=0.5),
        dict(scale_factor=10.0, isigma=0.5),
        dict(scale_factor=0.5, isigma=1.0),
        dict(scale_factor=1.0, isigma=1.0),
        dict(scale_factor=3.0, isigma=1.0),
        dict(scale_factor=6.0, isigma=1.0),
        dict(scale_factor=10.0, isigma=1.0),
        dict(scale_factor=0.5, isigma=2.0),
        dict(scale_factor=1.0, isigma=2.0),
        dict(scale_factor=3.0, isigma=2.0),
        dict(scale_factor=6.0, isigma=2.0),
        dict(scale_factor=10.0, isigma=2.0)
    ])
    run_feature_test.map(unmapped(encoder_type1), kwargs_list1)

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
flow.executor = LocalDaskExecutor(scheduler="processes", num_workers=4)
flow.run()
