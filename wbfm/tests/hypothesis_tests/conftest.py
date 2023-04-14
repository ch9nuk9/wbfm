from hypothesis import settings, Phase

#
settings.register_profile("quick", max_examples=2,
                          phases=[Phase.explicit, Phase.reuse])

#
settings.register_profile("noshrink", max_examples=100,
                          phases=[Phase.explicit, Phase.reuse, Phase.generate, Phase.target])

#
settings.register_profile("long", max_examples=1000)
