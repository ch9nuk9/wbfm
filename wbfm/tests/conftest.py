from hypothesis import settings, Phase

#
settings.register_profile("quick", max_examples=2,
                          phases=[Phase.explicit, Phase.reuse])

#
settings.register_profile("long", max_examples=1000)
