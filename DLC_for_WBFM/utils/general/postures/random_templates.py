import numpy as np


def generator_random_template_times(num_frames, t_template):
    yield t_template
    permuted_times = np.random.permutation(range(num_frames))
    for t_random in permuted_times:
        if t_random != t_template:
            yield int(t_random)
        else:
            continue


def generate_random_valid_template_frames(all_frames, min_neurons_for_template, num_frames, t_template,
                                          num_random_templates):
    template_generator = generator_random_template_times(num_frames, t_template)
    all_templates = [t for t in template_generator if all_frames[t].num_neurons > min_neurons_for_template]
    return all_templates[:num_random_templates]
