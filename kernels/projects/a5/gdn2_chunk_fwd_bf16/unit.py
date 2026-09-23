"""Standalone kernel-unit hooks: complete recurrence and per-stage checks."""
from ref.reference import make_inputs, reference, validate_inputs, validate_reference
from ref.stages import reference_stages


def execute_stages(inputs, options):
    from kernels.pipeline import run
    from _unit_runner import launch_kernel
    validate_inputs(inputs)
    if options['device'] != 'a5' or options['backend'] != 'cce':
        raise ValueError('only a5/cce is declared')
    if options['block_dim'] not in (1, 2, 4, 8):
        raise ValueError('block_dim must be 1, 2, 4 or 8')
    def launch(entry, sources, outputs, scalars):
        result = launch_kernel(entry, tuple(sources.values()) + tuple(outputs.values())
                               + tuple(scalars.values()), options)
        tensors = (result,) if len(outputs) == 1 else result
        return dict(zip(outputs, tensors))
    return run(inputs, launch)


def execute(inputs, options):
    stages = execute_stages(inputs, options)
    return {name: stages[name] for name in ('o', 'final_state')}
