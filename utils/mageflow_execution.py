"""Validate opt-in MageFlow checkpoint selection and block compilation."""


def validate_execution_config(config, num_blocks=None):
    model = config['model']
    selected = model.get('checkpoint_blocks')
    if 'checkpoint_blocks' in model:
        if not isinstance(selected, list) or any(type(i) is not int or i < 0 for i in selected):
            raise ValueError('model.checkpoint_blocks must be a list of non-negative, zero-based block indices')
        if len(set(selected)) != len(selected):
            raise ValueError('model.checkpoint_blocks must not contain duplicate indices')
        if config.get('activation_checkpointing') is not True:
            raise ValueError('model.checkpoint_blocks requires activation_checkpointing=true')
        if config.get('blocks_to_swap', 0):
            raise ValueError('Selective MageFlow checkpointing is not supported with blocks_to_swap')
        if num_blocks is not None and any(i >= num_blocks for i in selected):
            raise ValueError(f'model.checkpoint_blocks indices must be in [0, {num_blocks - 1}]')

    compile_blocks = model.get('compile_blocks', False)
    if type(compile_blocks) is not bool:
        raise ValueError('model.compile_blocks must be true or false')
    compile_options = {'compile_mode', 'compile_dynamic'}
    if not compile_blocks and compile_options.intersection(model):
        raise ValueError('model.compile_mode/compile_dynamic require model.compile_blocks=true')
    if compile_blocks:
        if config.get('compile', False):
            raise ValueError('Use model.compile_blocks=true with top-level compile=false to avoid nested compilation')
        if config.get('blocks_to_swap', 0):
            raise ValueError('MageFlow block compilation is not supported with blocks_to_swap')
        if config.get('activation_checkpointing') == 'unsloth':
            raise ValueError('MageFlow block compilation requires native checkpointing, not unsloth')
        if config.get('optimizer', {}).get('gradient_release', False):
            raise ValueError('MageFlow block compilation is not supported with gradient_release')
        if model.get('attention_backend', 'sdpa') != 'sdpa':
            raise ValueError('MageFlow block compilation currently supports attention_backend="sdpa" only')
        if model.get('compile_mode', 'default') not in ('default', 'max-autotune-no-cudagraphs'):
            raise ValueError('model.compile_mode must be default or max-autotune-no-cudagraphs')
        if type(model.get('compile_dynamic', True)) is not bool:
            raise ValueError('model.compile_dynamic must be true or false')


def compile_block_forward(forward, model_config):
    """Compile a function, preserving module ownership and checkpoint keys.

    One callable is shared across blocks. Only tensor computation is captured;
    offloader actions, Qwen and pipeline bookkeeping stay outside the graph.
    CUDA graphs are intentionally excluded: accumulation holds several forward
    graphs live and sampling has a different execution pattern.
    """
    import torch

    return torch.compile(
        forward,
        dynamic=model_config.get('compile_dynamic', True),
        mode=model_config.get('compile_mode', 'default'),
        fullgraph=True,
    )
