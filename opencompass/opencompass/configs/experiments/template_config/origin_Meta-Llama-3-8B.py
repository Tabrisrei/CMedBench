from mmengine.config import read_base

with read_base():
    from ..datasets.mmlu_health_ppl import mmlu_datasets
    from ..datasets.pubmedqa_ppl import pubmedqa_datasets
    from ..datasets.medqa_ppl import medqa_datasets
    from ..datasets.medmcqa_ppl import medmcqa_datasets
    from ..datasets.jmed_ppl import jmed_datasets
    from ..datasets.medexqa_ppl import medexqa_datasets
    from ..datasets.medxpertqa_ppl import medxpertqa_datasets
    from ..datasets.careqa_ppl import careqa_datasets
    from ..datasets.medbulletsop5_ppl import medbulletsop5_datasets
    from ..datasets.mmedxpertqa_ppl import medxpertqa_datasets

datasets = [
    *mmlu_datasets,
    *pubmedqa_datasets,
    *medqa_datasets,
    *medmcqa_datasets,
    *jmed_datasets,
    *medexqa_datasets,
    *medxpertqa_datasets,
    *careqa_datasets,
    *medbulletsop5_datasets,
    *medxpertqa_datasets
]

# =============================================================================
from opencompass.models import HuggingFaceCausalLM

models = [
    dict(
        type=HuggingFaceCausalLM,
        abbr='origin_Meta-Llama-3-8B',
        path='/root/path/to/base_models/meta-llama/Meta-Llama-3-8B', 
        tokenizer_path='/root/path/to/base_models/meta-llama/Meta-Llama-3-8B', 
        model_kwargs=dict(
                        cache_dir='/root/path/to/base_models/meta-llama/Meta-Llama-3-8B',
                        device_map='auto', 
                        trust_remote_code=True, 
                        ),
        tokenizer_kwargs=dict(
                        padding_side='left', 
                        truncation_side='left', 
                        trust_remote_code=True, 
                        use_fast=False), # kwargs for tokenizer loading from_pretrained
        batch_padding=True,
        max_out_len=300,
        max_seq_len=2048,
        batch_size=16,
        run_cfg=dict(num_gpus=1),
    )
]