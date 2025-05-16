import os
from time import time
from trustllm.generation.generation import LLMGeneration

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

tik = time()
print(f"start time: {tik}")


model_name_template = 'model_name'
model_path_template = 'path/to/your/model'

model_name = model_name_template
model_path = model_path_template

llm_gen = LLMGeneration(
    model_name=model_name, 
    model_path=model_path, 
    test_type="all",
    data_path='path/to/unziped/dataset',
    online_model=False, 
    use_deepinfra=False,
    use_replicate=False,
    repetition_penalty=1.0,
    num_gpus=1, 
    max_new_tokens=512,
    max_new_tokens=128,
    temperature=0.0, 
    debug=False,
    device='cuda'
)

llm_gen.generation_results()

tok = time()
print(f"end time: {tok}")
print(f"Time taken: {tok - tik}")

# cd TrustLLM
# nohup python -u run_generation.py > logs/aaaalogs.log 2>&1 & 

