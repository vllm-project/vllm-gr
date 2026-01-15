from vllm.benchmarks.datasets import get_samples as original_get_samples  
from vllm.benchmarks.datasets import add_dataset_parser as original_add_parser  
from vllm.benchmarks.datasets import OneRecDataset  
  
def patched_get_samples(args, tokenizer):  
    """Patch get_samples to handle OneRecDataset"""  
    # Handle default request_id_prefix  
    if not hasattr(args, "request_id_prefix"):  
        args.request_id_prefix = ""  
      
    # Check if dataset is onerec  
    if args.dataset_name == "onerec":          
        dataset = OneRecDataset(              
            task_types=args.task_types.split(","),  
            model_path=args.model,  
            dataset_path=args.data_dir,  
            disable_shuffle=args.disable_shuffle,  
            tokenizer=tokenizer,  
        )  
        input_requests = dataset.sample(  
            num_requests=args.num_prompts,  
            tokenizer=tokenizer,  
            request_id_prefix=args.request_id_prefix,  
            no_oversample=args.no_oversample,  
        )  
        return input_requests  
      
    # For all other datasets, call original function  
    return original_get_samples(args, tokenizer)  
  
def patched_add_parser(parser):  
    """Patch add_dataset_parser to include onerec in choices"""  
    # Call original parser first  
    result = original_add_parser(parser)  
      
    # Add onerec to dataset_name choices  
    for action in parser._actions:  
        if hasattr(action, 'dest') and action.dest == 'dataset_name':  
            if 'onerec' not in action.choices:  
                action.choices = list(action.choices) + ['onerec']  
            break  
      
    return result  
  
# Apply the patches  
import vllm.benchmarks.datasets  
vllm.benchmarks.datasets.get_samples = patched_get_samples  
vllm.benchmarks.datasets.add_dataset_parser = patched_add_parser