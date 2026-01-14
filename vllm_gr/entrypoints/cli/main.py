import sys
from vllm.entrypoints.cli.main import main as vllm_main
#import vllm_gr.serve
import importlib.metadata

def main():
    if "serve" in sys.argv and "--gr" in sys.argv:
        print("here")
        # Launch vLLM-gr specialized serving path    
        #vllm_gr.serve.main(sys.argv)
        from vllm.entrypoints.utils import VLLM_SUBCMD_PARSER_EPILOG, cli_env_setup
        #from vllm.utils.argparse_utils import FlexibleArgumentParser
        # import vllm_omni.entrypoints.cli.serve
        # CMD_MODULES = [
        #     vllm_omni.entrypoints.cli.serve,
        # ]

        cli_env_setup()

        #For now remove the --gr flag from sys.argv to avoid issues in vLLM CLI
        idx = sys.argv.index("--gr")  
        # Remove the flag and its value (if it has one)  
        if idx + 1 < len(sys.argv) and not sys.argv[idx + 1].startswith("--"):  
            sys.argv.pop(idx + 1)  # Remove value  
        sys.argv.pop(idx)  # Remove flag  

    # Delegate to original vLLM CLI
    vllm_main()


if __name__ == "__main__":
    main()