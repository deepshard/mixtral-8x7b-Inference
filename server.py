from flask import Flask, request, jsonify
from pathlib import Path
import json
import argparse
from generation import Llama  # Assuming b.py contains the Llama class

app = Flask(__name__)

def load_model(ckpt_dir, max_batch_size, num_gpus):
    ckpt_path = Path(ckpt_dir)
    tokenizer_path = ckpt_path / "tokenizer.model"

    # Ensure the tokenizer file exists
    if not tokenizer_path.exists():
        raise FileNotFoundError(f"Tokenizer file not found at {tokenizer_path}")

    # Load parameters from params.json
    params_file = ckpt_path / "params.json"
    if not params_file.exists():
        raise FileNotFoundError(f"params.json file not found at {params_file}")

    with open(params_file, 'r') as file:
        params = json.load(file)

    # Extract max_seq_len from the parameters
    max_seq_len = params.get('max_seq_len', 128)  # Provide a default value if not present

    # Initialize and load the model
    model = Llama.build(ckpt_dir, str(tokenizer_path), max_seq_len, max_batch_size, num_gpus)
    return model

# Parse command line arguments
parser = argparse.ArgumentParser(description='Run the text generation server.')
parser.add_argument('ckpt_dir', type=str, help='Directory containing model checkpoints')
parser.add_argument('--max_batch_size', type=int, default=1, help='Maximum batch size for generation')
parser.add_argument('--num_gpus', type=int, default=1, help='Number of GPUs to use')
args = parser.parse_args()

model = load_model(args.ckpt_dir, args.max_batch_size, args.num_gpus)

@app.route('/generate', methods=['POST'])
def generate_text():
    try:
        data = request.json
        prompts = data['prompts']
        temperature = data.get('temperature', 0.6)
        top_p = data.get('top_p', 0.9)
        max_gen_len = data.get('max_gen_len', 64)

        results = model.text_completion(
            prompts,
            temperature=temperature,
            top_p=top_p,
            max_gen_len=max_gen_len
        )

        return jsonify(results)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
