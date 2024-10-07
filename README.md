# mlx-engine
MLX engine for LM Studio

# Demo
To run a demo of model load and inference:
1. Clone the repository
```
git clone git@github.com:lmstudio-ai/mlx-engine.git
cd mlx-engine
```
2. Create a virtual environment (optional)
```
 python -m venv myenv
 source myenv/bin/activate
```
3. Install the required dependency packages
```
pip install -r requirements.txt
```
4. Run the `demo.py` script
```
python demo.py --model ~/.cache/lm-studio/models/mlx-community/Meta-Llama-3.1-8B-Instruct-4bit 
```