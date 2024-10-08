<p align="center">
  <br/>
  <picture> 
    <img alt="lmstudio + MLX" src="https://github.com/user-attachments/assets/4a7e8a2a-59e0-46d1-a5ad-55c0b51a10a0">

  </picture>
  <br/>
  <br/>
</p>

<p align="center"><bold><code>mlx-engine</code> - Apple MLX LLM Engine for <a href="https://lmstudio.ai/">LM Studio</a></bold></p>
<p align="center"><a href="https://discord.gg/aPQfnNkxGC"><img alt="Discord" src="https://img.shields.io/discord/1110598183144399058?logo=discord&style=flat&logoColor=white"></a></p>

# mlx-engine
MLX engine for LM Studio

## Demo
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
