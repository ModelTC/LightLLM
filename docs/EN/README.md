## Build the docs

```bash
# Install lightllm

# git clone https://github.com/ModelTC/lightllm.git
# cd lightllm
pip install --no-deps .  
```

```bash
# Install dependencies.

# cd docs/EN
pip install -r requirements-docs.txt

# Build the docs.
make clean
make html
```

## Open the docs with your browser

```bash
python -m http.server -d build/html/ 
```

Launch your browser and open localhost:8000.
