import os
import glob

for filename in glob.glob('./**/*.py', recursive=True):
    print(filename)
    os.system(f"autopep8 --max-line-length 140 --in-place --aggressive --aggressive {filename}")
