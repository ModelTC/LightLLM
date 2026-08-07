import glob
import subprocess

for filename in glob.glob('./**/*.py', recursive=True):
    print(filename)
    subprocess.run(['autopep8', '--max-line-length', '140', '--in-place', '--aggressive', '--aggressive', filename])
