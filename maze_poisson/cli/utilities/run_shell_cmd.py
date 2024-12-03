# Function to run a shell command, 
# example -  to make a directory run: run_shell_command('mkdir'+outpath), also used in output_md.py

def run_shell_command(cmd, quiet=False, print_only=False, capture=False, **kargs):
    from subprocess import run
    if (not quiet) or print_only:
        print(cmd, **kargs)
    if (not print_only):
        sp_result = run(cmd, capture_output=capture, shell=True)
        return sp_result