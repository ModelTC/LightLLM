"""Integer shape arithmetic independent of accelerator kernel runtimes."""


def ceil_div(value, divisor):
    return (value + divisor - 1) // divisor
