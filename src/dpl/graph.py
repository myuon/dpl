from dpl import Variable, Function
import tempfile
import subprocess
from pathlib import Path


def dot_variable(v: Variable, verbose=True):
    return f'{id(v)} [label="{v.name if v.name is not None else '' if not(verbose) else f"{v.name}: {v.shape} {v.dtype}"}" color=orange style=filled]\n'


def dot_function(f: Function, verbose=True):
    name = f.__class__.__name__
    txt = f'{id(f)} [label="{name}" color=lightblue style=filled]\n'

    for x in f.inputs:
        txt += f"{id(x)} -> {id(f)}\n"

    for y in f.outputs:
        y = y()
        txt += f"{id(f)} -> {id(y)}\n"

    return txt


def create_dot_graph(output, verbose=True):
    writer = ""
    stack = []
    visited = set()

    def add_func(f):
        if f not in visited:
            stack.append(f)
            visited.add(f)

    add_func(output.creator)
    writer += "digraph g {\n"
    writer += dot_variable(output, verbose)

    while stack:
        f = stack.pop()
        writer += dot_function(f, verbose)

        for x in f.inputs:
            writer += dot_variable(x, verbose)
            if x.creator is not None:
                add_func(x.creator)

    writer += "}\n"
    return writer


def plot_dot_graph(output, verbose=True, to_file=None):
    """
    Create a computational graph visualization and display it interactively.

    Args:
        output: Variable to visualize the computation graph for
        verbose: Whether to include detailed information
        to_file: Optional file path to save the image
    """
    dot_graph = create_dot_graph(output, verbose)

    # Create a temporary file for the dot graph
    with tempfile.NamedTemporaryFile(mode="w", suffix=".dot", delete=False) as f:
        f.write(dot_graph)
        dot_file = f.name

    try:
        # Generate PNG image using Graphviz
        if to_file:
            png_file = to_file
        else:
            png_fd, png_file = tempfile.mkstemp(suffix=".png")
            import os

            os.close(png_fd)

        subprocess.run(["dot", "-Tpng", dot_file, "-o", png_file], check=True)

        # Display using matplotlib
        try:
            import matplotlib.pyplot as plt
            import matplotlib.image as mpimg

            img = mpimg.imread(png_file)
            plt.figure(figsize=(12, 8))
            plt.imshow(img)
            plt.axis("off")
            plt.tight_layout()
            plt.show()
        except ImportError:
            print(f"matplotlib not available. Image saved to: {png_file}")

        # Clean up temporary files if not saving
        if not to_file:
            Path(png_file).unlink(missing_ok=True)
    finally:
        Path(dot_file).unlink(missing_ok=True)
