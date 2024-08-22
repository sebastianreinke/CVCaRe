from cx_Freeze import setup, Executable

setup(
    name = "CVCaRe",
    version = "7.0",
    description = "A tool for the evaluation and handling of cyclic voltammograms, with particular tools for determining resistance and capacitance from capacitive cyclic voltammograms accurately.",
    executables = [Executable("gui.py")]
)