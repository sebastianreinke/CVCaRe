# Welcome to the GitHub page for CVCaRe!

CVCaRe is an analysis tool for cyclic voltammograms (CVs). It can read CV data, parse cycles, calculate capacitance using both conventional methods, and implement a new *CaRe* technique to determine capacitance and resistance from a CV. Additionally, CVCaRe provides a GUI built with PySimpleGUI for easy interaction.

## Features

- **Data Parsing**: Reads cyclic voltammograms and parses them into cycles. If provided, cycle data will be used. Otherwise, cycles are inferred from the potential waveform.
- **Capacitance Calculation**:
  - **Conventional Methods**: Calculates capacitance using the current difference at selected potentials.
  - **MinMax**: Calculates the capacitance based on the current difference between the maximum and minimum current.
  - **CaRe Analysis**: A new technique that calculates both capacitance and resistance, factoring in distortion.
- **CV Analysis Tools**:
  - True scanrate plots
  - Cycle splitting
  - Integral calculations within specific voltage bounds
  - Distorted capacitive CV analysis
- **Data Export**: You may export your data files themselves in a universal format, or export a cycle-split version of your dataset to process further, as well as the results from the capacitance calculation.
- **User-friendly GUI**: Built with PySimpleGUI, making the analysis process interactive and accessible.

## Installation

There are two ways to use CVCaRe: There is an executable file, that will run without installation. It provides the same functionality, but is usually somewhat slower to start up.
Due to file size limitations on GitHub, download it here: https://drive.proton.me/urls/6E5JR995SG#7HNMXmInDhLB

You can also run the python files directly. For this, download this repository, for example by following these steps:


1. Clone the repository:
    ```bash
    git clone https://github.com/sebastianreinke/CVCaRe.git
    cd CVCaRe
    ```

2. Set up a Python virtual environment (optional but recommended):
    ```bash
    python3 -m venv venv
    source venv/bin/activate # On Windows, use `venv\Scripts\activate`
    ```

3. Install the necessary dependencies:
    ```bash
    pip install -r requirements.txt
    ```

4. Run the application:
    ```bash
    python main.py
    ```

## Issues
If you find bugs or errors in the program, please notify me at mailto:cvcare_github@sreinke.slmail.me

### Typical issues:
- The menu bar is not displayed correctly for some zoom settings of the monitor. If you do not use 100% zoom, set your monitor to 100% zoom before starting CVCaRe. This will fix the issue, even if you then return to your previous setting.
- A data file is not loaded. If it is a standard CSV file (comma as separator, dot as decimal marker) check the "Assume default CSV format" checkbox and retry. If the issue persists, the file format cannot yet be parsed by CVCaRe.
In this case, please send me an e-mail with an example data file that enables me to reproduce the issue.

## Usage

### GUI

- **Loading CV Files**: Select your files using the Browse File(s) button and click the "Load and preview CVs" button. Only after loading the CVs are they available for other calculations.
- **Save Files**: Save your data files in a uniform format by clicking the "Write CVs to file" button next to "Load and preview CVs". There, you may choose filenames, comments, and whether to save individual files or one composite file.
- **Capacitance Calculation**: Choose between
    - At selected voltage
    - Min/Max Current
    - CaRe analysis (for distortion-corrected capacitance)
- **Save Results**: After calculating capacitance or performing analysis, save the data to a file via the "Save Capacitance" button.
- **Additional Analysis**: Perform various analyses such as calculating integrals, bias analysis, and cycle splitting.

### Example

Here is a basic workflow to determine capacitance and resistance of CVs in the GUI:

1. Load CV files using "Load and preview CVs."
2. Enter a scan rate and the cycle number you wish to evaluate. If the cycle number is too high, it will default to the highest available.
3. On the right-hand side, under advanced CV calculations, enter which CV you wish to analyse in "Use CV No.". As there can be multiple CVs loaded at any one time on the left, they are numbered from top to bottom starting at 1.
4. Click "Perform distorted capacitive CV analysis". The results appear below to be copied for further use.
5. If you wish to perform this calculation in bulk and save it automatically, repeat steps 1 and 2 for all CVs.
6. On the left-hand side, find "Export to file:" and use the "Save As..." button to select a filename to save to.
7. Click "Save bulk distortion analysis" to write the results to the selected file.

Here is a basic workflow for a capacitance calculation using the GUI:

1. Load CV files using "Load and preview CVs."
2. Choose the capacitance calculation mode: 
    - "At selected voltage"
    - "MinMax Current"
    - "CaRe analysis"
3. A current-difference vs. scan rate plot will be generated, and a linear fit performed. If you select "Force fit through origin", the linear fit will have zero offset.
4. Review and save the calculated capacitance data by clicking "Save Capacitance."

## Dependencies
Please find the required packages listed in the requirements.txt file. 

## Contributing

Contributions are welcome! Feel free to fork the project and submit a pull request or open an issue for suggestions or bug reports.

### Steps to contribute:

1. Fork the repository.
2. Create a new branch (`git checkout -b feature-branch`).
3. Make your changes.
4. Push to the branch (`git push origin feature-branch`).
5. Open a pull request.

## License

This project is licensed under GPL-3.0-or-later.
Copyright (C) 2022-2024  Sebastian Reinke

## Shape-dependent Electrochemistry

Find our work also at https://shape-ec.ruhr-uni-bochum.de/ 

