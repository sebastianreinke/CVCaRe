# CVCaRe

CVCaRe is a desktop application for analysing cyclic voltammograms (CVs), with a particular focus on capacitive CVs. It implements a new method (CaRe) to calculate accurate capacitance and resistance values from a capacitive CV that deviates from a rectangular shape due to the influence of resistance. It reads common potentiostat export formats, selects individual cycles or half-cycles, plots loaded data, estimates capacitance from multiple scan rates, and performs CaRe resistance/capacitance/distortion analysis on complete CVs or forward/reverse scans.

The graphical user interface is built with **PySide6** and **pyqtgraph**. CVCaRe is distributed under the GNU General Public License, version 3 or later.

Please open the user guide (cvcare_user_guide.html) in your browser for more detailed information about the functions of the program.

## Installation and Running

### Use the executable (single-file, click and run)
If you do not wish to use the Python source code, an executable can be downloaded here:
 https://drive.proton.me/urls/A0JBNM4B94#pRIPVwUyEtiD
Select Version 8.0 or later for the new, modern interface. Version 7.2 is still available as an older version. 

### Run from Python source

Requirements:

- Python 3.10 or later is recommended
- A working Python environment with the dependencies in `requirements.txt`

Clone or download the repository, then create and activate a virtual environment:

```bash
git clone https://github.com/sebastianreinke/CVCaRe.git
cd CVCaRe

python -m venv .venv
```

On Windows Command Prompt:

```bat
.venv\Scripts\activate
```

On PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

On macOS or Linux:

```bash
source .venv/bin/activate
```

Install the dependencies:

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Start CVCaRe from the project root:

```bash
python -m cvcare
```

The package entry point is `cvcare/__main__.py`, which starts `cvcare.gui.app.run()`.


## What CVCaRe Does

### Read CV data

CVCaRe accepts text-based potentiostat exports such as `.txt`, `.csv`, `.dat`, and `.tsv` files. The reader is designed to handle common variations between potentiostat programs:

- Semicolon-, tab-, and whitespace-separated values
- Decimal commas and decimal points
- Voltage and current columns in V/mV/µV/nV and A/mA/µA/nA
- Explicit cycle-number columns, where available
- Cycle detection from the voltage trace when a file does not contain a cycle-number column

Voltage and current columns are identified from header text. If a new potentiostat format uses a different header, the corresponding signifier can be added to `cvcare/fileio/signifiers.py`.

### Load full cycles or half-cycles

The sidebar provides three loading modes:

| Mode | Meaning |
|---|---|
| **Full cycles** | Load one complete CV cycle from each selected file |
| **Anodic (forward)** | Load the ascending-potential half-cycle |
| **Cathodic (backward)** | Load the descending-potential half-cycle |

For each file row, enter or select:

- The source file
- The cycle number to use
- An optional evaluation voltage, `U [V]`
- The scan rate, `ν [mV/s]`
- Whether the CV is **active** for plotting and aggregate fits
- Whether the Savitzky–Golay filter should be used by default

If a requested cycle is unavailable, CVCaRe falls back to the highest available cycle and updates the displayed cycle number. Files without an explicit cycle column are handled through voltage-based cycle detection.

### Plot loaded CVs

The plot is always visible in the main window. It supports:

- Independent titles and units for both axes
- Optional Savitzky–Golay filtering for the displayed curves
- Stable colours for each loaded CV
- A plot-level **Isolate** selector for quickly showing one CV without changing which CVs are active for calculations
- CaRe model overlays as dashed curves
- Synthetic half-cycle mirror branches as dash-dot curves
- Shaded directional integration areas

The **active** checkbox is live: unchecking a loaded CV immediately hides its curve and associated overlays. It also excludes that CV from subsequent aggregate calculations. Re-checking it restores the curve without reloading the source file.

## Typical Workflow

### Load and inspect CVs

1. Start CVCaRe.
2. In the left sidebar, choose **Full cycles**, **Anodic (forward)**, or **Cathodic (backward)**.
3. Select one or more CV files using the `...` button in a row. Selecting multiple files populates available rows and creates additional rows when needed.
4. Enter the desired cycle number, scan rate, and—when relevant—evaluation voltage for each CV.
5. Click **Load**.
6. Inspect the curves in the plot. Use **Isolate** when you want to closely inspect one selected CV without altering the active set used for fitting.
7. Toggle **active** to exclude a CV from the plot and from subsequent aggregate fits without re-reading source files.

### Calculate capacitance from several CVs

The **Capacitance** tab performs a linear fit of the selected current quantity against scan rate using active CVs.

1. Load at least two CVs with valid scan rates.
2. Choose the calculation method:
   - `minmax_corrected (recommended)`
   - `minmax`
   - `at_selected_voltage`
3. Choose the cycle segment: `full`, `anodic`, or `cathodic`.
4. For `at_selected_voltage`, enter an evaluation voltage for each applicable CV in the sidebar.
5. Choose whether to force the linear fit through zero.
6. Click **Compute capacitance**.

Before calculating, the tab reads the current sidebar scan rates, evaluation voltages, active flags, and filter flags into the already-loaded dataset. Reloading files is therefore not required after correcting an input value.

A separate **Capacitance fit diagnostic** window opens after a successful calculation. It plots the measured current quantity against scan rate and shows the actual fitted line, allowing you to inspect its quality. Use **Save result** to export the capacitance calculation.

### Perform CaRe analysis on a full CV

The **CaRe** tab evaluates one selected full CV using either an analytical method or an optimisation-enhanced analytical method.

1. Load a CV in **Full cycles** mode.
2. Enter its scan rate in the sidebar.
3. Select the CV number and method in the CaRe tab.
4. Click **Compute for selected CV**.

CVCaRe reports:

- Resistance in Ω
- Capacitance in mF
- Distortion parameter

The reconstructed CaRe model is shown as a dashed curve in the plot, matching the colour of the measured CV. You can export an individual fitted V/I curve or bulk CaRe results for active CVs.

### Work with a half-cycle

The **HalfCV** tab is for datasets loaded in anodic or cathodic half-cycle mode.

- **Show mirrored branch** creates a synthetic opposite branch by rotating the recorded branch by 180° about the midpoint of its endpoints. The synthetic branch is displayed as a dash-dot curve in the same colour as the original half-cycle.
- **Compute CaRe on virtual full CV** joins the recorded half-cycle and its mirrored branch into a virtual closed CV and applies the selected CaRe method.

The result contains information only about the selected measured half-cycle, despite the appearance of a full CV.

### Integrate one scan direction

The **Integral** tab integrates current over a user-specified potential interval for a selected full CV.

1. Enter the CV number.
2. Type the lower and upper voltage bounds directly as plain text. Both `.` and `,` are accepted as decimal separators.
3. Select the anodic/forward or cathodic/backward direction.
4. Click **Compute integral**.

Invalid values are reported when computation is requested without rewriting the text you entered. Inverting the integral bounds changes the sign of the calculated integral.

The selected region between the curve and the zero-current line is shaded in the main plot using a translucent version of that CV's curve colour. **Clear shading** removes the displayed region.

### Export data

The **Export** tab can:

- Write each loaded CV to a separate standardized text file
- Write all loaded CVs side by side to one file
- Split a raw source file into individual cycle columns without first loading it into the application

The CaRe tab can additionally export the reconstructed fit of the selected full CV, while the Capacitance tab can export aggregate fit results.

## Important Notes and Limitations

- CVCaRe is intended for cyclic-voltammetry data and assumes that the selected data represent usable capacitive CVs for the chosen evaluation method.
- The CaRe analysis requires a full CV and a defined scan rate. The HalfCV workflow creates a virtual full CV to enable analysis, but contains only information present in the selected part of the CV.
- The default Savitzky–Golay filter can improve visual inspection and some analyses, but filtering changes the data used by operations that honour the row's filter setting. Compare filtered and unfiltered results where appropriate.
- Cycle detection without a cycle-number column relies on turning points in the voltage trace. Verify the selected cycle and half-cycle visually, especially for incomplete recordings, unusual waveforms, or strongly noisy data. The cycle splitting and HalfCV scan detection logic is robust against many practical hurdles in CV datasets, but it can fail. If you find that your data is handled incorrectly, please help improve CVCaRe and email an example dataset and describe the problem you encountered. 
- Use the visual overlay of the recalculated CV to judge whether the model is appropriate or whether it deviates considerably from the measured data. In the latter case, the obtained values for resistance and capacitance are not reliable.

## Troubleshooting

### A file does not load

- Verify that the file has identifiable voltage and current header columns.
- Leave **Force standard CSV format** unchecked for common potentiostat exports so delimiter detection can run automatically.
- Enable **Force standard CSV format** for plain comma-separated files using decimal points.
- If the file still cannot be read, send a representative data file with a description of the potentiostat software and export settings when reporting the issue.
- You can attempt to quick-fix the problem by identifying your particular current and voltage column headers, and add them in cvcare/fileio/signifiers.py, before running the program from Python source code.

### A calculation reports a missing scan rate

Enter the scan rate in `ν [mV/s]` in the sidebar. The Capacitance, CaRe, and HalfCV tabs reread sidebar settings immediately before calculation, so clicking **Load** again should not be necessary after entering or correcting a scan rate.

### A requested cycle changes after loading

The requested cycle number was not found. CVCaRe selected the highest available cycle and updates the entry to show the cycle actually used.

### The executable starts slowly

A Nuitka one-file executable extracts its bundled runtime before running. Cached one-file builds are normally slower only on their first launch for each application version and user account. A standalone build starts more directly, but requires distributing the complete `.dist` folder.

## Dependencies

The runtime requirements are listed in `requirements.txt`:

```text
PySide6>=6.6,<7
pyqtgraph>=0.13,<1
numpy>=1.24,<3
scipy>=1.10,<2
quantities>=0.16,<1
```

## Reporting Issues and Contributing

If you find bugs or errors in the program, please report them at [cvcare_github@sreinke.slmail.me](mailto:cvcare_github@sreinke.slmail.me). A minimal example CV data file, the loading mode, selected cycle, and scan rate make issues much easier to reproduce.

Contributions are welcome. Please fork the repository, create a feature branch, make and test your changes, then open a pull request or issue describing the change.

## License

This project is licensed under GPL-3.0-or-later.
Copyright (C) 2022-2026 Sebastian Reinke

## Acknowledgement
 
This software was developed in the framework of a fellowship from the Deutsche Bundesstiftung Umwelt (DBU) during PhD research at Ruhr-Universität Bochum (RUB) and continues to be developed at Universität Paderborn (UPB). The author gratefully acknowledges the institutional support provided at both RUB and UPB. The work has been supported by publicly funded research projects.

## Electrochemical Technology

Find our work also at https://chemie.uni-paderborn.de/arbeitskreise/technische-chemie/linnemann
