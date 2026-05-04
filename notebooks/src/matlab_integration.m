% --- jupyter:
%   jupytext:
%     text_representation:
%       extension: .m
%       format_name: percent
%       format_version: '1.3'
%       jupytext_version: 1.19.1
%   kernelspec:
%     display_name: MATLAB Kernel
%     language: matlab
%     name: jupyter_matlab_kernel
% ---

% %% [markdown]
%
% # Call qpdk from MATLAB
%
% This notebook demonstrates calling `qpdk` (and through it, `gdsfactory`) **directly from MATLAB**
% using MATLAB's built-in Python interface (`py.module.function(...)`); see [Ways to Call Python
% from MATLAB](https://se.mathworks.com/help/matlab/matlab_external/ways-to-call-python-from-matlab.
% html).
%
% The notebook itself is written for the **MATLAB Jupyter kernel** provided by
% [jupyter-matlab-proxy](https://github.com/mathworks/jupyter-matlab-proxy); see also the [MathWorks
% Jupyter reference
% architecture](https://se.mathworks.com/products/reference-architectures/jupyter.html).
%
% ## Prerequisites
%
% - A Python environment with `qpdk` installed including the `models` extra:
%   ```bash
%   uv pip install "qpdk[models]"
%   ```
% - MATLAB R2024a or newer (older releases may also work with `pyenv`). - `jupyter-matlab-proxy`
% installed alongside Jupyter:
%   ```bash
%   uv pip install jupyter jupyter-matlab-proxy
%   ```
% - The environment variable `QPDK_PYTHON` set to the absolute path of the
%   Python interpreter that has `qpdk` installed. With `uv`:
%   ```bash
%   export QPDK_PYTHON=$(uv run python -c 'import sys; print(sys.executable)')
%   ```

% %% [markdown]
%
% ## Configure MATLAB's Python interpreter and activate the PDK
%
% MATLAB's `pyenv` selects the Python interpreter used by `py.*` calls.

% %%
qpdk_python = getenv('QPDK_PYTHON');
if isempty(qpdk_python)
    error('matlab_integration:noPython', ...
        'Set the QPDK_PYTHON environment variable to a Python interpreter that has qpdk installed.');
end

pe = pyenv('Version', qpdk_python, 'ExecutionMode', 'OutOfProcess');
disp(pe);

% MATLAB parses ``py.qpdk.PDK`` as a function reference (PDK is a module variable, not a callable),
% so we fetch the attribute explicitly via py.getattr — see the *Limitations to Indexing into Python
% Objects* section of the MATLAB documentation.
qpdk_mod = py.importlib.import_module('qpdk');
PDK = py.getattr(qpdk_mod, 'PDK');
PDK.activate();
fprintf('Activated PDK: %s\n', string(py.getattr(PDK, 'name')));

% %% [markdown]
%
% ## Hello-world: build a coupled resonator and write a GDS
%
% Instantiate `qpdk.cells.resonator_coupled` with default parameters, query its size and ports from
% MATLAB, then write the layout to disk.

% %%
results_dir = fullfile(tempdir, 'qpdk_matlab_demo');
if ~exist(results_dir, 'dir'); mkdir(results_dir); end

component = py.qpdk.cells.resonator_coupled();
gds_path = fullfile(results_dir, 'resonator_coupled.gds');
component.write_gds(gds_path);

size_info = component.size_info;
width_um = double(size_info.width);
height_um = double(size_info.height);

fprintf('Wrote %s\n', gds_path);
fprintf('  size: %.1f x %.1f um\n', width_um, height_um);

% %% [markdown]
%
% ## Frequency sweep using `qpdk.models.resonator.resonator_frequency`
%
% MATLAB drives a `linspace` of resonator lengths, calls the analytical Python model for each value,
% and plots the resulting fundamental frequency. This is the basic pattern of *MATLAB driving the
% parameter sweep, Python providing the physics*.

% %%
lengths_um = linspace(2000, 10000, 81);
freqs_hz = zeros(size(lengths_um));
for k = 1:numel(lengths_um)
    freqs_hz(k) = double(py.qpdk.models.resonator.resonator_frequency( ...
        pyargs('length', lengths_um(k), 'is_quarter_wave', true)));
end

figure;
plot(lengths_um, freqs_hz / 1e9, 'LineWidth', 1.5); grid on;
xlabel('Resonator length (\mum)');
ylabel('Resonance frequency (GHz)');
title('Quarter-wave CPW resonator: f_0(L)');
hold on;
for f = [4 6 8]
    yline(f, '--', sprintf('%d GHz', f));
end
hold off;

% %% [markdown]
%
% ## Inverse design with MATLAB's `fzero`
%
% Use MATLAB's built-in root finder to invert the analytical model: for each target frequency, find
% the resonator length that hits it, then build the corresponding `resonator_coupled` cell and write
% the GDS. Combining MATLAB's optimisation built-ins with `qpdk`'s physics models is the most direct
% value-add of this integration.

% %%
target_ghz = [5, 6, 7];
solved_lengths = zeros(size(target_ghz));
for k = 1:numel(target_ghz)
    f_target_hz = target_ghz(k) * 1e9;
    objective = @(L) double(py.qpdk.models.resonator.resonator_frequency( ...
        pyargs('length', L, 'is_quarter_wave', true))) - f_target_hz;
    solved_lengths(k) = fzero(objective, [500, 50000]);

    fprintf('Target %d GHz -> length %.2f um\n', target_ghz(k), solved_lengths(k));

    component_k = py.qpdk.cells.resonator_coupled( ...
        pyargs('length', solved_lengths(k)));
    out_path = fullfile(results_dir, sprintf('resonator_%dGHz.gds', target_ghz(k)));
    component_k.write_gds(out_path);
end

% %% [markdown]
%
% ## Parametric chip variants
%
% Build a 2-D `ndgrid` of (coupling gap, resonator length) and call
% `qpdk.samples.resonator_test_chip.resonator_test_chip_python` for each combination. Collect
% bounding-box areas in a MATLAB `table` for inspection.

% %%
gaps_um = [12, 16, 20];
res_lengths_um = [3500, 4500];
[GG, LL] = ndgrid(gaps_um, res_lengths_um);
n = numel(GG);
areas_mm2 = zeros(n, 1);
files = strings(n, 1);

for k = 1:n
    chip = py.qpdk.samples.resonator_test_chip.resonator_test_chip_python( ...
        pyargs('coupling_gap', GG(k), 'resonator_length', LL(k)));
    sz = chip.size_info;
    w_um = double(sz.width);
    h_um = double(sz.height);
    areas_mm2(k) = (w_um * h_um) / 1e6;
    files(k) = fullfile(results_dir, sprintf('chip_gap%g_len%g.gds', GG(k), LL(k)));
    chip.write_gds(files(k));
end

T = table(GG(:), LL(:), areas_mm2, files, ...
    'VariableNames', {'coupling_gap_um', 'resonator_length_um', 'area_mm2', 'gds_file'});
disp(T);

% %% [markdown]
%
% ## What's next
%
% - Open the generated GDS files in [KLayout](https://www.klayout.de/) to view
%   the layouts.
% - Browse the qpdk [model catalog](all_models.ipynb) for additional
%   analytical models that compose nicely with MATLAB-driven sweeps.
