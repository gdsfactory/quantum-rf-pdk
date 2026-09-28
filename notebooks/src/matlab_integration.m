%---
%jupyter:
%   # The fences and root keys above sit tight against the `%` on purpose: the
%   # matlab-reflow-comments hook merges adjacent comment lines whose inner
%   # indent is a single space, which would fold `% ---` into `% jupyter:` and
%   # leak this header into the generated notebook as a cell. Inner indents of
%   # zero or of two-or-more spaces are both passed through untouched.
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
%---

% %% [markdown]
%
% # Call qpdk from MATLAB
%
% This notebook demonstrates calling `qpdk` (and through it, `gdsfactory`) **directly from MATLAB**
% using MATLAB's built-in Python interface (`py.module.function(...)`); see [Ways to Call Python
% from MATLAB][call-python-from-matlab].
%
% [call-python-from-matlab]:
% https://se.mathworks.com/help/matlab/matlab_external/ways-to-call-python-from-matlab.html
%
% The notebook itself is written for the **MATLAB Jupyter kernel** provided by
% [jupyter-matlab-proxy](https://github.com/mathworks/jupyter-matlab-proxy); see also the [MathWorks
% Jupyter reference
% architecture](https://se.mathworks.com/products/reference-architectures/jupyter.html).
%
% ## Prerequisites
%
% - A Python environment with `qpdk` installed including the `models` extra (see the
%   {ref}`extras reference <notebook-extras>` for what each extra installs):
%
%   ```bash
%   uv pip install "qpdk[models]"
%   ```
%
%   Until the SAX Touchstone helpers are released upstream, use `uv sync --extra models` from this
%   checkout so the pinned SAX fork commit is installed.
%
% - MATLAB R2024a or newer (older releases may also work with `pyenv`).
%
% - `jupyter-matlab-proxy` installed alongside Jupyter:
%
%   ```bash
%   uv pip install jupyter jupyter-matlab-proxy
%   ```
%
% - The environment variable `QPDK_PYTHON` set to the absolute path of the
%   Python interpreter that has `qpdk` installed. With `uv`:
%
%   ```bash
%   export QPDK_PYTHON=$(uv run python -c 'import sys; print(sys.executable)')
%   ```
%
% - **Optional:** the [RF Toolbox](https://se.mathworks.com/help/rf/index.html) for the
%   S-parameter sections at the end. Everything before them runs on base MATLAB. The RF
%   sections detect the toolbox at runtime and skip themselves if it is missing, and setting
%   `QPDK_SKIP_RF_TOOLBOX=1` skips them unconditionally — which is what the CI job does,
%   because the free MATLAB licence granted to open-source GitHub projects does not
%   necessarily include the RF Toolbox:
%
%   ```bash
%   export QPDK_SKIP_RF_TOOLBOX=1
%   ```
%
% - The version of this notebook rendered in the documentation carries the outputs of a saved run on
%   a MATLAB that *did* include the RF Toolbox, so the RF sections are visible there. CI re-executes
%   the notebook itself on a licence without the toolbox and checks the saved outputs against a
%   fingerprint printed by the last cell, so they cannot silently drift from the source.

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
fprintf('Python %s (%s)\n', string(pe.Version), string(pe.ExecutionMode));

% MATLAB parses ``py.qpdk.PDK`` as a function reference (PDK is a module variable, not a callable),
% so we fetch the attribute explicitly via py.getattr — see the *Limitations to Indexing into Python
% Objects* section of the MATLAB documentation.
qpdk_mod = py.importlib.import_module('qpdk');
PDK = py.getattr(qpdk_mod, 'PDK');
PDK.activate();
fprintf('Activated PDK: %s\n', string(py.getattr(PDK, 'name')));
sax_mod = py.importlib.import_module('sax');

plot_font = 'Inter';
plot_colors = [31 119 180; 255 127 14; 44 160 44; 214 39 40; 148 103 189] / 255;
set(groot, 'defaultFigureColor', 'w', 'defaultAxesColor', 'w', ...
    'defaultAxesFontName', plot_font, 'defaultAxesFontSize', 10, ...
    'defaultAxesLineWidth', 1.2, 'defaultAxesXGrid', 'on', ...
    'defaultAxesYGrid', 'on', 'defaultAxesGridAlpha', 0.3, ...
    'defaultAxesGridLineStyle', '--', 'defaultAxesColorOrder', plot_colors, ...
    'defaultLineLineWidth', 1.5);

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
title('Quarter-wave CPW resonator: f_0(L)', 'FontName', 'Outfit');
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
% ## SAX models as S-parameter boxes in the RF Toolbox
%
% Everything above treats Python as a *function* MATLAB calls. The remaining sections treat a qpdk
% model as a *component*: a SAX model is evaluated in Python, exported as a
% [Touchstone](https://ibis.org/touchstone_ver2.1/touchstone_ver2_1.pdf) file, and then read back
% into MATLAB as an `sparameters` object or an `nport` circuit element. From that point the qpdk
% physics is an ordinary black box inside MATLAB's own RF analysis — it cascades with lumped
% elements in a `circuit`, plots on a Smith chart, gets rational-fitted for a transient response,
% and can be written back out with `rfwrite`.
%
% Touchstone avoids manual complex-array marshalling between the two languages and produces a file
% that other RF tools can read too. SAX's `write_sdict_touchstone` exports model results directly;
% `read_sdict_touchstone` brings a MATLAB-written file back into a SAX S-dictionary.

% %% [markdown]
%
% ### Export two SAX models to Touchstone
%
% Two exports, both driven from MATLAB. The frequency grid is built with `linspace` in MATLAB and
% pushed into Python with `py.numpy.asarray`, so the *same* vector is used to evaluate the model and
% to label the Touchstone file — `write_sdict_touchstone` rejects a mismatch rather than silently
% misaligning the sweep.
%
% - A coplanar-waveguide `straight`, at 1 mm and 2 mm, is a two-port: `.s2p`.
%
% - `quarter_wave_resonator_coupled` is a three-port (through line plus the resonator's own
%   open end): `.s3p`.

% %%
freq_matlab = linspace(4e9, 9e9, 1001);
freq_py = py.numpy.asarray(freq_matlab);

s2p_1mm = fullfile(results_dir, 'cpw_1mm.s2p');
s2p_2mm = fullfile(results_dir, 'cpw_2mm.s2p');
sax_mod.write_sdict_touchstone( ...
    py.qpdk.models.waveguides.straight(pyargs('f', freq_py, 'length', 1000)), ...
    freq_py, s2p_1mm);
sax_mod.write_sdict_touchstone( ...
    py.qpdk.models.waveguides.straight(pyargs('f', freq_py, 'length', 2000)), ...
    freq_py, s2p_2mm);

% The resonator notch is only ~18 MHz wide, so it gets its own finer grid over a narrower span.
freq_res = linspace(6.5e9, 8.0e9, 3001);
freq_res_py = py.numpy.asarray(freq_res);
s3p_res = fullfile(results_dir, 'resonator_coupled.s3p');
sax_mod.write_sdict_touchstone( ...
    py.qpdk.models.resonator.quarter_wave_resonator_coupled(pyargs( ...
        'f', freq_res_py, 'length', 4000, ...
        'coupling_gap', 4.0, 'coupling_straight_length', 400.0)), ...
    freq_res_py, s3p_res);

fprintf('Wrote %s\n', s2p_1mm);
fprintf('Wrote %s\n', s2p_2mm);
fprintf('Wrote %s\n', s3p_res);

% %% [markdown]
%
% ### Is the RF Toolbox available?
%
% The sections below need the RF Toolbox, which the notebook may not have — most notably in CI,
% where the licence granted to open-source GitHub projects is limited. Rather than fail, detect it
% once and guard each subsequent cell with `if has_rf`.
%
% Three independent checks, because they fail in different ways: `QPDK_SKIP_RF_TOOLBOX` is the
% explicit opt-out for CI, `ver`/`license` cover an installation that is absent or unlicensed, and
% the trial `sparameters` call catches the case where the toolbox is installed but a licence cannot
% actually be checked out at run time.

% %%
rf_env = getenv('QPDK_SKIP_RF_TOOLBOX');
rf_skipped_by_env = ~isempty(rf_env) && ...
    ~ismember(lower(string(rf_env)), ["0", "false", "no", ""]);

has_rf = ~rf_skipped_by_env && ~isempty(ver('rf')) && license('test', 'RF_Toolbox') == 1;
if has_rf
    try
        sparameters(zeros(2, 2, 2), [4e9, 8e9]);
    catch
        has_rf = false;
    end
end

if rf_skipped_by_env
    fprintf('RF Toolbox sections skipped: QPDK_SKIP_RF_TOOLBOX=%s\n', rf_env);
elseif ~has_rf
    fprintf('RF Toolbox sections skipped: toolbox not installed or no licence available.\n');
else
    fprintf('RF Toolbox available; running the S-parameter sections.\n');
end

% %% [markdown]
%
% ### The resonator: three ports down to two, and a Smith chart
%
% `quarter_wave_resonator_coupled` exports three ports, but the third is the resonator's own open
% end — it is decoupled from the feedline in this model, so terminating it does not change the
% through response. `snp2smp` performs the reduction, keeping ports 1 and 2 and terminating the rest
% in 50 Ω.
%
% `smithplot` then shows the reflection $S_{11}$ traced on the Smith chart: the loop it makes is the
% resonance, and the notch in $S_{21}$ is the same feature viewed in magnitude.

% %%
if has_rf
    S_res3 = sparameters(s3p_res);
    coupling_to_port3 = max(abs(S_res3.Parameters(1, 3, :)));
    fprintf('Max |S13| = %.2e (port 3 is the decoupled open end)\n', coupling_to_port3);

    s2 = snp2smp(S_res3.Parameters, S_res3.Impedance, [1 2], 50);
    S_res2 = sparameters(s2, S_res3.Frequencies, S_res3.Impedance);

    s21 = rfparam(S_res2, 2, 1);
    [notch_db, idx] = min(20 * log10(abs(s21)));
    fprintf('Notch: %.1f dB at %.4f GHz\n', notch_db, S_res2.Frequencies(idx) / 1e9);

    figure;
    rfplot(S_res2);
    title('Coupled quarter-wave resonator, reduced to two ports', 'FontName', 'Outfit');

    figure;
    smithplot(S_res2, 1, 1);
    title('S_{11} on the Smith chart', 'FontName', 'Outfit');
end

% %% [markdown]
%
% ### Cascading SAX boxes in a MATLAB `circuit`
%
% `nport` wraps a Touchstone file as a circuit element, so a qpdk model can sit inside a MATLAB
% netlist alongside anything else the toolbox provides. Two 1 mm lines in series must reproduce the
% 2 mm line — a cheap end-to-end check that the export, the import, and MATLAB's cascade all agree
% with the analytical model.

% %%
if has_rf
    ckt = circuit('cpw_cascade');
    add(ckt, [1 2], nport(s2p_1mm, 'cpw_a'));
    add(ckt, [2 3], nport(s2p_1mm, 'cpw_b'));
    setports(ckt, [1 0], [3 0]);

    S_cascade = sparameters(ckt, freq_matlab);
    S_direct = sparameters(s2p_2mm);

    err_db = max(abs(20 * log10(abs(rfparam(S_cascade, 2, 1))) - ...
        20 * log10(abs(rfparam(S_direct, 2, 1)))));
    fprintf('1 mm + 1 mm vs. 2 mm: max |S21| difference %.2e dB\n', err_db);

    figure;
    rfplot(S_cascade, 2, 1); hold on;
    rfplot(S_direct, 2, 1); hold off;
    legend('cascade of two 1 mm nports', 'single 2 mm model');
    title('Cascade identity check', 'FontName', 'Outfit');
end

% %% [markdown]
%
% ### A hybrid circuit: SAX physics plus lumped MATLAB elements
%
% The point of the `nport` wrapper is that the qpdk model composes with components qpdk knows
% nothing about. Here the resonator is fed through a coupling capacitor and shunted by an inductor —
% a crude matching network, built entirely with RF Toolbox primitives around the imported box.

% %%
if has_rf
    ckt2 = circuit('resonator_matched');
    add(ckt2, [1 2], capacitor(50e-15, 'c_couple'));
    add(ckt2, [2 3], nport(S_res2, 'sax_resonator'));
    add(ckt2, [3 0], inductor(2.5e-9, 'l_shunt'));
    setports(ckt2, [1 0], [3 0]);

    S_hybrid = sparameters(ckt2, freq_res);

    figure;
    rfplot(S_hybrid, 2, 1); hold on;
    rfplot(S_res2, 2, 1); hold off;
    legend('with matching network', 'bare resonator');
    title('SAX resonator inside a MATLAB circuit', 'FontName', 'Outfit');
end

% %% [markdown]
%
% ### Rational fitting and transient response
%
% `rationalfit` turns the sampled $S_{21}$ into a pole-residue model; `freqresp` checks the fit
% against the sampled data and [`stepresp`](https://www.mathworks.com/help/rf/ref/stepresp.html)
% propagates a step edge through it. The prompt feedline response dominates the raw step plot. The
% resonator also stores energy and re-radiates it into the feedline; after the edge, its smaller
% oscillatory contribution decays on a scale set by the loaded quality factor, approximately
% $\tau_\text{amp} = 2Q_\text{L}/\omega_0$. We subtract the final level and plot the residual
% envelope to make that ring-down visible. This is a qualitative transient of a fit over the
% measured band, not a calibrated DC-to-microwave step response. The decay rate is related to the
% loaded quality factor {cite:p}`gyureTimeDomainBased2015`.

% %%
if has_rf
    [fit_s21, errdb] = rationalfit(S_res2.Frequencies, rfparam(S_res2, 2, 1));
    fprintf('Rational fit: %d poles, error %.2f dB\n', numel(fit_s21.A), errdb);

    resp = freqresp(fit_s21, S_res2.Frequencies);
    figure;
    plot(S_res2.Frequencies / 1e9, 20 * log10(abs(rfparam(S_res2, 2, 1))), 'LineWidth', 1.5);
    hold on;
    plot(S_res2.Frequencies / 1e9, 20 * log10(abs(resp)), '--', 'LineWidth', 1.5);
    hold off; grid on;
    xlabel('Frequency (GHz)'); ylabel('|S_{21}| (dB)');
    legend('SAX model', 'rational fit');
    title('Rational fit of the resonator response', 'FontName', 'Outfit');

    [step_out, t_step] = stepresp(fit_s21, 2e-12, 20000, 20e-12);
    ring = step_out - step_out(end);
    period_samples = max(1, round(1 / (S_res2.Frequencies(idx) * (t_step(2) - t_step(1)))));
    envelope = movmax(abs(ring), period_samples);
    figure;
    plot(t_step * 1e9, envelope); grid on;
    xlim([0.5 35]);
    xlabel('Time (ns)'); ylabel('Residual envelope (unit step)');
    title('Coupled-resonator ring-down after the step edge', 'FontName', 'Outfit');
end

% %% [markdown]
%
% ### Back to Python
%
% `rfwrite` exports a MATLAB `sparameters` object as Touchstone, closing the loop: the hybrid
% circuit — part qpdk model, part MATLAB lumped elements — becomes a file that SAX can pick up
% again. `read_sdict_touchstone` is the inverse of the export above; it returns frequencies in Hz
% and a SAX `SDict`, one entry per ordered port pair, ready to drop into a `sax.circuit` alongside
% purely analytical qpdk models.

% %%
if has_rf
    hybrid_file = fullfile(results_dir, 'resonator_matched.s2p');
    rfwrite(S_hybrid, hybrid_file);

    imported = sax_mod.read_sdict_touchstone(hybrid_file);
    freq_back = double(imported{1});
    sdict_back = imported{2};
    fprintf('Read back into SAX: %d port pairs over %d points (%.2f-%.2f GHz)\n', ...
        int64(py.len(sdict_back)), numel(freq_back), ...
        min(freq_back) / 1e9, max(freq_back) / 1e9);
end

% %% tags=["hide-input", "hide-output"]
%
% The rendered documentation shows the outputs of a saved run of this notebook, so the fingerprint
% below ties them to the source: CI recomputes it from notebooks/src/matlab_integration.m and
% rejects the saved outputs when the two have drifted apart.
try
    fid = fopen(fullfile('notebooks', 'src', 'matlab_integration.m'), 'rb');
    raw = fread(fid, '*uint8');
    fclose(fid);
    digest = py.hashlib.sha256(py.bytes(raw));
    fprintf('Executed source SHA256: %s\n', char(digest.hexdigest()));
catch
    fprintf(['Executed source SHA256: unavailable ', ...
        '(run from the repository root to record it)\n']);
end

% %% [markdown]
%
% ## What's next
%
% - Open the generated GDS files in [KLayout](https://www.klayout.de/) to view
%   the layouts.
%
% - Browse the qpdk [model catalog](all_models.ipynb) for additional
%   analytical models that compose nicely with MATLAB-driven sweeps.
%
% - Export any other qpdk model with `sax.write_sdict_touchstone` and drop it into
%   a MATLAB `circuit` the same way — the pattern is not specific to the CPW line or the
%   resonator used here.

% %% [markdown]
%
% ## References
%
%```{bibliography}
%:filter: docname in docnames
%```
