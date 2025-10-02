# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.17.3
# ---

# %% [markdown]
# # Model Comparison to Qucs-S
#
# This notebook compares the S-parameter models from qpdk (using sax) against reference
# results from Qucs-S simulations. Each test suite validates a different component model.
#
# The comparisons include:
# - Polar plots showing S-parameters in the complex plane
# - Magnitude and phase plots versus frequency
# - Visual validation of model accuracy against reference data

# %% tags=["hide-input", "hide-output"]
import inspect

# Add the tests directory to the path so we can import the test modules
import sys

from matplotlib import pyplot as plt

from qpdk.config import PATH as QPDKPath

test_module_path = QPDKPath.tests / "models"
sys.path.insert(0, str(test_module_path.parent))

from models.test_compare_to_qucs import BaseCompareToQucs  # noqa: E402

# %% [markdown]
# ## Discover Test Suites
#
# Dynamically discover all test suite classes that compare qpdk models to Qucs-S results.
# We find all classes that:
# 1. Are subclasses of `BaseCompareToQucs`
# 2. Are not the base class itself
# 3. Are concrete classes (not abstract)


# %% tags=["hide-input"]
def discover_test_suites() -> list[type[BaseCompareToQucs]]:
    """Discover all test suite classes for Qucs-S comparison.

    Returns:
        List of test suite classes that inherit from BaseCompareToQucs.
    """
    # Import the module to get all classes
    from models import test_compare_to_qucs

    test_suites = []

    # Get all members of the module
    for _name, obj in inspect.getmembers(test_compare_to_qucs):
        # Check if it's a class
        if not inspect.isclass(obj):
            continue

        # Check if it's a subclass of BaseCompareToQucs but not the base class itself
        if not issubclass(obj, BaseCompareToQucs) or obj is BaseCompareToQucs:
            continue

        # Check if it's a concrete class (not abstract)
        if inspect.isabstract(obj):
            continue

        test_suites.append(obj)

    return test_suites


# %%
# Discover all available test suites
test_suites = discover_test_suites()
print(f"Found {len(test_suites)} test suite(s):")
for suite in test_suites:
    print(f"  - {suite.__name__}")

# %% [markdown]
# ## Capacitor Model Comparison
#
# Compare the capacitor S-parameter model against Qucs-S reference data.
# The capacitor is modeled as a lumped element with a given capacitance value of $60\,\mathrm{fF}$.

# %%
# Find and plot the capacitor test suite
capacitor_suite = next(
    (suite for suite in test_suites if "Capacitor" in suite.__name__), None
)

if capacitor_suite:
    plt.figure(figsize=(14, 6))
    test_instance = capacitor_suite()
    test_instance.plot_comparison()
else:
    print("Capacitor test suite not found")

# %% [markdown]
# ## Inductor Model Comparison
#
# Compare the inductor S-parameter model against Qucs-S reference data.
# The inductor is modeled as a lumped element with a given inductance value of $10\,\mathrm{nH}$.

# %%
# Find and plot the inductor test suite
inductor_suite = next(
    (suite for suite in test_suites if "Inductor" in suite.__name__), None
)

if inductor_suite:
    plt.figure(figsize=(14, 6))
    test_instance = inductor_suite()
    test_instance.plot_comparison()
else:
    print("Inductor test suite not found")

# %% [markdown]
# ## Coplanar Waveguide (CPW) Model Comparison
#
# Compare the coplanar waveguide (CPW) transmission line model against Qucs-S reference data.
# The CPW is modeled using scikit-rf media models with specified geometry parameters.

# %%
# Find and plot the CPW test suite
cpw_suite = next((suite for suite in test_suites if "CPW" in suite.__name__), None)

if cpw_suite:
    plt.figure(figsize=(14, 6))
    test_instance = cpw_suite()
    test_instance.plot_comparison()
else:
    print("CPW test suite not found")

# %% [markdown]
# ## Summary
#
# The plots above show comparisons between qpdk models (dashed lines) and Qucs-S
# reference simulations (solid lines) for various passive components:
#
# - **Left plot**: Polar representation showing S-parameters in the complex plane
# - **Right plot**: Magnitude (in dB) and phase (in radians) versus frequency
#
# Good agreement between the models validates the accuracy of the qpdk implementations
# for use in circuit simulations and design optimization.
