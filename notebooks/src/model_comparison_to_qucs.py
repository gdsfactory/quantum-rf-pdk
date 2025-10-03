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
import sys

import numpy as np
from IPython.display import Markdown, display

from qpdk.config import PATH as QPDKPath

# Add the tests directory to the path so we can import the test modules
sys.path.insert(0, str(QPDKPath.tests))

from models.test_compare_to_qucs import BaseCompareToQucs

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
        List of test suite classes that inherit from :class:`~BaseCompareToQucs`.
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
    print(f"\t· {suite.__name__}")

# %% [markdown]
# ## Model Comparison
#
# Compare the S-parameter models against Qucs-S reference data.

# %%
# Find and plot all test suites

for suite in test_suites:
    test_instance = suite()
    display(Markdown(f"### {test_instance.component_name}"))
    display(Markdown(f"**Test Suite:** `{suite.__name__}`"))
    display(
        Markdown(
            f"**Parameter:** {test_instance.parameter_name} = {test_instance.parameter_value / test_instance.parameter_unit:.2f} × 10^{int(np.log10(test_instance.parameter_unit))}"
        )
    )
    display(Markdown(f"**CSV:** `{test_instance.csv_filename}`"))
    test_instance.plot_comparison()

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
