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
# # Sample Test File
# 
# This demonstrates how to create a py:percent jupytext script.

# %%
import gdsfactory as gf
from qpdk import cells

# %% [markdown]
# ## Sample Function
# 
# This function creates a sample component.

# %%
@gf.cell
def sample_test() -> gf.Component:
    """Returns a test component."""
    c = gf.Component()
    wg1 = c << cells.straight(length=1, width=1)
    return c

# %% [markdown]
# ## Main Execution
# 
# When run as a script, this shows the component.

# %%
if __name__ == "__main__":
    from qpdk import PDK
    PDK.activate()
    
    c = sample_test()
    c.show()
    print("Test successful!")