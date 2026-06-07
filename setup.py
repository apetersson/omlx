# SPDX-License-Identifier: Apache-2.0
"""Setuptools hooks for platform-tagged wheels.

The oMLX package vendors a tiny macOS arm64 DS4 executable.  Keep the Python
code pure, but mark built wheels as platform-specific so installers never treat
that Mach-O binary as portable ``py3-none-any`` data.
"""

from setuptools import setup
from setuptools.command.bdist_wheel import bdist_wheel as _bdist_wheel


class bdist_wheel(_bdist_wheel):
    """Build a platform wheel while keeping the Python/ABI tags generic."""

    def finalize_options(self):
        super().finalize_options()
        self.root_is_pure = False

    def get_tag(self):
        _python, _abi, plat = super().get_tag()
        return "py3", "none", plat


setup(cmdclass={"bdist_wheel": bdist_wheel})
