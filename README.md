Creating a project manager to visualize and analyze distortions in videos.<br>

This version uses [uv](https://github.com/astral-sh/uv), a python package installer that replaces pip, pip-tools and parts of virtualenv.<br>

To install uv:<br>
macOS or Linux: curl -LsSf https://astral.sh/uv/install.sh | sh<br>
Windows: irm https://astral.sh/uv/install.ps1 | iex<br>

To set up the environment, use the following commands:<br>
Create a virtual environment:<br>
`uv venv`<br>
Activate the virtual environment:<br>
macOS/Linux: `source .venv/bin/activate`<br>
Windows: `.venv\Scripts\activate`<br>
Sync the environment:<br>
`uv sync`<br>

To run:<br>
uv run -m video_viewer.analysis_platform