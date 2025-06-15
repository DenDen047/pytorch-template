# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

For Cursor IDE users, also see `.cursor/rules` files for additional development guidelines.

## 🔨 Most Important Rule - Process for Adding New Rules

When receiving instructions from the user that are deemed necessary to be followed consistently, not just for a single instance:

1. Ask, "Should this be made a standard rule?"
2. If the answer is YES, document it as an additional rule in CLAUDE.md.
3. From then on, apply it as a standard rule in all future interactions.

Through this process, the project's rules are continuously improved.

## Development Environment Setup

### Docker (Recommended)
- Build and run: `cd docker && docker compose up -d --build`
- Access container: `docker exec -it project-core bash`
- GPU-enabled container with CUDA 12.1 support

### Python Environment (uv preferred)
- Setup with uv: `uv python install 3.11 && uv venv --python 3.11 && source .venv/bin/activate`
- Install dependencies: `uv add <package_name>`
- Run scripts: `uv run python script.py`

## Project Architecture

### Directory Structure
- `src/`: Core Python package with utilities and tools
- `conf/`: Configuration files (parameters.yml, secrets.yml)
- `data/`: Follows Kedro data engineering convention with structured layers (01_raw through 08_reporting)
- `notebooks/`: Jupyter notebooks for experiments and analysis
- `docker/`: Containerization setup with GPU support

### Core Utilities (`src/utils/`)
- `camera_geometry.py`: Camera-LiDAR coordinate transformations, 3D-to-2D projections, point cloud processing
- `visualization.py`: Depth image colorization utilities
- `data_loader.py`, `pointcloud_geometry.py`, `plot_style.py`: Additional utilities

### Key Components
- Camera geometry functions support both batched and single tensor operations via `without_batch_dim` parameter
- LiDAR-to-camera projection pipeline: `lidar_to_camera_coordinates()` → `camera_to_image_coordinates()`
- Point cloud filtering and batching utilities for ML workflows
- Rerun integration for 3D visualization of camera-LiDAR projections

## Development Workflow

### Git Workflow
- Follows git-flow model
- New features branch from `develop`
- Archive unused branches as `archive/<branch-name>`
- Use conventional commits (https://www.conventionalcommits.org/)

### Common Commands
- Python execution: `uv run python <script>`
- Docker rebuild: `cd docker && docker compose down && docker compose up -d --build`

## Data Management
Data directory follows Kedro convention:
- `01_raw/`: Original immutable data
- `02_intermediate/` through `07_model_output/`
- `08_reporting/`: Final outputs and visualizations