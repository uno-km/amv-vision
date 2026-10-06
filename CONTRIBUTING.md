# Contributing to termux-vision

Thank you for your interest in contributing to `termux-vision` (AMEVA-Vision)!

## Core Architectural Principles

1. **Zero-Heavy C++ Dependency**: Do not introduce dependencies that require CMake, OpenCV, or node-gyp builds on mobile devices.
2. **Fail-Closed Parameter Boundaries**: Always validate inputs strictly. Never employ silent dummy fallbacks for null/invalid values.
3. **Dual-Engine Parity**: Any new feature must be implemented and verified in both Python and Node.js.
4. **0-Point Baseline Compliance**: All functions must pass granular audit scoring without regression.

## Development Workflow

1. Fork and clone the repository:
   ```bash
   git clone https://github.com/uno-km/termux-vision.git
   cd termux-vision
   ```

2. Install development dependencies:
   ```bash
   pip install -e .
   pip install pytest
   ```

3. Run automated tests:
   ```bash
   pytest tests/ -v
   node tests/node_smoke.test.js
   ```
