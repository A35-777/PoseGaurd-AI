# Contributing to PoseGuard AI

Thank you for your interest in improving PoseGuard AI! We welcome bug reports, documentation updates, biomechanical defect algorithms, and performance improvements.

## Code of Conduct
Please review and adhere to [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md) when participating in discussions, reporting issues, or opening pull requests.

## Development Workflow

1. Fork the repository on GitHub.
2. Clone your fork locally:
   ```cmd
   git clone https://github.com/your-username/poseguard-ai.git
   cd poseguard-ai
   ```
3. Create and activate a local virtual environment:
   ```cmd
   python -m venv .venv
   .venv\Scripts\activate
   ```
4. Install development dependencies and editable package:
   ```cmd
   pip install -r requirements-dev.txt
   pip install -e .
   ```
5. Run the test suite:
   ```cmd
   pytest tests/ -v
   ```

## Guidelines for Contributions

- **CPU & Lightweight Focus**: All core runtime inference must run smoothly on standard consumer CPUs without requiring CUDA or cloud GPU infrastructure.
- **Biomechanical Grounding**: Any new defect detection logic should define clear geometric parameters (velocities, thresholds, joint hierarchies) rather than arbitrary heuristic penalization.
- **Testing**: Add or update test cases under `tests/` for any modified parser, detector, or model module.
- **Deterministic Evaluation**: Do not alter default inference thresholds without providing comparative benchmark metrics on the validation suite.

## Submitting Pull Requests

1. Create a descriptive feature branch (`git checkout -b feature/improved-contact-detector`).
2. Commit your changes with clear messages (`git commit -m "Enhance ankle height threshold for contact inconsistency"`).
3. Push to your fork (`git push origin feature/improved-contact-detector`).
4. Open a Pull Request referencing any related issues.

