# Coding Guidelines

## Always

Reason deeply, critically and iteratively. Take your time. This step is critical to the research project. Support yourself with a detailed review of the `overview.md` and `roadmap.md`. Update the `roadmap.md` to check progress and know the current status of the project development. I am confident that you will do a good job.

## General Development Guidelines

    - Language: Python 3.12+
    - Style: 4-space indentation, 120 chars/line max.
    - Strictness: Strict mode enabled.
    - Comments: Required (inline #, docstrings ''').
    - Packaging: Use miniforge.
    - Config: YAML-based configuration with parameter validation.
    - Implement a chain of though especially before making a major code modification.
    - Don't hallucinate or delete code from the codebase.
    - Persist game plan.
    - Persist learning.
    - Be thorough.
    - Be surgical and laser-focused.
    - Always choose the most straightforward implementation option.
    - Make absolutely sure you do not break existing code.
    - Always verify this by explicitly reason about this aspect before proposing a code change. Always present your explicit reasoning on this.
    - Always reconsider if the codebase actually works by double checking explicitly for logical flaws or forgotten code alignment.
    - Use vectorized operations where possible.
    - Implement parallel processing for simulations.
    - Consider GPU acceleration for learning algorithms.
    - Use a modular approach with clear interfaces between components.
    - Implement a flexible configuration system for experiments.
    - Create well-defined APIs between simulation, agents, and market.

## Code Structure and Organization

    - Project Layout:
        /src             # Source code (Environment Setup section in the `roadmap.md` file)
        /tests           # Test files
        /docs            # Documentation
        /config          # Configuration files
    - Modular Design:
        - Separate files for models, services, controllers, utilities
        - Follow Single Responsibility Principle
        - Implement DRY (Don't Repeat Yourself)

## Naming Conventions

    - Variables/Functions: snake_case
    - Classes/Interfaces: PascalCase
    - Files/Directories: snake_case
    - Constants: UPPERCASE_WITH_UNDERSCORES

## Type System and Documentation

    - Type Hints: Required for all functions and classes
    - Return Types: Must be explicitly declared
    - Docstrings: Google Python Style Guide format
    - Documentation: README.md in each major directory

## Testing Framework

    - Primary: pytest (no unittest)
    - Coverage: Minimum 80%
    - Test Location: ./tests directory
    - Required Fixtures:
        - CaptureFixture
        - FixtureRequest
        - LogCaptureFixture
        - MonkeyPatch
        - MockerFixture
    - Test Files: Must include type annotations and docstrings

## Error Handling and Logging

    - Exception Handling: try-except with specific exceptions
    - Logging Levels: debug/info/warn/error
    - Log Retention: 7 days
    - Context: Capture and log relevant error context
    - Monitoring: Track processing time, accuracy, error rates

## Development Tools

    - Dependency Management: rye or uv
    - Virtual Environments: Required
    - Code Formatting: Black
    - Linting: Ruff
    - Static Type Checking: mypy
    - Security: bandit

## Development Workflow

    - Iterative development with increasing complexity
    - Regular code reviews and refactoring
    - Continuous integration with automated testing
    - Experiment tracking with metadata
    - Documentation updates with each significant change

## Version Control and CI/CD

    - System: Git
    - Commit Style: Conventional Commits
    - Branch Protection: Required reviews
    - Build Checks: All tests must pass

## Configuration Management

    - Environment Variables: Required for secrets and config
    - Config Files: .env with python-dotenv
    - Secrets: Never in code or version control
    - Settings: Hierarchical (default → env → override)

## Specific Requirements

    - Use ray-rllib (preferred) or PyTorch for deep learning and reinforcement learning components
    - Implement vectorized operations where possible for performance
    - Ensure reproducibility through fixed random seeds
    - Create visualization tools for agent behavior and market dynamics
    - Design flexible interfaces for different market clearing mechanisms
    - Build comprehensive logging for experimental analysis
    - Implement metrics for measuring implicit coordination
    - Create tools for policy interpretation and analysis

## AI-Friendly Development Practices

    - Variable Names: Descriptive and meaningful
    - Function Names: Action-oriented and clear
    - Comments: Explain complex logic and reasoning
    - Type Hints: Comprehensive for better code assistance
    - Documentation: Include examples and edge cases
    - Error Messages: Detailed and actionable

## AI Collaboration Guidelines

    - Provide code snippets with comprehensive type hints
    - Include detailed explanations for complex algorithms
    - Focus on maintainability and readability for long-term research
    - Highlight potential bottlenecks or scalability issues
    - Suggest best practices for reinforcement learning implementation
    - Recommend appropriate design patterns for decentralized systems

## Performance Considerations

    - Profile code regularly for bottlenecks
    - Use parallel processing for simulations where appropriate
    - Implement efficient data structures for large state spaces
    - Consider GPU acceleration for neural network training
    - Optimize memory usage for long experiment runs

## Security Guidelines

    - Input Validation: Required for all external data
    - Authentication: Required for protected endpoints
    - Authorization: Role-based access control
    - Data Protection: Encryption at rest and in transit
    - Dependency Scanning: Regular security updates

## Code Review Standards

    - Functionality: Verified against requirements
    - Quality: Meets all style guidelines
    - Security: No obvious vulnerabilities
    - Performance: Within specified limits
    - Documentation: Complete and accurate
