Project info:
- Project name: sample-langchain
- Project description: A sample project for demonstrating LangChain capabilities
- Project version: 1.0.0
- Author name: Ron Webb
- Author email: ron@ronella.xyz
- Main package: sample

Technology stack:
- Python `^3.13` for the scripting language
- Poetry `2.0` for dependency management

Poetry setup:
- Use the poetry virtual environment for development.
- Use `poetry init` to create the initial project structure and use the <PROJECT_VERSION> as initial version.
- Use `poetry add` to add new dependencies.
- Use `poetry add --dev` to add new development dependencies.

Git setup:
- Create a `.gitignore` file to exclude IDE files, Python cache files, .env, log files, and other unnecessary files. The poetry.lock should not be ignored.
- Create a `.gitattributes` file to prefer the LF for any non-Windows line endings.

Dev dependencies:
- Black for code formatting
- Pylint for linting
- Pytest for testing
- Pytest-Cov for test coverage

Pylint setup: 
- Includes a default `.pylintrc` file for Pylint configuration and must be based on the one from the following gist:
 https://gist.githubusercontent.com/rcw3bb/ae5c1e497e25839dc88a2eba4298f03b/raw/899b2b79ef1b33b692f1901db1064b68b3796db0/.pylintrc.

Logging implementation:
- The logging configuration must be named `logging.ini` in the root of the project. The configuration must match line by line of the following gist:
 https://gist.githubusercontent.com/rcw3bb/236dd009ae22a1a5c3e1b3c2d1298d61/raw/681dc781b64fee81949d427c34a5f9b79105b98b/logging-python.ini
- The name of the log file must be `<MAIN_PACKAGE>.log` and it must be used in the `logging.ini` file.
- A `util` module inside the <MAIN_PACKAGE> must have a `setup_logger` method that initializes a logger and use the `logging.ini`. The implementation must match line by line of the following gist:
 https://gist.githubusercontent.com/rcw3bb/889cab577dccdf129a8e4d2842070ea1/raw/b430690c51b55a6bd76ec4aac74b4a3fe1786a87/logger.py

Project structure:
```
<MAIN_PACKAGE>/__init__.py
tests/__init__.py
.pylintrc
logging.ini
poetry.lock
pyproject.toml
CHANGELOG.md
README.md
```

Coding principles:
- SOLID principles must be followed.
- DRY principle must be followed.
- Prefer composition over inheritance.
- Use dependency injection where applicable.

Coding standards:
- Place all new modules into main package.
- Place all tests in the `tests` package, mirroring the main package structure where possible.
- The test modules should be named `test_*.py` and placed in the same directory as the module they are testing.
- The test packages must mirror the structure of the source code packages.
- Use relative imports within the package.
- Only add comments if the code is not self-explanatory.
- Always add docstrings to all modules, methods and classes. 
- Add author and since using project version for every module.
- For project version not equals to 1.0.0, add author and since if adding a new methods or classes into an existing module.
- Use type hints for all method arguments and outputs.
- Avoid using deprecated types from the `typing` module. Use `collections.abc` instead.
- Use snake_case for methods and variables, PascalCase for classes, and UPPER_CASE for constants.
- Maintain a minimum test coverage of 80% for the project.
- To run tests with coverage and generate an HTML report, use:
  `poetry run pytest --cov=<MAIN_PACKAGE> tests --cov-report html`
- To format and lint the code in one step, use:
  `poetry run black <MAIN_PACKAGE>; poetry run pylint <MAIN_PACKAGE>`
- The linter must always achieve a score of 10/10.
- Always use `.env` for environment variables and load them using `python-dotenv`.
- Private members should be prefixed with an underscore `__`.
- Decompose large methods into smaller, reusable private methods.