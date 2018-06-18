# Makefile only for clean (for now...)

.PHONY: clean
clean:
	python setup.py clean
	rm -rf simsusy.egg-info htmlcov .coverage simsusy/__pycache__ simsusy/tests/__pycache__

.PHONY: test unittest typetest formattest
test: unittest typetest formattest

unittest:
	@echo "# Performing the tests..."
	nosetests --config="" --cover-package=simsusy --with-coverage
	@echo

typetest:
	@echo "# Checking typing by mypy..."
	mypy --ignore-missing-imports --follow-imports=silent --no-strict-optional .
	@echo

formattest:
	@echo "# Checking PEP format..."
	flake8 --max-line-length=120 .
	@echo


