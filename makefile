# Makefile only for clean (for now...)

.PHONY: clean
clean:
	python setup.py clean
	rm -rf simsusy.egg-info htmlcov .coverage simsusy/__pycache__ simsusy/tests/__pycache__