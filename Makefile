# A Self-Documenting Makefile: http://marmelab.com/blog/2016/02/29/auto-documented-makefile.html

PY = python3.6

.PHONY: help check-required-toolset dep-install
.DEFAULT_GOAL := help

check-required-toolset:
	@command -v pip-compile > /dev/null || (echo "Install pip-compile..." && pip3 install pip-compile)

dep-install: check-required-toolset ## install dependencies
	pip-compile requirements.in

help: ## help
	@echo "Shoujen Makefile Tasks list:"
	@awk 'BEGIN {FS = ":.*?## "} /^[a-zA-Z_-]+:.*?## / {sub("\\\\n",sprintf("\n%22c"," "), $$2);printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}' $(MAKEFILE_LIST)
