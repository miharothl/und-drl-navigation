test: ## run pytest
	pytest

clean: ## clean
	rm -r _tests	



train: ## train 
	./train.sh &

train-rgb: ## train-rgb
	./train-rgb.sh &


.PHONY: help

help:
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-30s\033[0m %s\n", $$1, $$2}'

.DEFAULT_GOAL := help
