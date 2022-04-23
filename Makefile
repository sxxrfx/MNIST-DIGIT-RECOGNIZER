APP 			= $(SRC)/app.py
SRC 			= src
PROJECT 		= $(SRC)/project.ipynb  
MODEL_DIR 		= model
MODEL 			= $(MODEL_DIR)/model.json
REQUIREMENTS 	= requirements.txt
VENV 			= venv
PIP 			= $(VENV)/bin/pip
PYTHON 			= $(VENV)/bin/python3
STREAMLIT 		= $(VENV)/bin/streamlit
JUPYTER 		= $(VENV)/bin/jupyter
NBCONVERT 		= $(JUPYTER) nbconvert --execute --to notebook --inplace
DEMO_DIR 		= demo
STREAMLIT_RUN 	= $(STREAMLIT) run
ACTIVATE 		= $(VENV)/bin/activate

SHELL=/bin/bash

ifneq (,$(findstring xterm,${TERM}))
	BLACK        := $(shell tput -Txterm setaf 0)
	RED          := $(shell tput -Txterm setaf 1)
	GREEN        := $(shell tput -Txterm setaf 2)
	YELLOW       := $(shell tput -Txterm setaf 3)
	LIGHTPURPLE  := $(shell tput -Txterm setaf 4)
	PURPLE       := $(shell tput -Txterm setaf 5)
	BLUE         := $(shell tput -Txterm setaf 6)
	WHITE        := $(shell tput -Txterm setaf 7)
	RESET 		 := $(shell tput -Txterm sgr0)
else
	BLACK        := ""
	RED          := ""
	GREEN        := ""
	YELLOW       := ""
	LIGHTPURPLE  := ""
	PURPLE       := ""
	BLUE         := ""
	WHITE        := ""
	RESET        := ""
endif

.PHONY: run all clean model help cleanenv

.DEFAULT_GOAL = help

help:
	@echo "$(LIGHTPURPLE)MNIST DIGIT RECOGNITION APP$(RESET)"
	@echo "$(PURPLE)Sagar Negi$(RESET)"
	@echo ""
	@echo "$(YELLOW)Usage:$(RESET)"
	@echo "    $(BLUE)make$(RESET) $(GREEN)<SUBCOMMAND>$(RESET)"
	@echo ""
	@echo "$(YELLOW)SUBCOMMANDS:$(RESET)"
	@echo "    $(GREEN)setup  $(RESET)    Setup the virtual env."
	@echo "    $(GREEN)model  $(RESET)    Create the model from project.ipynb."
	@echo "    $(GREEN)run    $(RESET)    Run the app."
	@echo "    $(GREEN)help   $(RESET)    Print help information."
	@echo "    $(GREEN)clean  $(RESET)    Deletes Model and cache files."
	@echo "    $(GREEN)cleanenv$(RESET)   Clears out the virtual env."


run: 
	$(STREAMLIT_RUN) $(APP)

model:
	$(NBCONVERT) $(PROJECT)

setup:
	python3 -m venv $(VENV)
	$(PIP) install -r $(REQUIREMENTS)
	mkdir $(MODEL_DIR)
	source $(ACTIVATE)

clean:
	$(RM) $(MODEL)
	

cleanenv:
	$(RM) -r $(VENV)
