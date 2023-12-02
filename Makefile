split_dataset: generate_dataset
	cd workspace/dataset && python split_dataset.py

generate_dataset:
	cd MS-SNSD && python noisyspeech_synthesizer.py
