#!/bin/bash

function generator() {
	while read -r ROW; do
		echo "$ROW"
	done <<<$(find ./data -type f | grep -E 'json$')
}
generator | sudo -E /home/bisenbek/.pyenv/versions/meatballai/bin/python ./indxr.py /dev/sdb
