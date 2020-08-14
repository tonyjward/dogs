#!/bin/sh
screen -d -m -S DATABASE python -m populate_database "$1"


