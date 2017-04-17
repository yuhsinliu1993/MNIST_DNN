#! /bin/bash

# {reg}_{alpha}_{opt}_{neuron}_{layer}_{learning_rate}
python mnist.py -o adam -l 100 10 -t 0.0001
python mnist.py -o adam -l 256 256 256 10 -t 0.0001
python mnist.py -o adam -l 256 256 256 10 -t 0.00005
python mnist.py -o sgd -l 100 10 -t 0.001
python mnist.py -o sgd -l 256 256 256 10 -t 0.001

python mnist.py -o adam -r l1 -l 100 10 -a 0.001 -t 0.0001
python mnist.py -o adam -r l1 -l 256 256 256 10 -a 0.0005 -t 0.0001
python mnist.py -o adam -r l1 -l 256 256 256 10 -a 0.001 -t 0.0001
python mnist.py -o sgd -r l1 -l 100 10 -a 0.001 -t 0.001
python mnist.py -o sgd -r l1 -l 256 256 256 10 -a 0.001 -t 0.001

python mnist.py -o adam -r l2 -l 100 10 -a 0.001 -t 0.0001
python mnist.py -o adam -r l2 -l 256 256 256 10 -a 0.0005 -t 0.0001
python mnist.py -o adam -r l2 -l 256 256 256 10 -a 0.001 -t 0.0001
python mnist.py -o sgd -r l2 -l 100 10 -a 0.001 -t 0.001
python mnist.py -o sgd -r l2 -l 256 256 256 10 -a 0.001 -t 0.001

# {reg}_{alpha}_{keep_prob}_{opt}_{neuron}_{layer}_{learning_rate}
python mnist.py -o adam -r dropout -l 100 10 -a 0.001 -t 0.0001 -p 0.75
python mnist.py -o adam -r dropout -l 256 256 256 10 -a 0.0005 -t 0.0001 -p 0.75
python mnist.py -o adam -r dropout -l 256 256 256 10 -a 0.001 -t 0.0001 -p 0.2
python mnist.py -o sgd -r dropout -l 100 10 -a 0.001 -t 0.001 -p 0.75
python mnist.py -o sgd -r dropout -l 256 256 256 10 -a 0.001 -t 0.001 -p 0.75
