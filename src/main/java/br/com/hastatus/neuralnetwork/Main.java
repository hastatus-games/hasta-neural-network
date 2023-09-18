package br.com.hastatus.neuralnetwork;

import br.com.hastatus.neuralnetwork.base.Neuron;
import br.com.hastatus.neuralnetwork.base.NeuronSigmoid;

public class Main {


    public static void main(String[] args) {
        double[] inputs = {0};
        Neuron neuron = new NeuronSigmoid(1);
        double output = neuron.activate(inputs);
        System.out.printf("Output: %s\n", output);
    }
}
