package br.com.hastatus.neuralnetwork.layer;

import br.com.hastatus.neuralnetwork.base.Neuron;

public interface NeuralLayer {

    double[] forward(double[] inputs);

    void adjustWeights(double[] inputs, double[] weightDeltas, double learningRate);


    void adjustBias(double[] biasDeltas);

    Neuron getNeuron(int index);

    int getTotalNeurons();

    double[] getLastInputs();
}
