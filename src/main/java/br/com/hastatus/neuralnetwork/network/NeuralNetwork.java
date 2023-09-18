package br.com.hastatus.neuralnetwork.network;

import br.com.hastatus.neuralnetwork.layer.NeuralLayer;

public interface NeuralNetwork {

    double[] feedForward(double[] inputs);

    void train(double[][] trainingInputs, double[][] expectedOutputs, int numEpochs, double learningRate);

    int getTotalLayers();

    NeuralLayer getLayer(int index);
}
