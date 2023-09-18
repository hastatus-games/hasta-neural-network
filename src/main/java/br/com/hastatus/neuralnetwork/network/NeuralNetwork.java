package br.com.hastatus.neuralnetwork.network;

public interface NeuralNetwork {

    double[] feedForward(double[] inputs);

    void train(double[][] trainingInputs, double[][] expectedOutputs, int numEpochs, double learningRate);

}
