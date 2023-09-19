package br.com.hastatus.neuralnetwork.train;

public interface NeuralNetworkTrain {

    void train(double[][] trainingInputs, double[][] expectedOutputs, int numEpochs, double learningRate, double meanSquareErrorsAcceptable);


}

