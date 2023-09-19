package br.com.hastatus.neuralnetwork.network;

import br.com.hastatus.neuralnetwork.layer.NeuralLayer;

public interface NeuralNetwork {

    double[] feedForward(double[] inputs);


    int getTotalLayers();

    NeuralLayer getLayer(int index);

}
