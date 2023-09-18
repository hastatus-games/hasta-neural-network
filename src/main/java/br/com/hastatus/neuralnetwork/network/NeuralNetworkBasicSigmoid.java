package br.com.hastatus.neuralnetwork.network;

import br.com.hastatus.neuralnetwork.layer.NeuralLayer;
import br.com.hastatus.neuralnetwork.layer.NeuralLayerSigmoid;


public class NeuralNetworkBasicSigmoid implements NeuralNetwork {

    private final NeuralLayer[] layers;

    public NeuralNetworkBasicSigmoid(int inputSize, int[] hiddenSizes, int outputSize) {
        if(inputSize < 1 || outputSize < 1 || hiddenSizes.length < 1) {
            throw new RuntimeException("invalid");
        }

        int totalCamadas = hiddenSizes.length + 1;//including the last output layer

        this.layers = new NeuralLayer[totalCamadas];

        for (int i = 0; i < hiddenSizes.length; i++) {

            //the number of inputs is equals to the total number of neurons from previous layer
            int numInputs;
            if(i==0) {
                numInputs = inputSize;
            }
            else {
                numInputs = hiddenSizes[i - 1];
            }

            //the neurons value definied in the previous layer
            int neuroniosDaCamada = hiddenSizes[i];

            this.layers[i] = new NeuralLayerSigmoid(neuroniosDaCamada, numInputs);
        }

        //the last layer is always the output
        this.layers[this.layers.length - 1] = new NeuralLayerSigmoid(outputSize, hiddenSizes[hiddenSizes.length - 1]);




//        System.out.println("Total layers:" + layers.length);
//
//        for(int i = 0; i< layers.length; i++) {
//            System.out.println("Layer "+i+": neurons:" + layers[i].getTotalNeurons());
//        }
    }


    public double[] feedForward(double[] inputs) {
        double[] outputs = null;

        for (NeuralLayer layer : this.layers) {
            outputs = layer.forward(inputs);
            inputs = outputs;
        }

        return outputs;
    }



}
