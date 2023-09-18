package br.com.hastatus.neuralnetwork.network;

import br.com.hastatus.neuralnetwork.layer.NeuralLayer;
import br.com.hastatus.neuralnetwork.layer.NeuralLayerSigmoid;
import br.com.hastatus.neuralnetwork.train.backpropagation.TrainBackPropagation;
import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;


public class NeuralNetworkBasicSigmoid implements NeuralNetwork {
    private static final Logger logger = LogManager.getLogger(NeuralNetworkBasicSigmoid.class);
    private final NeuralLayer[] layers;
    final private int totalLayers;

    public NeuralNetworkBasicSigmoid(int inputSize, int[] hiddenSizes, int outputSize) {
        if(inputSize < 1 || outputSize < 1 || hiddenSizes.length < 1) {
            throw new RuntimeException("invalid");
        }

        this.totalLayers = hiddenSizes.length + 1;//including the last output layer

        this.layers = new NeuralLayer[totalLayers];

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




        logger.info("Total layers: {}", layers.length);
        for(int i = 0; i< layers.length; i++) {
            logger.info("Layer {} neurons: {}", i, layers[i].getTotalNeurons());
        }
    }


    public double[] feedForward(double[] inputs) {
        double[] outputs = null;

        for (NeuralLayer layer : this.layers) {
            outputs = layer.forward(inputs);
            inputs = outputs;
        }

        return outputs;
    }


    /**
     * Trains the neural network using the backpropagation algorithm.
     * This method iteratively adjusts the weights and biases of the neurons in the network based on the provided training inputs and expected outputs. The goal is to minimize the error between the actual output of the network and the expected outputs.
     *
     * @param trainingInputs A 2D array where each row represents a set of inputs to the network.
     * @param expectedOutputs A 2D array where each row represents the expected outputs of the network for the corresponding inputs in the trainingInputs array.
     * @param numEpochs The number of iterations over the entire training dataset to perform. More epochs can allow for more fine-tuned adjustments, but also increases the risk of overfitting.
     * @param learningRate The step size to use when adjusting weights and biases. Smaller values can lead to more precise adjustments but may require more epochs to converge to a solution.
     */
    @Override
    public void train(double[][] trainingInputs, double[][] expectedOutputs, int numEpochs, double learningRate) {

        TrainBackPropagation trainBackPropagation = new TrainBackPropagation(this);
        trainBackPropagation.train(trainingInputs, expectedOutputs, numEpochs, learningRate);
    }

    @Override
    public int getTotalLayers() {
        return totalLayers;
    }

    @Override
    public NeuralLayer getLayer(int index) {
        return layers[index];
    }
}
