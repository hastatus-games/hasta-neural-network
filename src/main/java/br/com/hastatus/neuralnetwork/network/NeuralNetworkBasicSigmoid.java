package br.com.hastatus.neuralnetwork.network;

import br.com.hastatus.neuralnetwork.base.Neuron;
import br.com.hastatus.neuralnetwork.base.NeuronSigmoid;
import br.com.hastatus.neuralnetwork.layer.NeuralLayer;
import br.com.hastatus.neuralnetwork.layer.NeuralLayerSigmoid;
import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;


public class NeuralNetworkBasicSigmoid implements NeuralNetwork {
    private static final Logger logger = LogManager.getLogger(NeuralNetworkBasicSigmoid.class);
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


        for (int epoch = 0; epoch < numEpochs; epoch++) {

            logger.debug(" *** Training Epoch: {} / {}", epoch, numEpochs-1);

            for (int i = 0; i < trainingInputs.length; i++) {
                // Forward pass
                double[] output = feedForward(trainingInputs[i]);

                double[] expectedOutput = expectedOutputs[i];

                TrainingDeltas trainingDeltas = getTrainingDeltas(trainingInputs[i], output, expectedOutput);
                adjustWeightsAndBias(trainingDeltas, learningRate);


            }
        }
    }


    private TrainingDeltas getTrainingDeltas(double[] inputs, double[] outputs, double[] expected) {
        if(outputs.length!=expected.length) {
            throw new RuntimeException("The number of total values expected cant be different from the network output ");
        }

        TrainingDeltas trainingDeltas = new TrainingDeltas(inputs, layers.length);

        trainingDeltas.initDeltaLayer(layers.length-1, expected.length);

        //iterate over values
        for (int j = 0; j < outputs.length; j++) {

            double error = outputs[j] - expected[j];
            double delta = error * NeuronSigmoid.deriveSigmoid(outputs[j]);

            trainingDeltas.addDelta(layers.length-1, j, delta);
            logger.debug("Output[{}/{}]: {} expected:{} error:{} trainingDelta:{}", j, outputs.length-1, outputs[j], expected[j], error, delta);

        }

        // Calculate deltas for the hidden layers
        for (int j = layers.length - 2; j >= 0; j--) {
            NeuralLayer layer = layers[j];
            NeuralLayer nextLayer = layers[j + 1];
            double[] inputNeuronio = layer.getLastInputs();

            trainingDeltas.initDeltaLayer(j, layer.getTotalNeurons());

            for (int k = 0; k < layer.getTotalNeurons(); k++) {
                double sum = 0;
                for (int l = 0; l < nextLayer.getTotalNeurons(); l++) {
                    sum += nextLayer.getNeuron(l).getWeight(k) * trainingDeltas.getDelta(j+1, l);
                }


                double activation = layer.getNeuron(k).activate(inputNeuronio);
                double delta = sum * NeuronSigmoid.deriveSigmoid(activation);
                trainingDeltas.addDelta(j, k, delta);
            }
        }

        // Update weights and biases
        return trainingDeltas;
    }



    private void adjustWeightsAndBias(TrainingDeltas trainingDeltas, double learningRate) {
        for (int j = 0; j < layers.length; j++) {

            double[] inputsFromLayer;
            if(j == 0) {
                inputsFromLayer = trainingDeltas.getInputs();
            } else {
                inputsFromLayer = layers[j-1].getLastInputs();
            }


            for (int i = 0; i < layers[j].getTotalNeurons(); i++) {
                Neuron neuron = layers[j].getNeuron(i);

                neuron.adjustBias(trainingDeltas.getDelta(j, i) * learningRate);

                for (int l = 0; l < inputsFromLayer.length; l++) {
                    neuron.adjustWeight(l, trainingDeltas.getDelta(j, i) * inputsFromLayer[l] * learningRate);
                }

            }
        }
    }




}
