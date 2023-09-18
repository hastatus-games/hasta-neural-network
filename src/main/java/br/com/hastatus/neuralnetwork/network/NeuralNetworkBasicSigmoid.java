package br.com.hastatus.neuralnetwork.network;

import br.com.hastatus.neuralnetwork.base.Neuron;
import br.com.hastatus.neuralnetwork.base.NeuronSigmoid;
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


    @Override
    public void train(double[][] trainingInputs, double[][] expectedOutputs, int numEpochs, double learningRate) {

        if(trainingInputs.length!=trainingInputs.length) {
            throw new RuntimeException("It is necessary to inform the expected values for training for all outputs");
        }

        for (int epoch = 0; epoch < numEpochs; epoch++) {

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

            System.out.println("Output:"+outputs[j]+" expected:"+expected[j]+" error:"+error+" trainingDelta:"+delta);
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

            double[] inputsDaCamada;
            if(j == 0) {
                inputsDaCamada = trainingDeltas.getEntrada();
            } else {
                inputsDaCamada = layers[j-1].getLastInputs();
            }


            for (int i = 0; i < layers[j].getTotalNeurons(); i++) {
                Neuron neuron = layers[j].getNeuron(i);

                neuron.adjustBias(trainingDeltas.getDelta(j, i) * learningRate);

                for (int l = 0; l < inputsDaCamada.length; l++) {
                    neuron.adjustWeight(l, trainingDeltas.getDelta(j, i) * inputsDaCamada[l] * learningRate);
                }

            }
        }
    }




}
