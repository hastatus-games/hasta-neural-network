package br.com.hastatus.neuralnetwork.network;

import br.com.hastatus.neuralnetwork.layer.NeuralLayer;
import br.com.hastatus.neuralnetwork.layer.NeuralLayerSigmoid;
import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;

import java.util.ArrayList;
import java.util.List;


public class NeuralNetworkBasicSigmoid implements NeuralNetwork {
    private static final Logger logger = LogManager.getLogger(NeuralNetworkBasicSigmoid.class);

    private final List<NeuralLayer> layers;
    final private int totalLayers;



    public NeuralNetworkBasicSigmoid(int inputSize, int[] hiddenSizes, int outputSize) {
        if(inputSize < 1 || outputSize < 1 || hiddenSizes.length < 1) {
            throw new RuntimeException("invalid");
        }

        this.totalLayers = hiddenSizes.length + 1;//including the last output layer

        this.layers = new ArrayList<>();


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

            this.layers.add(new NeuralLayerSigmoid(neuroniosDaCamada, numInputs));

        }

        //the last layer is always the output
        this.layers.add(new NeuralLayerSigmoid(outputSize, hiddenSizes[hiddenSizes.length - 1]));




        logger.info("Total layers: {}", layers.size());
        for(int i = 0; i< layers.size(); i++) {
            logger.info("Layer {} neurons: {}", i, layers.get(i).getTotalNeurons());
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


    @Override
    public int getTotalLayers() {
        return totalLayers;
    }

    @Override
    public NeuralLayer getLayer(int index) {
        return layers.get(index);
    }
}
