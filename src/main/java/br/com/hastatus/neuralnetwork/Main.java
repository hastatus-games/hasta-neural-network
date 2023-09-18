package br.com.hastatus.neuralnetwork;

import br.com.hastatus.neuralnetwork.network.NeuralNetwork;
import br.com.hastatus.neuralnetwork.network.NeuralNetworkBasicSigmoid;

public class Main {


    public static void main(String[] args) {


        int inputSize = 1;
        int[] hiddenSizes = {10};
        int outputSize = 1;
        NeuralNetwork neuralNetwork = new NeuralNetworkBasicSigmoid(inputSize, hiddenSizes, outputSize);

        double[][] entradasTreinamento = {{0}, {0.1}, {0.2}, {0.3}, {0.4}, {0.5}, {0.6}, {0.7}, {0.8}, {0.9}, {1.0}};
        double[][] resultadosEsperados = {{0}, {0.1}, {0.2}, {0.3}, {0.4}, {0.5}, {0.6}, {0.7}, {0.8}, {0.9}, {1.0}};


        System.out.println("\n\n -------  TRAINing ------ ");


        neuralNetwork.train(entradasTreinamento, resultadosEsperados, 10000, 0.1);


        System.out.println(" -------  end ------ \n\n");

        double[] inputs1 = {0.1};
        double[] outputs1 = neuralNetwork.feedForward(inputs1);
        System.out.println("Output for 0.1: " + outputs1[0]);

        double[] inputs2 = {0.5};
        double[] outputs2 = neuralNetwork.feedForward(inputs2);
        System.out.println("Output for 0.5: " + outputs2[0]);


        double[] inputs3 = {1};
        double[] outputs3 = neuralNetwork.feedForward(inputs3);
        System.out.println("Output for 1: " + outputs3[0]);
    }
}
